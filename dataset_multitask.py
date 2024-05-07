import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import pickle
import re
import os

# Pytorch 
import torch
import torchmetrics
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.functional import accuracy, f1_score, roc, precision, recall, confusion_matrix
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from torch.utils.data import WeightedRandomSampler

# Visualisation
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc



# pl.seed_everything(42)
def remove_english(text):
    return re.sub(r'[a-zA-Z]', '', text) #v3.1
    # return re.sub(r'[a-zA-Z@0-9]', '', text) #v3.2

def add_words_to_string(input_string, words_at_start, words_at_end):
    words = input_string.split()
    words = words_at_start + words
    words.extend(words_at_end)
    return ' '.join(words)


def load_dataset(TrainData_name,TestData_name,selectedTarget,Apply_Weight_loss = False,arabert_prep = None,random_state=42):
    df = pd.read_csv(TrainData_name)
    test_df = pd.read_csv(TestData_name)

    if not arabert_prep is None:
      print("applying arabert preprocess")
      df['text'] = df['text'].apply(lambda text: arabert_prep.preprocess(text))
      test_df['text'] = test_df['text'].apply(lambda text: arabert_prep.preprocess(text))


    df= df[df['target'] == selectedTarget]
    test_df= test_df[test_df['target'] == selectedTarget]
    df = df[['ID','text','stance','sarcasm','sentiment']]
    test_df = test_df[['ID','text','stance','sarcasm','sentiment']]

    df['stance'] = df['stance'].fillna(value="None")
    df[df["stance"].isna()]
    test_df['stance'] = test_df['stance'].fillna(value="None")
    test_df[test_df["stance"].isna()]
    
    mapping = { 'Yes': 0, 'No': 1}
    df['sarcasm'] = df['sarcasm'].apply(lambda x: mapping[x])
    test_df['sarcasm'] = test_df['sarcasm'].apply(lambda x: mapping[x])
    mapping = {'Neutral': 0, 'Positive': 1, 'Negative': 2}
    df['sentiment'] = df['sentiment'].apply(lambda x: mapping[x])
    test_df['sentiment'] = test_df['sentiment'].apply(lambda x: mapping[x])

    # df['sarcasm'] = df['sarcasm'].fillna(value="None")
    # df['sentiment'] = df['sentiment'].fillna(value="None")
    # test_df['sarcasm'] = test_df['sarcasm'].fillna(value="None")
    # test_df['sentiment'] = test_df['sentiment'].fillna(value="None")

    mapping = {'None': 0, 'Favor': 1, 'Against': 2}
    df['stance'] = df['stance'].apply(lambda x: mapping[x])
    test_df['stance'] = test_df['stance'].apply(lambda x: mapping[x])
    train_df, val_df = train_test_split(df, test_size=0.18, stratify=df['stance'],random_state=random_state)
    # train_df, val_df = train_test_split(df, test_size=0.05, stratify=df['stance'],random_state=42)

    print(train_df.head())
    if not Apply_Weight_loss:
      return train_df, val_df, test_df, None

    targets = torch.tensor(np.array(train_df['stance']))
    # Calculate class frequencies
    class_counts = torch.bincount(targets)
    # Calculate inverse class frequencies
    total_samples = class_counts.sum().float()
    class_frequencies = class_counts / total_samples
    # Calculate inverse class weights
    class_weights = 1.0 / class_frequencies

    ## decrease importance of None 
    # class_weights[0] = (class_weights[1]+class_weights[2]) /2  #v1.1
    # class_weights[0] = min(class_weights[1],class_weights[2])   #v1.2

    # Normalize weights
    class_weights_STA = class_weights / class_weights.sum()


    targets = torch.tensor(np.array(train_df['sentiment']))
    class_counts = torch.bincount(targets)
    total_samples = class_counts.sum().float()
    class_frequencies = class_counts / total_samples
    class_weights = 1.0 / class_frequencies
    class_weights_SENT = class_weights / class_weights.sum()

    targets = torch.tensor(np.array(train_df['sarcasm']))
    class_counts = torch.bincount(targets)
    total_samples = class_counts.sum().float()
    class_frequencies = class_counts / total_samples
    class_weights = 1.0 / class_frequencies
    class_weights_SAR = class_weights / class_weights.sum()


    print("Class Weights:", class_weights_STA,class_weights_SENT,class_weights_SAR)

    return train_df, val_df, test_df,{"STA":class_weights_STA,"SENT":class_weights_SENT,"SAR":class_weights_SAR}



class TweetEmotionDataset(Dataset):

  def __init__(
      self, 
      data: pd.DataFrame, 
      tokenizer: AutoTokenizer,
      text_preprocessor=None,
      max_token_len: int = 128,
      nlp_aug = None   
    ):
    self.data = data
    self.tokenizer = tokenizer
    self.text_preprocessor = text_preprocessor
    self.max_token_len = max_token_len
    self.nlp_aug = nlp_aug
    
  def __len__(self):
    return len(self.data)

  def __getitem__(self, index: int):
    data_row = self.data.iloc[index]

    text = data_row.text

    if self.nlp_aug is not None:
      for aug in self.nlp_aug:
        text = (aug).augment(text)[0]
    if self.text_preprocessor is not None:
      text = self.text_preprocessor.preprocess(text)

    encoding = self.tokenizer.encode_plus(
      text,
      add_special_tokens=True,
      max_length=self.max_token_len,
      return_token_type_ids=False,
      padding="max_length",
      truncation=True,
      return_attention_mask=True, # to make sure each sequence is maximum of max length
      return_tensors='pt', #return it as pytorch
    )

    return dict(
      text=text,
      input_ids=encoding["input_ids"].flatten(), ##we use flatten to remove X dimension
      attention_mask=encoding["attention_mask"].flatten(),
      labels = dict(STA=torch.tensor(data_row.stance, dtype=torch.long),
                SAR=torch.tensor(data_row.sarcasm, dtype=torch.long),
                SENT=torch.tensor(data_row.sentiment, dtype=torch.long))
    )


class TweetDataModule(pl.LightningDataModule):
  def __init__(
      self, 
      train_df, 
      val_df,
      test_df,
      tokenizer, 
      text_preprocessor=None,
      batch_size=8, ## This default value
      token_len=128, ## This default value
      class_weights= None, 
      weighted_sampler = False,
      nlp_aug = None
    ):
    super().__init__()
    self.batch_size = batch_size
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df
    self.tokenizer = tokenizer
    self.text_preprocessor = text_preprocessor
    self.token_len = token_len
    self.class_weights = class_weights
    self.weighted_sampler = weighted_sampler
    self.nlp_aug = nlp_aug

  ## setup function for loading the train and test sets
  def setup(self, stage=None):
    self.train_dataset = TweetEmotionDataset(
      self.train_df, 
      self.tokenizer,
      #self.label_columns,
      self.text_preprocessor,
      self.token_len,
      nlp_aug = self.nlp_aug
    )
    assert len(self.train_dataset) == len(self.train_df), "data missing, check TweetEmotionDataset"

    self.val_dataset = TweetEmotionDataset(
      self.val_df, 
      self.tokenizer,
      self.text_preprocessor,
      self.token_len
    )
    assert len(self.val_dataset) == len(self.val_df), "data missing, check TweetEmotionDataset"

    self.test_dataset = TweetEmotionDataset(
      self.test_df,
      self.tokenizer,
      self.text_preprocessor,
      self.token_len
    )
    assert len(self.test_dataset) == len(self.test_df), "data missing, check TweetEmotionDataset"
  
  def get_train_dataset(self):
    return self.train_dataset 

  def get_val_dataset(self):
    return self.val_dataset

  def get_test_dataset(self):
    return self.test_dataset

  def train_dataloader(self):
    sampler = None
    shuffle=True, #########

    if self.weighted_sampler: 
      # print(self.train_dataset['labels'])
      sample_weights = [self.class_weights['STA'][i['labels']['STA'].item()].item() for i in self.train_dataset]
      # print('sample_weights',sample_weights[:5])
      sampler = WeightedRandomSampler(weights=sample_weights,replacement = True,num_samples=len(self.train_dataset))
      shuffle= False
    return DataLoader(
      self.train_dataset,
      batch_size=self.batch_size,
      shuffle=shuffle, #########
      num_workers=os.cpu_count(), # or num_workers=4 
      sampler = sampler,
    )


  def val_dataloader(self):
    return DataLoader(self.val_dataset, batch_size=1, num_workers=os.cpu_count()) # or num_workers=4 

  def test_dataloader(self):
    return DataLoader(self.test_dataset, batch_size=1, num_workers=os.cpu_count()) # or num_workers=4 







def load_blind_dataset(TestData_name,selectedTarget,Apply_Weight_loss = False,arabert_prep = None,random_state=42):
    test_df = pd.read_csv(TestData_name)

    if not arabert_prep is None:
      print("applying arabert preprocess")
      test_df['text'] = test_df['text'].apply(lambda text: arabert_prep.preprocess(text))

    test_df= test_df[test_df['target'] == selectedTarget]
    test_df = test_df[['ID','text']]


    print(test_df.head())

    return test_df


class BlindTweetEmotionDataset(Dataset):

  def __init__(
      self, 
      data: pd.DataFrame, 
      tokenizer: AutoTokenizer,
      text_preprocessor=None,
      max_token_len: int = 128,
    ):
    self.data = data
    self.tokenizer = tokenizer
    self.text_preprocessor = text_preprocessor
    self.max_token_len = max_token_len
    
  def __len__(self):
    return len(self.data)

  def __getitem__(self, index: int):
    data_row = self.data.iloc[index]

    text = data_row.text

    if self.text_preprocessor is not None:
      text = self.text_preprocessor.preprocess(text)

    encoding = self.tokenizer.encode_plus(
      text,
      add_special_tokens=True,
      max_length=self.max_token_len,
      return_token_type_ids=False,
      padding="max_length",
      truncation=True,
      return_attention_mask=True, # to make sure each sequence is maximum of max length
      return_tensors='pt', #return it as pytorch
    )

    return dict(
      text=text,
      input_ids=encoding["input_ids"].flatten(), ##we use flatten to remove X dimension
      attention_mask=encoding["attention_mask"].flatten(),
      tweet_id = data_row.ID
    )


class BlindTweetDataModule(pl.LightningDataModule):
  def __init__(
      self, 
      test_df,
      tokenizer, 
      text_preprocessor=None,
      batch_size=8, ## This default value
      token_len=128, ## This default value
    ):
    super().__init__()
    self.batch_size = batch_size
    self.test_df = test_df
    self.tokenizer = tokenizer
    self.text_preprocessor = text_preprocessor
    self.token_len = token_len

  ## setup function for loading the train and test sets
  def setup(self, stage=None):
    self.test_dataset = BlindTweetEmotionDataset(
      self.test_df,
      self.tokenizer,
      self.text_preprocessor,
      self.token_len
    )
    assert len(self.test_dataset) == len(self.test_df), "data missing, check TweetEmotionDataset"
  
  def get_test_dataset(self):
    return self.test_dataset

  def test_dataloader(self):
    return DataLoader(self.test_dataset, batch_size=32, num_workers=os.cpu_count()) # or num_workers=4 


