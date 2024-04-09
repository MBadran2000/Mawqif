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

# Visualisation
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc



# pl.seed_everything(42)

def load_dataset(TrainData_name,TestData_name,selectedTarget,Apply_Weight_loss = False):
    df = pd.read_csv(TrainData_name)
    test_df = pd.read_csv(TestData_name)

    df= df[df['target'] == selectedTarget]
    test_df= test_df[test_df['target'] == selectedTarget]
    df = df[['ID','text','stance','target']]
    test_df = test_df[['ID','text','stance','target']]

    df['stance'] = df['stance'].fillna(value="None")

    ### prompt
    mapping_t = { 'Covid Vaccine': ' تطعيم كورونا ', 'Women empowerment': ' تمكين المرأة ', 'Digital Transformation': ' التحول الرقمي '}
    target = df['target'].copy()
    target = target.apply(lambda x: mapping_t[x])
    df['text']=df['text']+'. موقف الكاتب من' +target + 'هو'
    target = test_df['target'].copy()
    target = target.apply(lambda x: mapping_t[x])
    test_df['text']=test_df['text']+'. موقف الكاتب من' +target + 'هو'
    df=df.drop('target', axis=1)
    test_df=test_df.drop('target', axis=1)
    ##

    df[df["stance"].isna()]
    test_df['stance'] = test_df['stance'].fillna(value="None")
    test_df[test_df["stance"].isna()]

    mapping = {'None': 0, 'Favor': 1, 'Against': 2}
    df['stance'] = df['stance'].apply(lambda x: mapping[x])
    test_df['stance'] = test_df['stance'].apply(lambda x: mapping[x])
    train_df, val_df = train_test_split(df, test_size=0.18, stratify=df['stance'],random_state=42)
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
    class_weights = class_weights / class_weights.sum()

    print("Class Weights:", class_weights)

    return train_df, val_df, test_df,class_weights


class TweetEmotionDataset(Dataset):

  def __init__(
      self, 
      data: pd.DataFrame, 
      tokenizer: AutoTokenizer,
      text_preprocessor=None,
      max_token_len: int = 128   
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
      labels = torch.tensor(data_row.stance, dtype=torch.long)
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
      token_len=128 ## This default value
    ):
    super().__init__()
    self.batch_size = batch_size
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df
    self.tokenizer = tokenizer
    self.text_preprocessor = text_preprocessor
    self.token_len = token_len

  ## setup function for loading the train and test sets
  def setup(self, stage=None):
    self.train_dataset = TweetEmotionDataset(
      self.train_df, 
      self.tokenizer,
      #self.label_columns,
      self.text_preprocessor,
      self.token_len
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
    return DataLoader(
      self.train_dataset,
      batch_size=self.batch_size,
      shuffle=True, #########
      num_workers=os.cpu_count() # or num_workers=4 
    )


  def val_dataloader(self):
    return DataLoader(self.val_dataset, batch_size=1, num_workers=os.cpu_count()) # or num_workers=4 

  def test_dataloader(self):
    return DataLoader(self.test_dataset, batch_size=1, num_workers=os.cpu_count()) # or num_workers=4 