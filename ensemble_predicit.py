import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import pickle
import re
import os
from comet_ml import Experiment
from typing import Any, Dict, Optional
import csv

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
from pytorch_lightning.loggers import CometLogger, TensorBoardLogger
from torchmetrics.functional import accuracy, f1_score, roc, precision, recall, confusion_matrix
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from pytorch_lightning.callbacks import EarlyStopping

from peft import LoraConfig, TaskType
from peft import get_peft_model

import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas

# Visualisation
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc

tqdm.pandas()

from arabert.preprocess import ArabertPreprocessor

import config
from dataset_multitask import BlindTweetEmotionDataset, BlindTweetDataModule, load_blind_dataset
from model_multitask import TweetPredictor
from utils.prediction_utils_multitask import get_predictions, predict, get_predictions_blind
from utils.save_result_multitask import save_pred_gt, log_results
from utils.MyStanceEval import log_StanceEval

pl.seed_everything(42)


def save_pred_gt_blind(  Comb_name,Ex_name,texts, predictions, texts_id,target):
    print(texts)

    mapping = {0:'NONE', 1:'FAVOR', 2:'AGAINST'}
    data_pred_1 = [mapping[t.item()] for t in predictions]
    folder_path = "/home/dr-nfs/m.badran/mawqif/results/predictions/"+Comb_name
    folder_path1 ='-'.join((folder_path+"/"+Ex_name).split("-")[:-1]) 
    file_path_pred_File = folder_path1+ ".csv"
    print(file_path_pred_File,"*"*19)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    with open(file_path_pred_File, 'a', newline='', encoding='utf-8') as output_file:
        csv_writer = csv.writer(output_file)

        if output_file.tell() == 0:
            csv_writer.writerow(['ID', 'target', 'text', 'Stance'])

        for i in range(len(data_pred_1)):
            # parts = line.strip().split('\t')
            csv_writer.writerow([texts_id[i], target, texts[i], data_pred_1[i]])


if __name__ == '__main__': 
  os.environ["TOKENIZERS_PARALLELISM"] = "false"

  config_dict = {attr: getattr(config, attr) for attr in dir(config) if not attr.startswith('__')}
  print(config_dict)
  ensemble_versions = [ "V101.20","V101.13","V101.15","V101.3","V101.5"]
  ensemble_versions = [ "V101.20","V101.13"]

  Comb_name = "".join(ensemble_versions)
  print(Comb_name)
  config.USE_ARABERT_PRE = True ##True for 101, False for 104
  config.selectedModels= {'Women empowerment':"aubmindlab/bert-base-arabertv02-twitter",'Covid Vaccine':'moha/arabert_c19', 'Digital Transformation':"aubmindlab/bert-base-arabertv02-twitter"}
  config_dict['WEIGHTED_LOSS'] = False
  for v in ensemble_versions:
    config.Version = v
    for selectedTarget in config.selectedTarget:        
        if not config.selectedModels is None:     
            config.selectedModel = config.selectedModels[selectedTarget]
            config_dict['selectedModel']=config.selectedModels[selectedTarget]
            print('selected model:',config.selectedModel)

        Ex_name =  config.Version+"-"+selectedTarget.replace(" ","")
        arabert_prep = ArabertPreprocessor(model_name=config.selectedModel) if config.USE_ARABERT_PRE else None
        tokenizer = AutoTokenizer.from_pretrained(config.selectedModel)
        model = AutoModel.from_pretrained(config.selectedModel)

        test_df = load_blind_dataset(config.BlindTestData_name,selectedTarget, arabert_prep = arabert_prep )

        data_module = BlindTweetDataModule( test_df, tokenizer, batch_size=config.BATCH_SIZE, token_len=config.MAX_TOKEN_COUNT)
        data_module.setup()

        for batch in data_module.test_dataloader():
            assert len(batch) == 4
            assert batch["input_ids"].shape == batch["attention_mask"].shape
            # assert batch["input_ids"].shape[0] == config.BATCH_SIZE
            assert batch["input_ids"].shape[1] == config.MAX_TOKEN_COUNT
            # assert batch["tweet_id"].shape[0] == config.BATCH_SIZE
            # print(batch)
            break

        model = TweetPredictor(n_classes=3, steps_per_epoch=len(test_df) // config.BATCH_SIZE, config = config_dict )

        logger = None

        trained_model = model.load_from_checkpoint("final_checkpoints/"+Ex_name+"/best-checkpoint.ckpt")
        trained_model.freeze()

        texts, predictions, texts_id = get_predictions_blind(trained_model, data_module.get_test_dataset())

        save_pred_gt_blind(  Comb_name,Ex_name,texts, predictions, texts_id,selectedTarget)

        # log_results(Ex_name, trained_model, logger, selectedTarget, data_module.get_test_dataset(), "test")
        
        del trained_model, model, data_module
        



csv_files = [file for file in os.listdir("/home/dr-nfs/m.badran/mawqif/results/predictions/"+Comb_name) if file.endswith('.csv')]

print(csv_files)

combined_df =  None

for file in csv_files:
    if 'combined_data' in file:
        continue
    df = pd.read_csv("/home/dr-nfs/m.badran/mawqif/results/predictions/"+Comb_name+"/"+file)

    df = df.rename(columns={'Stance': f'Stance_{file[:-4]}'})
    print(f'Stance_{file[:-4]}')
    df = df.drop(columns=['text'])  
    if combined_df is None:
        combined_df = df
    else:
        # print(combined_df.columns,df.columns)
        combined_df = pd.merge(combined_df, df, on=['ID', 'target'], how='outer',)

df1 = pd.read_csv(config.BlindTestData_name)
# df1 = df1.drop(df1.columns.difference(['ID','target','text']), axis=1)

combined_df = pd.merge(combined_df, df1, on=['ID', 'target'], how='outer')

combined_df.to_csv("/home/dr-nfs/m.badran/mawqif/results/predictions/"+Comb_name+'/combined_data.csv', index=False)

def find_max(row):
    # List = [row['Stance1'],row['Stance2'], row['Stance3'], row['Stance4'], row['Stance5']]
    # List = [row['Stance1'],row['Stance2'], row['Stance3'], row['Stance4'], row['Stance5'], row['Stance6'], row['Stance7']]#7
    List = [row['Stance1'],row['Stance2']]
    # print(List, max((List), key = List.count))
    return max((List), key = List.count)

df = pd.read_csv("/home/dr-nfs/m.badran/mawqif/results/predictions/"+Comb_name+'/combined_data.csv')

results = [f for f in df.columns if 'Stance_' in f ]
print(results,len(results))
for i in range(len(results)):
    df['Stance'+str(i+1)] = df[results[i]].copy()
df['Label_ensemble'] = df.apply(find_max, axis=1)

with open("/home/dr-nfs/m.badran/mawqif/results/predictions/"+Comb_name+'/Results.csv', 'w', newline='', encoding='utf-8') as output_file:
    csv_writer = csv.writer(output_file)

    csv_writer.writerow(['ID', 'target', 'text', 'Stance'])

    for index, row in df.iterrows():
        # parts = line.strip().split('\t')
        csv_writer.writerow([row['ID'], row['target'], row['text'], row['Label_ensemble']])
