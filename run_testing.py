import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import pickle
import re
import os
from comet_ml import Experiment
from typing import Any, Dict, Optional

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
from dataset import TweetEmotionDataset, TweetDataModule, load_dataset
from model import TweetPredictor
from utils.prediction_utils import get_predictions, predict
from utils.save_result import save_pred_gt, log_results_test
from utils.MyStanceEval import log_StanceEval

pl.seed_everything(42)

def run(): 
  os.environ["TOKENIZERS_PARALLELISM"] = "false"
  Ex_name =  config.selectedModel.split('/')[-1]+"-"+config.Version+"-Test"

  config_dict = {attr: getattr(config, attr) for attr in dir(config) if not attr.startswith('__')}

  logger = CometLogger(
    experiment_name=Ex_name ,
    api_key='jFe8sB6aGNDL7p2LHe9V99VNy',
    workspace='mbadran2000',  # Optional
    #  save_dir='lightning_logs',  # Optional
    project_name='Mawqif',  # Optional
    )

  for selectedTarget in config.selectedTarget:
    Ex_name1 =  config.selectedModel.split('/')[-1]+"-"+config.Version+"-"+selectedTarget.replace(" ","")
    arabert_prep = ArabertPreprocessor(model_name=config.selectedModel) if config.USE_ARABERT_PRE else None
    tokenizer = AutoTokenizer.from_pretrained(config.selectedModel)
    model = AutoModel.from_pretrained(config.selectedModel)

    train_df, val_df, test_df,class_weights = load_dataset(config.TrainData_name,config.TestData_name,selectedTarget,config.WEIGHTED_LOSS or config.WEIGHTED_SAMPLER, arabert_prep = arabert_prep )

    data_module = TweetDataModule(train_df, val_df, test_df, tokenizer, batch_size=config.BATCH_SIZE, token_len=config.MAX_TOKEN_COUNT,class_weights= class_weights, weighted_sampler = config.WEIGHTED_SAMPLER, nlp_aug = None)
    data_module.setup()

    for batch in data_module.train_dataloader():
      assert len(batch) == 4
      assert batch["input_ids"].shape == batch["attention_mask"].shape
      assert batch["input_ids"].shape[0] == config.BATCH_SIZE
      assert batch["input_ids"].shape[1] == config.MAX_TOKEN_COUNT
      assert batch["labels"].shape[0] == config.BATCH_SIZE
      # print(batch)
      break

    print(config_dict)
    trained_model = TweetPredictor(n_classes=3, steps_per_epoch=len(train_df) // config.BATCH_SIZE,class_weights=class_weights, config = config_dict )
    # trained_model = trained_model.load_from_checkpoint("checkpoints/"+Ex_name1+"/best-checkpoint.ckpt")

    checkpoint = torch.load("checkpoints/"+Ex_name1+"/best-checkpoint.ckpt")
    # print(checkpoint.keys())
    # trained_model.load_state_dict(checkpoint['state_dict'])

    checkpoint = {key: value for key, value in checkpoint['state_dict'].items() if key.startswith("bert") or key.startswith("classifier") }
    trained_model.load_state_dict(checkpoint)


    trained_model.freeze()


    log_results_test(Ex_name1, trained_model, logger, selectedTarget, data_module.get_test_dataset(), "test")
    
    del trained_model, model, data_module
    
      # exit()
    # config.selectedModel.split('/')[-1]+"-"+selectedTarget.replace(" ","")
  log_StanceEval(Ex_name,logger,"test")
  del logger



if __name__ == '__main__': 
  v = [ 'V6.2' ,'V7.021','V7.4']
  #V1.2,V1_WL,"V1.6","V6.1","V8.013","V7.2","V10.01","V10.23" 'V10.03', 'V10.21', 
  for i in v: 
      config.Version = str(i)
      print(config.Version)
      run()
