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
from utils.save_result import save_pred_gt, log_results
from utils.MyStanceEval import log_StanceEval

pl.seed_everything(42)

if __name__ == '__main__': 
  os.environ["TOKENIZERS_PARALLELISM"] = "false"

  config_dict = {attr: getattr(config, attr) for attr in dir(config) if not attr.startswith('__')}
  print(config_dict)

  if config.USE_NLPAUG:
    # nlp_aug = naw.SynonymAug(aug_src='ppdb', lang='arb',model_path ='/home/dr-nfs/m.badran/mawqif/ppdb-1.0-l-lexical',aug_p=config.NLPAUG_PROB)
    # nlp_aug = naw.RandomWordAug(action='delete', aug_p=config.NLPAUG_PROB)
    # nlp_aug = [naw.RandomWordAug(action='swap', aug_p=config.NLPAUG_PROB)]
    nlp_aug = [naw.RandomWordAug(action='delete', aug_p=config.NLPAUG_PROB,stopwords=['تطعيم','تطعيم','المرأة','تمكين','الرقمي','التحول','تغريدة',':','.','موقف','التغريدة','هو','من']),naw.SynonymAug(aug_src='ppdb', lang='arb',model_path ='/home/dr-nfs/m.badran/mawqif/ppdb-1.0-s-lexical',aug_p=config.NLPAUG_PROB,stopwords=['تطعيم','تطعيم','المرأة','تمكين','الرقمي','التحول','تغريدة',':','.','موقف','التغريدة','هو','من'])]
    print(nlp_aug)
  else:
    nlp_aug = None


  for selectedTarget in config.selectedTarget:
    if not config.selectedModels is None:
      config_dict['selectedModel'] = config.selectedModels[selectedTarget]
      config.selectedModel = config.selectedModels[selectedTarget]
    # Ex_name =  config.selectedModel.split('/')[-1]+"-"+config.Version+"-"+selectedTarget.replace(" ","")
    Ex_name = config.Version+"-"+selectedTarget.replace(" ","")
    arabert_prep = ArabertPreprocessor(model_name=config.selectedModel) if config.USE_ARABERT_PRE else None
    tokenizer = AutoTokenizer.from_pretrained(config.selectedModel)
    model = AutoModel.from_pretrained(config.selectedModel)


    if selectedTarget == 'All':
        train_df, val_df, test_df,class_weights = load_dataset(config.TrainData_name,config.TestData_name,'Covid Vaccine',config.WEIGHTED_LOSS or config.WEIGHTED_SAMPLER, arabert_prep = arabert_prep ) 
        for s in [  'Women empowerment','Digital Transformation']: 
          train_df1, val_df1, test_df1,class_weights1 = load_dataset(config.TrainData_name,config.TestData_name,s,config.WEIGHTED_LOSS or config.WEIGHTED_SAMPLER, arabert_prep = arabert_prep )
          print(len(train_df1),len(test_df1),len(val_df1),class_weights)
        
          train_df = pd.concat([train_df, train_df1], ignore_index=True)
          val_df = pd.concat([val_df, val_df1], ignore_index=True)
          test_df = pd.concat([test_df, test_df1], ignore_index=True)
          if not class_weights is None:
            class_weights = class_weights + class_weights1
          
        if not class_weights is None:
            class_weights = class_weights/3
        print("****************")

    else:
      train_df, val_df, test_df,class_weights = load_dataset(config.TrainData_name,config.TestData_name,selectedTarget,config.WEIGHTED_LOSS or config.WEIGHTED_SAMPLER, arabert_prep = arabert_prep )
    print(len(train_df),len(val_df),len(test_df),class_weights)

    data_module = TweetDataModule(train_df, val_df, test_df, tokenizer, batch_size=config.BATCH_SIZE, token_len=config.MAX_TOKEN_COUNT,class_weights= class_weights, weighted_sampler = config.WEIGHTED_SAMPLER, nlp_aug = nlp_aug)
    data_module.setup()

    for batch in data_module.train_dataloader():
      assert len(batch) == 4
      assert batch["input_ids"].shape == batch["attention_mask"].shape
      assert batch["input_ids"].shape[0] == config.BATCH_SIZE
      assert batch["input_ids"].shape[1] == config.MAX_TOKEN_COUNT
      assert batch["labels"].shape[0] == config.BATCH_SIZE
      # print(batch)
      break

    model = TweetPredictor(n_classes=3, steps_per_epoch=len(train_df) // config.BATCH_SIZE,class_weights=class_weights, config = config_dict )

    
    logger = CometLogger(
     experiment_name=Ex_name ,
     api_key='jFe8sB6aGNDL7p2LHe9V99VNy',
     workspace='mbadran2000',  # Optional
    #  save_dir='lightning_logs',  # Optional
     project_name='Mawqif',  # Optional
     )
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/"+Ex_name,
        filename="best-checkpoint",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min"
    )
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5)

    trainer = pl.Trainer(
      logger=logger,
      # enable_checkpointing=checkpoint_callback,
      callbacks=[early_stopping_callback,checkpoint_callback],
      max_epochs=config.N_EPOCHS,
      gpus=1,
      #progress_bar_refresh_rate=30
      enable_progress_bar=True,
      progress_bar_refresh_rate=1,
      log_every_n_steps=10 #default is 50
    )
    logger.log_hyperparams(config_dict) 
    logger.log_hyperparams({'selectedTarget':selectedTarget}) 
    logger.log_hyperparams({'Loss Weights':class_weights}) 



    trainer.fit(model, data_module)
    trainer.test(datamodule=data_module)


    trained_model = model.load_from_checkpoint("checkpoints/"+Ex_name+"/best-checkpoint.ckpt")
    trained_model.freeze()

    if selectedTarget == "All":
      for s in [ 'Covid Vaccine', 'Women empowerment','Digital Transformation']: 
        train_df, val_df, test_df,class_weights = load_dataset(config.TrainData_name,config.TestData_name,s,config.WEIGHTED_LOSS )
        data_module = TweetDataModule(train_df, val_df, test_df, tokenizer, batch_size=config.BATCH_SIZE, token_len=config.MAX_TOKEN_COUNT)
        data_module.setup()
        log_results(Ex_name, trained_model, logger, s, data_module.get_val_dataset(), "val")
        if config.log_test:
          log_results(Ex_name, trained_model, logger, s, data_module.get_test_dataset(), "test")
    else:
      log_results(Ex_name, trained_model, logger, selectedTarget, data_module.get_val_dataset(), "val")
      if config.log_test:
        log_results(Ex_name, trained_model, logger, selectedTarget, data_module.get_test_dataset(), "test")
    
    del trained_model, model, data_module
    
    # exit()
if len(config.selectedTarget) == 3 or config.selectedTarget[0] == "All":
  # Ex_name =  config.selectedModel.split('/')[-1]+"-"+config.Version+"-Overall"
  # config.selectedModel.split('/')[-1]+"-"+selectedTarget.replace(" ","")
  Ex_name =  config.Version+"-Overall"
  logger = CometLogger(
      experiment_name=Ex_name ,
      api_key='jFe8sB6aGNDL7p2LHe9V99VNy',
      workspace='mbadran2000',  # Optional
      #  save_dir='lightning_logs',  # Optional
      project_name='Mawqif',  # Optional
  )    
  log_StanceEval(Ex_name,logger,"val")
  if config.log_test:
    log_StanceEval(Ex_name,logger,"test")



