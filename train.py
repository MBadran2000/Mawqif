import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import pickle
import re
import os
# from comet_ml import Experiment
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
#from pytorch_lightning.metrics.functional import accuracy, f1, auroc
#https://torchmetrics.readthedocs.io/en/stable/metrics.html
#https://stackoverflow.com/questions/69139618/torchmetrics-does-not-work-with-pytorchlightning
from torchmetrics.functional import accuracy, f1_score, roc, precision, recall, confusion_matrix
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from pytorch_lightning.callbacks import EarlyStopping

# Visualisation
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc

# %matplotlib inline
# %config InlineBackend.figure_format='retina'
# sns.set(style='whitegrid', palette='muted', font_scale=1.2)
# rcParams['figure.figsize'] = 16, 10
tqdm.pandas()

from arabert.preprocess import ArabertPreprocessor

import config
from dataset import TweetEmotionDataset, TweetDataModule, load_dataset
from model import TweetPredictor
from utils.prediction_utils import get_predictions,  predict


pl.seed_everything(42)

if __name__ == '__main__': 
  config_dict = {attr: getattr(config, attr) for attr in dir(config) if not attr.startswith('__')}
  print(config_dict)


  for selectedTarget in config.selectedTarget:
    Ex_name =  config.selectedModel.split('/')[-1]+"-"+selectedTarget.replace(" ","")+config.Version
    arabert_prep = ArabertPreprocessor(model_name=config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.selectedModel)
    model = AutoModel.from_pretrained(config.selectedModel)

    train_df, val_df, test_df = load_dataset(config.TrainData_name,config.TestData_name,selectedTarget)

    data_module = TweetDataModule(train_df, val_df, test_df, tokenizer, batch_size=config.BATCH_SIZE, token_len=config.MAX_TOKEN_COUNT)
    data_module.setup()

    for batch in data_module.train_dataloader():
      assert len(batch) == 4
      assert batch["input_ids"].shape == batch["attention_mask"].shape
      assert batch["input_ids"].shape[0] == config.BATCH_SIZE
      assert batch["input_ids"].shape[1] == config.MAX_TOKEN_COUNT
      assert batch["labels"].shape[0] == config.BATCH_SIZE
      # print(batch)
      break

    model = TweetPredictor(n_classes=3, steps_per_epoch=len(train_df) // config.BATCH_SIZE, n_epochs=config.N_EPOCHS,selectedModel =config.selectedModel)

    
    # logger = CometLogger("lightning_logs", name=Ex_name) 
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
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=2)

    trainer = pl.Trainer(
      logger=logger,
      # enable_checkpointing=checkpoint_callback,
      callbacks=[early_stopping_callback,checkpoint_callback],
      max_epochs=config.N_EPOCHS,
      gpus=1,
      #progress_bar_refresh_rate=30
      enable_progress_bar=True,
      #log_every_n_steps=16 #default is 50
    )
    logger.log_hyperparams({'selectedTarget':selectedTarget}) 
    logger.log_hyperparams(config_dict) 


    trainer.fit(model, data_module)
    trainer.test(datamodule=data_module)


    val_dataset = TweetEmotionDataset(val_df, tokenizer, max_token_len=config.MAX_TOKEN_COUNT)
    test_dataset = TweetEmotionDataset(test_df, tokenizer, max_token_len=config.MAX_TOKEN_COUNT)

    

    p = "checkpoints/"+Ex_name+"/best-checkpoint.ckpt"
    trained_model = model.load_from_checkpoint(p,n_classes=3)
    trained_model.freeze()


    val_texts, val_pred, val_pred_probs, val_true = get_predictions(
      model,
      data_module.get_val_dataset()
    )

    class_names = ['None','Favor','Against']
    report_val = classification_report(val_true, val_pred, target_names=class_names, zero_division=0, digits=4, output_dict=True)


    for key, value in report_val.items():
        if key == "accuracy":
            logger.experiment.log_metric("val_"+key, value)
        else:
            logger.experiment.log_metrics(value,prefix="val_"+key)
    logger.experiment.log_confusion_matrix(
        val_true.tolist(),
        val_pred.tolist(),
        title="Val Confusion Matrix",
        file_name="val-confusion-matrix.json"
    )

    if config.log_test:
      test_texts, test_pred, test_pred_probs, test_true = get_predictions(
        model,
        data_module.get_test_dataset()
      )
      report_test = classification_report(test_true, test_pred, target_names=class_names, zero_division=0, digits=4, output_dict=True)
      for key, value in report_test.items():
          if key == "accuracy":
              logger.experiment.log_metric("test_"+key, value)
          else:
              logger.experiment.log_metrics(value,prefix="test_"+key)
      logger.experiment.log_confusion_matrix(
          test_true.tolist(),
          test_pred.tolist(),
          title="Test Confusion Matrix",
          file_name="test-confusion-matrix.json"
      )
    
    # exit()




