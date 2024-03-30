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
from torch.optim.lr_scheduler import LambdaLR

# Visualisation
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc


class TweetPredictor(pl.LightningModule):

  def __init__(self, n_classes: list = 3, steps_per_epoch=None, n_epochs=None,selectedModel=None,class_weights= None):
    super().__init__()
    self.bert = AutoModel.from_pretrained(selectedModel, return_dict=True)
    # # Freeze the BERT model
    # for param in self.bert.parameters():
    #     param.requires_grad = False

    self.classifier1 = nn.Linear(self.bert.config.hidden_size, 256)
    self.classifier2 = nn.Linear(256, 128)
    self.classifier3 = nn.Linear(128, n_classes)


    self.steps_per_epoch = steps_per_epoch
    self.n_epochs = n_epochs
    self.criterion = nn.CrossEntropyLoss(weight=class_weights) # for multi-class
    self.save_hyperparameters()


  def forward(self, input_ids, attention_mask, labels=None):
    output = self.bert(input_ids, attention_mask=attention_mask)
    output = self.classifier1(output.pooler_output) 
    output = self.classifier2(output)  
    output = self.classifier3(output)  
    output = torch.softmax(output, dim=1) # for multi-class   
    loss = 0
    if labels is not None:
        loss = self.criterion(output, labels) 
    return loss, output


  def training_step(self, batch, batch_idx):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    loss, outputs = self(input_ids, attention_mask, labels)
    self.log("train_loss", loss, prog_bar=True, logger=True)
    
    lr = self.optimizers().param_groups[0]['lr']
    self.log('lr_abs', lr, prog_bar=True, logger=True )#, on_step=False, on_epoch=True)#, on_step=True, on_epoch=False)

    return {"loss": loss, "predictions": outputs, "labels": labels}


  def validation_step(self, batch, batch_idx):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    loss, outputs = self(input_ids, attention_mask, labels)
    self.log("val_loss", loss, prog_bar=True, logger=True)
    return loss


  def test_step(self, batch, batch_idx):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    loss, outputs = self(input_ids, attention_mask, labels)
    self.log("test_loss", loss, prog_bar=True, logger=True)
    return loss


  def configure_optimizers(self):
    optimizer = AdamW(self.parameters(), lr=2e-5)
    warmup_steps = self.steps_per_epoch // 3     ## we will use third of the training examples for warmup
    total_steps = self.steps_per_epoch * self.n_epochs - warmup_steps

    scheduler = get_linear_schedule_with_warmup(
      optimizer, 
      warmup_steps, 
      total_steps
    )
    return [optimizer] ,[{'scheduler':scheduler, 'interval': 'step'}]

  # def configure_optimizers(self):
  #   optimizer = AdamW(self.parameters(), lr=2e-5)
  #   return [optimizer]

  # def configure_optimizers(self):
  #     optimizer = AdamW(self.parameters(), lr=2e-5)
  #     warmup_epochs = self.steps_per_epoch // 3 
  #     # num_training_steps = len(self.train_dataloader()) * self.max_epochs
  #     warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: min((epoch + 1) / warmup_epochs, 1.0))
      
  #     return [ optimizer],[{'scheduler':warmup_scheduler, 'interval': 'epoch'}] 
          
  def lr_scheduler_step(self, scheduler,optimizer_idx, metric):
    # scheduler.step(epoch=self.current_epoch)
    scheduler.step()  