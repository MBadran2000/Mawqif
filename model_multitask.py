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
    self.classifierSTA = nn.Linear(self.bert.config.hidden_size, n_classes)
    self.classifierSENT = nn.Linear(self.bert.config.hidden_size, n_classes)
    self.classifierSAR = nn.Linear(self.bert.config.hidden_size, 2)####2 classes
    self.classifierSTA_final = nn.Linear(n_classes+n_classes+2,3)
    self.taskWeights = [0.6,0.3,0.1]
    # self.taskWeights = [1,1,1]
    # self.lossSTA=0
    # self.lossSENT=0
    

    self.steps_per_epoch = steps_per_epoch
    self.n_epochs = n_epochs
    self.criterionSTA = nn.CrossEntropyLoss(weight=class_weights) # for multi-class
    self.criterion = nn.CrossEntropyLoss() # for multi-class
    self.criterion = nn.CrossEntropyLoss() # for multi-class

    self.save_hyperparameters()
    


  def forward(self, input_ids, attention_mask, labels=None):
    output = self.bert(input_ids, attention_mask=attention_mask)
    outputSTA1 = self.classifierSTA(output.pooler_output) 
    # outputSTA = torch.softmax(outputSTA1, dim=1) # for multi-class  

    outputSENT1 = self.classifierSENT(output.pooler_output) 
    outputSENT = torch.softmax(outputSENT1, dim=1) # for multi-class   

    outputSAR1 = self.classifierSAR(output.pooler_output) 
    # outputSAR = torch.softmax(outputSAR, dim=1) # for multi-class  
    outputSAR = torch.sigmoid(outputSAR1) 

    output_concat = torch.cat(( outputSTA1, outputSENT1, outputSAR1), dim=1) 
    output_concat = self.classifierSTA_final(output_concat)
    outputSTA = torch.softmax(output_concat, dim=1)

    
    loss = 0
    if labels is not None:
      loss = dict(
        lossSTA = self.taskWeights[0]*self.criterionSTA(outputSTA,labels['STA']) ,
        lossSENT = self.taskWeights[1]*self.criterion(outputSENT,labels['SENT']) ,
        lossSAR = self.taskWeights[2]*self.criterion(outputSAR,labels['SAR']))
    return loss, outputSTA
 


  def training_step(self, batch, batch_idx):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    loss, outputs = self(input_ids, attention_mask, labels)
    self.log("Stance_train_loss", loss['lossSTA'],  logger=True)
    self.log("Sentiment_train_loss", loss['lossSENT'],  logger=True)
    self.log("Sarcasm_train_loss", loss['lossSAR'],logger=True)
    loss = loss['lossSTA']+loss['lossSAR']+loss['lossSENT']
    self.log("train_loss", loss, prog_bar=True, logger=True)

    lr = self.optimizers().param_groups[0]['lr']
    self.log('lr_abs', lr, prog_bar=True, logger=True )#, on_step=False, on_epoch=True)#, on_step=True, on_epoch=False)

    return {"loss": loss, "predictions": outputs, "labels": labels}


  def validation_step(self, batch, batch_idx):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    loss, outputs = self(input_ids, attention_mask, labels)
    self.log("Stance_val_loss", loss['lossSTA'],  logger=True)
    self.log("Sentiment_val_loss", loss['lossSENT'],  logger=True)
    self.log("Sarcasm_val_loss", loss['lossSAR'], logger=True)

    ##Hierarchical weighting
    # self.lossSTA+=loss['lossSTA']
    # self.lossSENT+=loss['lossSENT']

    loss = loss['lossSTA']+loss['lossSAR']+loss['lossSENT']    
    self.log("val_loss", loss, prog_bar=True, logger=True)


    return loss

  ##Hierarchical weighting
  # def validation_epoch_end(self, validation_step_outputs):
  #   self.taskWeights[0] = max(min((self.lossSTA/self.lossSENT)*self.taskWeights[0],2),1)
  #   self.log("Sentance task weight ", self.taskWeights[0], logger=True)
  #   self.lossSTA=0
  #   self.lossSENT=0



  def test_step(self, batch, batch_idx):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    loss, outputs = self(input_ids, attention_mask, labels)
    self.log("Stance_test_loss", loss['lossSTA'], logger=True)
    self.log("Sentiment_test_loss", loss['lossSENT'],  logger=True)
    self.log("Sarcasm_test_loss", loss['lossSAR'],  logger=True)
    loss = loss['lossSTA']+loss['lossSAR']+loss['lossSENT']

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