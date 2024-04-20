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
import torch.nn.functional as F
import torch.nn.functional as F
from peft import LoraConfig, TaskType, LoKrConfig, LoHaConfig, AdaLoraConfig
from peft import get_peft_model
from transformers import AutoModelForCausalLM

# Visualisation
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc

from utils.model_utils import FocalLoss, LabelSmoothingCrossEntropyLoss, getPeftModel


class TweetPredictor(pl.LightningModule):

  def __init__(self, n_classes: list = 3, steps_per_epoch=None, class_weights=None ,config = None): 
    super().__init__()

    # self.bert = AutoModel.from_pretrained(config['selectedModel'], return_dict=True)
    self.bert = AutoModelForCausalLM.from_pretrained(config['selectedModel'], return_dict=True)
    # # Freeze the BERT model
    if config['FREEZE_BERT']==True:
      print("Freezing Bert")
      for param in self.bert.parameters():
          param.requires_grad = False

    # self.classifier1 = nn.Linear(self.bert.config.hidden_size, 256)
    # self.classifier2 = nn.Linear(256, 128)
    # self.classifier3 = nn.Linear(128, n_classes)
    # self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
    # self.dropout = nn.Dropout(config["DROPOUT"])

    self.steps_per_epoch = steps_per_epoch
    self.n_epochs = config['N_EPOCHS']

    if config['LOSS']==0:
      self.criterion = nn.CrossEntropyLoss(weight=class_weights) # for multi-class
      self.softmax = torch.softmax
      print("loss: CrossEntropyLoss" )
    elif config['LOSS']==1:
      self.criterion = LabelSmoothingCrossEntropyLoss(epsilon=0.1, num_classes=n_classes)
      self.softmax = F.log_softmax
      print("loss: LabelSmoothingCrossEntropyLoss" )
    elif config['LOSS']==2:
      self.criterion = FocalLoss(gamma=4, weight=class_weights)
      self.softmax = F.log_softmax
      print("loss: FocalLoss" )

    self.save_hyperparameters()

    # Initialize weight decay
    self.weight_decay = config['WEIGHT_DECAY']
    self.lr = config['LEARNING_RATE']

    self.bert = getPeftModel(self.bert,config['USE_PEFT'])

    

  
  def forward(self, input_ids, attention_mask, labels=None):
    output = self.bert(input_ids, attention_mask=attention_mask)

    # output = self.dropout(output.pooler_output)
    # output = self.classifier1(output)
    # output = self.dropout(output) 
    # output = self.classifier2(output)  
    # output = self.dropout(output)
    # output = self.classifier3(output)  

    # output = self.dropout(output.pooler_output)
    # output = self.classifier(output) 

    # output = self.classifier(output.pooler_output) 

    # output = self.softmax(output, dim=1)
    # output = torch.softmax(output, dim=1) # for cross entropy
    # output = F.log_softmax(output, dim=1) # for focal or nllloss

    loss = 0
    # batch_size = output['logits'].size(0)
    # sentence_size = output['logits'].size(1)
    # vocab_size = output['logits'].size(2)

    if labels is not None:
      # print(labels)
      # print(output)
      # print(labels.squeeze(1).shape,output['logits'].shape)
      # loss =  output.loss
      # softmax_prediction = F.softmax(output['logits'], dim=2)
      batch_size = output['logits'].size(0)
      logits = output['logits'].view(batch_size * 128, 64000)
      gt = labels.view(batch_size * 128)
      
      # print("aa",softmax_prediction.squeeze().shape, labels.squeeze().long().shape)
      loss = F.cross_entropy(logits, gt)
      # loss = self.criterion(softmax_prediction.squeeze(), labels.squeeze().long()) 
    # {'None': 59910, 'Favor': 354, 'Against': 1092}

    word_indices = [59910, 354, 1092]
    pred_reshaped = output['logits'].view(batch_size, 128, 64000)
    word_probs = []
    for idx in word_indices:
        word_prob = pred_reshaped[:, 0, idx]
        word_probs.append(word_prob.unsqueeze(1))

    output = torch.cat(word_probs, dim=1)
    # output = self.softmax(output, dim=1)
    # print(word_probs.shape) 

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
    # optimizer = AdamW(self.parameters(), lr=2e-5, weight_decay=self.weight_decay) 
    optimizer = AdamW(self.parameters(), lr=self.lr)
    # optimizer = torch.optim.Adagrad(self.parameters(), lr=self.lr)
    print(optimizer)

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