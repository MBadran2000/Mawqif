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
from pytorch_metric_learning import losses

# Visualisation
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc

from utils.model_utils import FocalLoss, LabelSmoothingCrossEntropyLoss, getPeftModel

def get_constrastive_loss(constrastive_loss, config, bert,n_classes=3):
  if constrastive_loss == 0:
    print("Without using Constrastive Loss")
    return None
  elif constrastive_loss == 1:
    print("Using Constrastive Loss NTXentLoss")
    return losses.NTXentLoss()
  elif constrastive_loss == 2:
    print("Using Constrastive Loss AngularLoss")
    return losses.AngularLoss()
  elif constrastive_loss == 3:    
    print("Using Constrastive Loss ArcFaceLoss")
    return losses.ArcFaceLoss(n_classes, bert.config.hidden_size)
  elif constrastive_loss == 4:    
    print("Using Constrastive Loss CircleLoss")
    return losses.CircleLoss()
  elif constrastive_loss == 5:    
    print("Using Constrastive Loss ContrastiveLoss")
    return losses.ContrastiveLoss()
  elif constrastive_loss == 6:    
    print("Using Constrastive Loss CosFaceLoss")
    return losses.CosFaceLoss(n_classes, bert.config.hidden_size)
  
  elif constrastive_loss == 7:    
    print("Using Constrastive Loss DynamicSoftMarginLoss")
    return losses.DynamicSoftMarginLoss()
  elif constrastive_loss == 8:    
    print("Using Constrastive Loss FastAPLoss")
    return losses.FastAPLoss()
  elif constrastive_loss == 9:    
    print("Using Constrastive Loss GeneralizedLiftedStructureLoss")
    return losses.GeneralizedLiftedStructureLoss()
  elif constrastive_loss == 10:    
    print("Using Constrastive Loss InstanceLoss")
    return losses.InstanceLoss()
  elif constrastive_loss == 11:    
    print("Using Constrastive Loss HistogramLoss")
    return losses.HistogramLoss()
  elif constrastive_loss == 12:    
    print("Using Constrastive Loss IntraPairVarianceLoss")
    return losses.IntraPairVarianceLoss()
  
  elif constrastive_loss == 13:    
    print("Using Constrastive Loss TripletMarginLoss")
    return losses.TripletMarginLoss()
  elif constrastive_loss == 14:    
    print("Using Constrastive Loss TupletMarginLoss")
    return losses.TupletMarginLoss()
  elif constrastive_loss == 15:    
    print("Using Constrastive Loss SoftTripleLoss")
    return losses.SoftTripleLoss(n_classes, bert.config.hidden_size)
  elif constrastive_loss == 16:    
    print("Using Constrastive Loss SphereFaceLoss")
    return losses.SphereFaceLoss(n_classes, bert.config.hidden_size)


  elif constrastive_loss == 17:    
    print("Using Constrastive Loss LiftedStructureLoss")
    return losses.LiftedStructureLoss()
  elif constrastive_loss == 18:    
    print("Using Constrastive Loss MarginLoss")
    return losses.MarginLoss()
  elif constrastive_loss == 19:    
    print("Using Constrastive Loss MultiSimilarityLoss")
    return losses.MultiSimilarityLoss()
  elif constrastive_loss == 20:    
    print("Using Constrastive Loss NCALoss")
    return losses.NCALoss()
  elif constrastive_loss == 21:    
    print("Using Constrastive Loss NormalizedSoftmaxLoss")
    return losses.NormalizedSoftmaxLoss(n_classes, bert.config.hidden_size)
  elif constrastive_loss == 22:    
    print("Using Constrastive Loss PNPLoss")
    return losses.PNPLoss()
  elif constrastive_loss == 23:    
    print("Using Constrastive Loss ProxyAnchorLoss")
    return losses.ProxyAnchorLoss(n_classes, bert.config.hidden_size)
  elif constrastive_loss == 24:    
    print("Using Constrastive Loss ProxyNCALoss")
    return losses.ProxyNCALoss(n_classes, bert.config.hidden_size)
  elif constrastive_loss == 25:    
    print("Using Constrastive Loss SignalToNoiseRatioContrastiveLoss")
    return losses.SignalToNoiseRatioContrastiveLoss()

  return None



class TweetPredictor(pl.LightningModule):

  def __init__(self, n_classes: list = 3, steps_per_epoch=None, class_weights=None ,config = None): 
    super().__init__()

    self.bert = AutoModel.from_pretrained(config['selectedModel'], return_dict=True)

    # # Freeze the BERT model
    if config['FREEZE_BERT']==True:
      print("Freezing Bert")
      for param in self.bert.parameters():
          param.requires_grad = False

    # self.classifier1 = nn.Linear(self.bert.config.hidden_size, 256)
    # self.classifier2 = nn.Linear(256, 128)
    # self.classifier3 = nn.Linear(128, n_classes)
    self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
    self.dropout = nn.Dropout(config["DROPOUT"])

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
    self.contrastiveloss = get_constrastive_loss(config['CONTRASTIVE_LOSS'], config, self.bert,n_classes=n_classes)
    self.constrastivelosslambda = config['CONTRASTIVE_LOSS_LAMBDA']
    print("constrastive loss lambda: ",self.constrastivelosslambda)

  
  def forward(self, input_ids, attention_mask, labels=None):
    output = self.bert(input_ids, attention_mask=attention_mask)

    # output = self.dropout(output.pooler_output)
    # output = self.classifier1(output)
    # output = self.dropout(output) 
    # output = self.classifier2(output)  
    # output = self.dropout(output)
    # output = self.classifier3(output)  


    output_pooler = output.pooler_output
    output = self.dropout(output.pooler_output)
    output = self.classifier(output) 

    # output = self.classifier(output.pooler_output) 
    output = self.softmax(output, dim=1)
    # output = torch.softmax(output, dim=1) # for cross entropy
    # output = F.log_softmax(output, dim=1) # for focal or nllloss

    loss,con_loss = 0,0
    if labels is not None:
      loss = self.criterion(output, labels) 
      if not self.contrastiveloss is None:
        con_loss = self.contrastiveloss(output_pooler, labels)*self.constrastivelosslambda
        
    return loss, con_loss, output


  def training_step(self, batch, batch_idx):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    loss, con_loss, outputs = self(input_ids, attention_mask, labels)
    self.log("train_loss", loss, prog_bar=True, logger=True)
    if not self.contrastiveloss is None:
      self.log("train_con_loss", con_loss, prog_bar=True, logger=True)

    
    lr = self.optimizers().param_groups[0]['lr']
    self.log('lr_abs', lr, prog_bar=False, logger=True )#, on_step=False, on_epoch=True)#, on_step=True, on_epoch=False)

    return {"loss": loss+con_loss, "predictions": outputs, "labels": labels}


  def validation_step(self, batch, batch_idx):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    loss, con_loss, outputs = self(input_ids, attention_mask, labels)
    self.log("val_loss", loss, prog_bar=True, logger=True)
    # if not self.contrastiveloss is None: ## not suitable for batch_size = 1
    #   self.log("val_cons_loss", con_loss, prog_bar=True, logger=True)
    #   return loss+con_loss
    return loss


  def test_step(self, batch, batch_idx):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    loss, con_loss, outputs = self(input_ids, attention_mask, labels)
    self.log("test_loss", loss, prog_bar=True, logger=True)
    # if not self.con_loss is None: ## not suitable for batch_size = 1
    #   self.log("test_cons_loss", con_loss, prog_bar=True, logger=True)
    #   return loss+con_loss
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