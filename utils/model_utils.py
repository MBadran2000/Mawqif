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

# Visualisation
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc

class LabelSmoothingCrossEntropyLoss(torch.nn.Module):
    def __init__(self, epsilon=0.1, num_classes=10):
        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        self.epsilon = epsilon
        self.num_classes = num_classes

    def forward(self, logits, target):
        # Convert target labels to one-hot encoding
        target_one_hot = F.one_hot(target, num_classes=self.num_classes)
        
        # Apply label smoothing
        target_smooth = (1 - self.epsilon) * target_one_hot + (self.epsilon / self.num_classes)
        # print(target_smooth)
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(logits, target_smooth.argmax(dim=1))
        
        return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average
        

    def forward(self, logits, target):
        if logits.dim() > 2:
            logits = logits.view(logits.size(0), logits.size(1), -1)  # N,C,H,W => N,C,H*W
            logits = logits.transpose(1, 2)                        # N,C,H*W => N,H*W,C
            logits = logits.contiguous().view(-1, logits.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        # logpt = F.log_softmax(logits, dim=1)
        logpt = logits
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        if self.weight is not None:
            logpt = logpt * self.weight.gather(0, target.view(-1))

        loss = -1 * (1 - pt) ** self.gamma * logpt

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

def getPeftModel(model, use_PEFT):
  if use_PEFT==0:
    print("without PEFT")
  elif use_PEFT==1:
    config = LoraConfig(
    r=16,
    target_modules=["query", "value"],
    # task_type=TaskType.SEQ_CLS,
    lora_alpha=32,
    lora_dropout=0.05
    )
    model = get_peft_model(model, config)
    print("LORA")
    model.print_trainable_parameters()
  elif use_PEFT==2:
    config = LoHaConfig(
    r=16,
    alpha=16,
    target_modules=["query", "value"],
    module_dropout=0.1,
    )
    model = get_peft_model(model, config)
    print("LoHa")
    model.print_trainable_parameters()
  elif use_PEFT==3:
    config = LoKrConfig(
    r=16,
    alpha=16,
    target_modules=["query", "value"],
    module_dropout=0.1,
    )
    model = get_peft_model(model, config)
    print("LoKr")
    model.print_trainable_parameters()
  elif use_PEFT==4:
    config = AdaLoraConfig(
    r=8,
    init_r=12,
    tinit=200,
    tfinal=1000,
    deltaT=10,
    target_modules=["query", "value"],
    )
    model = get_peft_model(model, config)
    print("AdaLora")
    model.print_trainable_parameters()

  return model


def get_labels_tokens(selectedModel):
  tokenizer = AutoTokenizer.from_pretrained(selectedModel)
  # if tokenizer.pad_token is None:
  #     # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
  #     tokenizer.pad_token = tokenizer.eos_token

  labels_tokens = [0,0,0]

  encoding_stance = tokenizer.encode_plus(
        'حياد',
        add_special_tokens=False,
        max_length=10,
        return_token_type_ids=False,
        # padding="max_length",
        truncation=True,
        return_attention_mask=False, # to make sure each sequence is maximum of max length
        return_tensors='pt', #return it as pytorch
      )
  labels_tokens[0]=encoding_stance['input_ids'][0][0].item()

  encoding_stance = tokenizer.encode_plus(
        'مع',
        add_special_tokens=False,
        max_length=10,
        return_token_type_ids=False,
        # padding="max_length",
        truncation=True,
        return_attention_mask=False, # to make sure each sequence is maximum of max length
        return_tensors='pt', #return it as pytorch
      )
  labels_tokens[1]=encoding_stance['input_ids'][0][0].item()

  encoding_stance = tokenizer.encode_plus(
        'ضد',
        add_special_tokens=False,
        max_length=10,
        return_token_type_ids=False,
        # padding="max_length",
        truncation=True,
        return_attention_mask=False, # to make sure each sequence is maximum of max length
        return_tensors='pt', #return it as pytorch
      )
  labels_tokens[2]=encoding_stance['input_ids'][0][0].item()
  print("labels tokens: ",labels_tokens)
  return labels_tokens