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
#from pytorch_lightning.metrics.functional import accuracy, f1, auroc
#https://torchmetrics.readthedocs.io/en/stable/metrics.html
#https://stackoverflow.com/questions/69139618/torchmetrics-does-not-work-with-pytorchlightning
from torchmetrics.functional import accuracy, f1_score, roc, precision, recall, confusion_matrix
from sklearn.metrics import classification_report, multilabel_confusion_matrix

# Visualisation
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc

import torch.nn.functional as F


def predict(text, model, tokenizer):

  encoding = tokenizer.encode_plus(
    text,
    add_special_tokens=True,
    max_length=128,
    return_token_type_ids=False,
    padding="max_length",
    return_attention_mask=True,
    return_tensors='pt',
  )

  _, prediction_prop = model(encoding["input_ids"], encoding["attention_mask"])
  if isinstance(prediction_prop, dict):
      prediction_prop = prediction_prop['logits']
  prediction_prop = prediction_prop.detach()
  print("prediction_prop: ",prediction_prop)
  prediction = torch.max(prediction_prop, dim=1)
  print("prediction: ",prediction)
  
  return prediction

def get_predictions(model, data_loader):
  texts = []
  predictions = []
  prediction_probs = []
  labels = []
  for item in tqdm(data_loader):

    text = item["text"]
    if "labels_no" in item.keys():
      labels.append(item["labels_no"])
    else:
      labels.append(item["labels"])

    _, _, output = model(item["input_ids"].unsqueeze(dim=0), item["attention_mask"].unsqueeze(dim=0))
    if isinstance(output, dict):
      output = output['logits']
    output = output.detach()

    _, preds = torch.max(output, dim=1)
    probs = F.softmax(output, dim=1)
  
    texts.append(text) # we can use .append instead of .extend
    predictions.extend(preds.detach())
    prediction_probs.extend(probs.detach())


  predictions = torch.stack(predictions).cpu()
  prediction_probs = torch.stack(prediction_probs).cpu()
  labels = torch.stack(labels).cpu()

  return texts, predictions, prediction_probs, labels

# def get_predictions_parrell(model, data_loader):
#   texts = []
#   predictions = []
#   prediction_probs = []
#   labels = []
#   for batch in tqdm(data_loader):
#       # print(batch)
#       texts.extend(batch["text"])  # assuming "text" is a list in batch
#       if "labels_no" in batch.keys():
#           labels.extend(batch["labels_no"])
#       else:
#           labels.extend(batch["labels"])

#       with torch.no_grad():
#           outputs = model(batch["input_ids"], batch["attention_mask"])
#           # print(outputs.shape)
#           if isinstance(output, dict):
#             output = output['logits']
#           output = output.detach()

#           _, preds = torch.max(logits, dim=1)
#           probs = F.softmax(logits, dim=1)

#       predictions.extend(preds.cpu().tolist())
#       prediction_probs.extend(probs.cpu().tolist())

#   return texts, predictions, prediction_probs, labels
