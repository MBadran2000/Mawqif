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
from utils.prediction_utils import get_predictions,  predict
from text_preprocessor import TextPreprocessor


trained_model = TweetPredictor.load_from_checkpoint("lightning_logs/tweet-stance/version_0/checkpoints/epoch=19-step=620.ckpt", n_classes=3)
trained_model.freeze()

val_dataset = TweetDataModule(val_df, tokenizer, batch_size=config.BATCH_SIZE, token_len=config.MAX_TOKEN_COUNT, text_preprocessor=TextPreprocessor())
test_df = TweetDataModule(test_df, tokenizer, batch_size=config.BATCH_SIZE, token_len=config.MAX_TOKEN_COUNT, text_preprocessor=TextPreprocessor())

val_texts, val_pred, val_pred_probs, val_true = get_predictions(
  trained_model,
  val_dataset
)
test_texts, test_pred, test_pred_probs, test_true = get_predictions(
  trained_model,
  test_dataset
)

report_val = classification_report(val_true, val_pred, target_names=class_names, zero_division=0, digits=4)
report_test = classification_report(test_true, test_pred, target_names=class_names, zero_division=0, digits=4)

experiment.log_text("classification_report", report)
