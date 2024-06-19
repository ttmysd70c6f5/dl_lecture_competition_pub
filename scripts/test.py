# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 19:12:10 2024

@author: Tatsumi
"""

import os, sys
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
import hydra
from omegaconf import DictConfig # Operate configs as a dict
import wandb
from termcolor import cprint
from tqdm import tqdm

from src.datasets import ThingsMEGDataset
from src.models import BasicConvClassifier
from src.utils import set_seed


set_seed(1234)

# Load data
loader_args = {"batch_size": 128, "num_workers": 4}

train_set = ThingsMEGDataset("train", "data")
train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **loader_args)
val_set = ThingsMEGDataset("val", "data")
val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, **loader_args)
test_set = ThingsMEGDataset("test", "data")
test_loader = torch.utils.data.DataLoader(
    test_set, shuffle=False, batch_size=128, num_workers=4
    )