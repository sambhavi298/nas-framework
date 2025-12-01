import os
import time
import math
import torch
import shutil
import logging
from typing import Optional, Dict
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

# Local imports (assumes your project structure)
from nas.models.supernet import SuperNet
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip, RandomCrop

# Setup a basic logger
logger = logging.getLogger("darts_trainer")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)