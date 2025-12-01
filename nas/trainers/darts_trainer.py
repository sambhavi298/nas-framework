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


class DartsTrainer:
    def __init__(
        self,
        work_dir: str,
        init_channels: int = 16,
        num_layers: int = 6,
        num_classes: int = 10,
        batch_size: int = 64,
        train_val_split: float = 0.9,   # fraction for training (rest for validation)
        lr_w: float = 0.025,
        momentum: float = 0.9,
        weight_decay: float = 3e-4,
        lr_alpha: float = 3e-4,
        epochs: int = 50,
        device: Optional[str] = None,
        num_workers: int = 4,
        grad_clip: float = 5.0,
        use_amp: bool = False,
        save_every: int = 5,
        seed: int = 42,
        unrolled: bool = False,
    ):
        self.work_dir = work_dir
        os.makedirs(work_dir, exist_ok=True)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_workers = num_workers
        self.grad_clip = grad_clip
        self.use_amp = use_amp
        self.save_every = save_every
        self.seed = seed
        self.unrolled = unrolled

        # Model
        self.model = SuperNet(init_channels=init_channels, num_layers=num_layers, num_classes=num_classes)
        self.model.to(self.device)