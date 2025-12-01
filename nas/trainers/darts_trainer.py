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

        # Optimizers: W (SGD) and alpha (Adam)
        self.optimizer_w = SGD(self.model.parameters(), lr=lr_w, momentum=momentum, weight_decay=weight_decay)
        # architecture params: only alphas
        self.optimizer_alpha = Adam([self.model.alphas], lr=lr_alpha, betas=(0.5, 0.999), weight_decay=1e-3)

        # LR scheduler for weights
        self.scheduler = CosineAnnealingLR(self.optimizer_w, T_max=epochs)

        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # Datasets / loaders (created later in setup_data)
        self.train_loader = None
        self.valid_loader = None

        # Save config
        self.config = dict(
            init_channels=init_channels,
            num_layers=num_layers,
            num_classes=num_classes,
            batch_size=batch_size,
            lr_w=lr_w,
            lr_alpha=lr_alpha,
            epochs=epochs,
            device=self.device,
            use_amp=self.use_amp,
            unrolled=self.unrolled,
        )

    # -------------------------
    # Dataset and DataLoaders
    # -------------------------
    def setup_data(self, data_root: str = "./data", cifar_download: bool = True, normalize_mean=None, normalize_std=None):
        """
        Prepare CIFAR-10 train/validation split and dataloaders.
        """
        if normalize_mean is None:
            normalize_mean = (0.4914, 0.4822, 0.4465)
        if normalize_std is None:
            normalize_std = (0.2470, 0.2435, 0.2616)

        train_transform = Compose([
            RandomCrop(32, padding=4),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize(normalize_mean, normalize_std)
        ])
        test_transform = Compose([
            ToTensor(),
            Normalize(normalize_mean, normalize_std)
        ])

        full_train = CIFAR10(root=data_root, train=True, download=cifar_download, transform=train_transform)
        n_train = len(full_train)
        split = int(self.config.get("batch_size", self.batch_size) * 0)  # placeholder, not used
        # We will split with fixed ratio controlled by train_val_split
        train_size = int(n_train * 0.9)  # default 90% train
        val_size = n_train - train_size

        train_dataset, _ = random_split(full_train, [train_size, val_size], generator=torch.Generator().manual_seed(self.seed))
        # Because random_split returns subsets with same transform, we need a val dataset with test transform.
        # Easiest: create a new CIFAR test dataset and pick the same indices for validation.
        # Simpler: load the full dataset twice and slice indices
        full_train_for_val = CIFAR10(root=data_root, train=True, download=False, transform=test_transform)
        # Acquire indices for validation (deterministic)
        indices = list(range(n_train))
        val_indices = indices[train_size:]
        from torch.utils.data import Subset
        train_subset = Subset(full_train, indices[:train_size])
        val_subset = Subset(full_train_for_val, val_indices)

        self.train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
        self.valid_loader = DataLoader(val_subset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

        logger.info(f"Data loaders ready. Train size: {len(train_subset)}, Val size: {len(val_subset)}")

    # -------------------------
    # Training utilities
    # -------------------------
    def _save_checkpoint(self, epoch: int, tag: str = "ckpt.pth"):
        state = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_w_state": self.optimizer_w.state_dict(),
            "optimizer_alpha_state": self.optimizer_alpha.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state": self.scaler.state_dict() if self.use_amp else None,
            "config": self.config,
        }
        path = os.path.join(self.work_dir, f"{epoch:03d}-{tag}")
        torch.save(state, path)
        logger.info(f"Saved checkpoint: {path}")
        return path

    def _load_checkpoint(self, path: str, resume_optimizers: bool = True):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        if resume_optimizers:
            self.optimizer_w.load_state_dict(ckpt["optimizer_w_state"])
            self.optimizer_alpha.load_state_dict(ckpt["optimizer_alpha_state"])
            self.scheduler.load_state_dict(ckpt["scheduler_state"])
            if self.use_amp and ckpt.get("scaler_state") is not None:
                self.scaler.load_state_dict(ckpt["scaler_state"])
        logger.info(f"Loaded checkpoint from {path}")
        return ckpt.get("epoch", 0)

    @staticmethod
    def accuracy(output, target, topk=(1,)):
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0)
                res.append((100.0 * correct_k / batch_size).item())
            return res