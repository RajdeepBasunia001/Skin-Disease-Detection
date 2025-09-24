# =======================================================
# dataloader.py
# =======================================================

import os
import sys
import json
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, WeightedRandomSampler
from pathlib import Path
import numpy as np
import warnings

# --- Setup project root path ---
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.dataset import SkinDataset  # Make sure dataset.py exists

# --- Configuration ---
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
LABEL_MAPPING_PATH = PROCESSED_DIR / "label_mapping.json"
DATASET_STATS_PATH = PROCESSED_DIR / "dataset_stats.json"

# =======================================================
# Focal Loss
# =======================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(weight=self.alpha, reduction="none")

    def forward(self, input, target):
        ce_loss = self.ce_loss(input, target)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss

# =======================================================
# Weighted Sampler for imbalanced dataset
# =======================================================
def get_weighted_sampler(csv_path=PROCESSED_DIR / "train.csv"):
    df = pd.read_csv(csv_path)
    class_counts = df["label"].value_counts().sort_index().values
    total_samples = len(df)
    class_weights = total_samples / class_counts
    sample_weights = df["label"].map(lambda x: class_weights[x]).values
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=total_samples,
        replacement=True,
    )

# =======================================================
# Optional Mixup (toggleable)
# =======================================================
def mixup_data(x, y, alpha=1.0):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# =======================================================
# DataLoader
# =======================================================
def get_dataloaders(batch_size=32, num_workers=4, image_size=224, use_sampler=False, use_mixup=False):
    """
    Returns train, val, test DataLoaders
    """

    # --- Load dataset stats ---
    if not DATASET_STATS_PATH.exists():
        warnings.warn(f"⚠️ dataset_stats.json not found at {DATASET_STATS_PATH}. Using ImageNet defaults.")
    else:
        with open(DATASET_STATS_PATH, "r") as f:
            stats = json.load(f)
        print(f"✅ Loaded dataset stats: mean={stats['mean']}, std={stats['std']}")

    # --- Create datasets ---
    train_ds = SkinDataset(
        csv_path=PROCESSED_DIR / "train.csv",
        processed_dir=PROCESSED_DIR,
        split_name="train",
        is_train=True,
        image_size=image_size,
        already_resized=True,
        use_without_hair=True
    )
    val_ds = SkinDataset(
        csv_path=PROCESSED_DIR / "val.csv",
        processed_dir=PROCESSED_DIR,
        split_name="val",
        is_train=False,
        image_size=image_size,
        already_resized=True,
        use_without_hair=True
    )
    test_ds = SkinDataset(
        csv_path=PROCESSED_DIR / "test.csv",
        processed_dir=PROCESSED_DIR,
        split_name="test",
        is_train=False,
        image_size=image_size,
        already_resized=True,
        use_without_hair=True
    )

    # --- Weighted sampler ---
    sampler = get_weighted_sampler() if use_sampler else None
    shuffle = not use_sampler

    # --- Dataloaders ---
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0 and os.name != "nt"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0 and os.name != "nt"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0 and os.name != "nt"),
    )

    print(f"✅ DataLoaders created | batch_size={batch_size}, num_workers={num_workers}, sampler={use_sampler}, mixup={use_mixup}")
    return train_loader, val_loader, test_loader, use_mixup

# =======================================================
# Class weights computation
# =======================================================
def compute_class_weights(csv_path=PROCESSED_DIR / "train.csv"):
    df = pd.read_csv(csv_path)
    class_counts = df["label"].value_counts().sort_index().values
    total_samples = sum(class_counts)
    num_classes = len(class_counts)
    weights = total_samples / (num_classes * class_counts)
    return torch.tensor(weights, dtype=torch.float)

# =======================================================
# Optional self-test
# =======================================================
if __name__ == "__main__":
    print("🔄 Testing DataLoader with one batch...")
    train_loader, val_loader, test_loader, _ = get_dataloaders(batch_size=8, num_workers=0, use_sampler=True, use_mixup=False)

    images, labels = next(iter(train_loader))
    print(f"✅ Loaded one batch. Images: {images.shape}, Labels: {labels.shape}")

    # Test Focal Loss
    weights = compute_class_weights()
    dummy_input = torch.randn(8, len(weights))
    dummy_target = torch.randint(0, len(weights), (8,))
    fl = FocalLoss(alpha=weights)
    loss = fl(dummy_input, dummy_target)
    print(f"✅ Focal Loss test: {loss.item():.4f}")
