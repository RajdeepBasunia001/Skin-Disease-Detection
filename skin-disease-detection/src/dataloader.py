import os
import sys
import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, WeightedRandomSampler
from pathlib import Path

# --- Setup project root path ---
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.dataset import SkinDataset

# --- Configuration ---
PROCESSED_DIR = ROOT_DIR / "data/processed"
LABEL_MAPPING_PATH = PROCESSED_DIR / "label_mapping.json"


def get_weighted_sampler(csv_path=PROCESSED_DIR / "train.csv"):
    """
    Creates a WeightedRandomSampler to handle class imbalance.
    Each sample gets a weight inversely proportional to its class frequency.
    """
    df = pd.read_csv(csv_path)
    class_counts = df['label'].value_counts().sort_index().values
    total_samples = len(df)
    class_weights = total_samples / class_counts
    sample_weights = df['label'].map(lambda x: class_weights[x]).values
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=total_samples,
        replacement=True
    )


def get_dataloaders(batch_size=32, num_workers=4, image_size=224, use_sampler=False):
    """
    Creates and returns train, validation, and test dataloaders.
    """
    data_paths = {
        "train": (PROCESSED_DIR / "train.csv", PROCESSED_DIR / "train"),
        "val": (PROCESSED_DIR / "val.csv", PROCESSED_DIR / "val"),
        "test": (PROCESSED_DIR / "test.csv", PROCESSED_DIR / "test"),
    }

    # ✅ Pass already_resized=True since preprocessing resizes images
    train_ds = SkinDataset(
        csv_path=data_paths["train"][0],
        img_dir=data_paths["train"][1],
        is_train=True,
        image_size=image_size,
        already_resized=True
    )
    val_ds = SkinDataset(
        csv_path=data_paths["val"][0],
        img_dir=data_paths["val"][1],
        is_train=False,
        image_size=image_size,
        already_resized=True
    )
    test_ds = SkinDataset(
        csv_path=data_paths["test"][0],
        img_dir=data_paths["test"][1],
        is_train=False,
        image_size=image_size,
        already_resized=True
    )

    sampler = get_weighted_sampler() if use_sampler else None

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0 and os.name != "nt")
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0 and os.name != "nt")
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0 and os.name != "nt")
    )

    print(f"✅ DataLoaders created with batch_size={batch_size}, num_workers={num_workers}, sampler={use_sampler}")
    return train_loader, val_loader, test_loader


def compute_class_weights(csv_path=PROCESSED_DIR / "train.csv"):
    """
    Computes class weights for an imbalanced dataset.
    """
    df = pd.read_csv(csv_path)
    class_counts = df['label'].value_counts().sort_index().values

    total_samples = sum(class_counts)
    num_classes = len(class_counts)

    weights = total_samples / (num_classes * class_counts)
    return torch.tensor(weights, dtype=torch.float)


# --- Self-test block ---
if __name__ == "__main__":
    print("🔄 Testing DataLoader...")

    # For local test use num_workers=0 to avoid multiprocessing issues
    train_loader, _, _ = get_dataloaders(batch_size=8, num_workers=0, use_sampler=True)

    # Fetch one batch
    images, labels = next(iter(train_loader))
    print(f"✅ Loaded one batch. Images shape: {images.shape}, Labels shape: {labels.shape}")

    # Visualize 4 sample images
    fig, axs = plt.subplots(1, 4, figsize=(12, 4))
    fig.suptitle("Sample Augmented Images from DataLoader")
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    for i in range(4):
        img = images[i].numpy().transpose((1, 2, 0))
        img = img * std + mean  # ✅ denormalize
        img = img.clip(0, 1)
        axs[i].imshow(img)
        axs[i].set_title(f"Label: {labels[i].item()}")
        axs[i].axis("off")
    plt.show()

    # Test class weight calculation
    print("\n⚖️ Testing class weight calculation...")
    weights = compute_class_weights()
    print(f"✅ Computed class weights: {weights}")
