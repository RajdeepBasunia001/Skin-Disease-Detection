import os
import sys
import torch
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from src.dataset import SkinDataset

# Ensure project root is in sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def get_dataloaders(batch_size=32, num_workers=4):
    """Return train, validation, and test dataloaders."""
    train_ds = SkinDataset("data/processed/train.csv", "data/processed/train", is_train=True)
    val_ds   = SkinDataset("data/processed/val.csv", "data/processed/val", is_train=False)
    test_ds  = SkinDataset("data/processed/test.csv", "data/processed/test", is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


def compute_class_weights(csv_path="data/processed/train.csv"):
    """Compute class weights for imbalanced dataset."""
    df = pd.read_csv(csv_path)
    class_counts = df['label'].value_counts().sort_index().values
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * len(class_counts)
    return torch.tensor(weights, dtype=torch.float)


# --------------------------
# Self-test when run directly
# --------------------------
if __name__ == "__main__":
    print("🔄 Testing DataLoader...")
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=8)

    batch = next(iter(train_loader))
    images, labels = batch
    print(f"✅ Loaded batch. Images shape: {images.shape}, Labels shape: {labels.shape}")

    # Show 4 sample images
    fig, axs = plt.subplots(1, 4, figsize=(10, 3))
    for i in range(4):
        img = images[i].permute(1, 2, 0).numpy()  # CHW -> HWC
        axs[i].imshow(img)
        axs[i].set_title(f"Label: {labels[i].item()}")
        axs[i].axis("off")
    plt.show()
