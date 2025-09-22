import os
import sys
import torch
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.dataset import SkinDataset

def get_dataloaders(batch_size=32, num_workers=4):
    """Return train, validation, and test dataloaders."""
    # Create absolute paths for the data files
    train_csv_path = os.path.join(ROOT_DIR, "data/processed/train.csv")
    train_img_dir = os.path.join(ROOT_DIR, "data/processed/train")
    
    val_csv_path = os.path.join(ROOT_DIR, "data/processed/val.csv")
    val_img_dir = os.path.join(ROOT_DIR, "data/processed/val")
    
    test_csv_path = os.path.join(ROOT_DIR, "data/processed/test.csv")
    test_img_dir = os.path.join(ROOT_DIR, "data/processed/test")

    # Pass the absolute paths to the SkinDataset constructor
    train_ds = SkinDataset(train_csv_path, train_img_dir, is_train=True)
    val_ds = SkinDataset(val_csv_path, val_img_dir, is_train=False)
    test_ds = SkinDataset(test_csv_path, test_img_dir, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader

def compute_class_weights(csv_path="data/processed/train.csv"):
    """Compute class weights for imbalanced dataset."""
    # Ensure this function also uses an absolute path
    abs_csv_path = os.path.join(ROOT_DIR, csv_path)
    df = pd.read_csv(abs_csv_path)
    class_counts = df['label'].value_counts().sort_index().values
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * len(class_counts)
    return torch.tensor(weights, dtype=torch.float)

# --------------------------
# Self-test when run directly
# --------------------------
if __name__ == "__main__":
    print("🔄 Testing DataLoader...")
    # It's a good practice to set num_workers=0 for local testing to avoid multiprocessing issues
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=8, num_workers=0)

    # The rest of your code should now run correctly
    batch = next(iter(train_loader))
    images, labels = batch
    print(f"✅ Loaded batch. Images shape: {images.shape}, Labels shape: {labels.shape}")

    # Show 4 sample images
    fig, axs = plt.subplots(1, 4, figsize=(10, 3))
    for i in range(4):
        # Denormalize and clip the image tensor
        img = images[i].numpy().transpose((1, 2, 0)) # Convert from (C, H, W) to (H, W, C)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img = img * std + mean
        img = img.clip(0, 1)

        axs[i].imshow(img)
        axs[i].set_title(f"Label: {labels[i].item()}")
        axs[i].axis("off")
    plt.show()
