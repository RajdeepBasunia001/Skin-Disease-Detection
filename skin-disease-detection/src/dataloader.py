import os
import sys
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pathlib import Path

# Get the absolute path of the project's root directory.
# This assumes the script is located inside a subdirectory like 'src'.
ROOT_DIR = Path(__file__).resolve().parent.parent

# Add the project's root directory to the system path.
# This makes the 'src' package discoverable.
sys.path.append(str(ROOT_DIR))

# Now, the 'src' module can be imported correctly.
from src.dataset import SkinDataset

# --- Configuration ---
PROCESSED_DIR = ROOT_DIR / "data/processed"

def get_dataloaders(batch_size=32, num_workers=4, image_size=224):
    """
    Creates and returns train, validation, and test dataloaders.

    Args:
        batch_size (int): The number of samples per batch.
                        **ADVICE for RTX 3050 4GB**: Start with 16 or 32. 
                        If you get a "CUDA out of memory" error, decrease this value.
        num_workers (int): Number of subprocesses for data loading.
                        **ADVICE for i5 11th Gen**: 4 is a good start. 
                        On Windows, set to 0 if you encounter multiprocessing errors.
        image_size (int): The size to which images will be resized (height and width).
    """
    data_paths = {
        "train": (PROCESSED_DIR / "train.csv", PROCESSED_DIR / "train"),
        "val": (PROCESSED_DIR / "val.csv", PROCESSED_DIR / "val"),
        "test": (PROCESSED_DIR / "test.csv", PROCESSED_DIR / "test"),
    }

    # ✅ Correctly pass `image_size` to the constructor
    train_ds = SkinDataset(
        csv_path=data_paths["train"][0],
        img_dir=data_paths["train"][1],
        is_train=True,
        image_size=image_size
    )
    val_ds = SkinDataset(
        csv_path=data_paths["val"][0],
        img_dir=data_paths["val"][1],
        is_train=False,
        image_size=image_size
    )
    test_ds = SkinDataset(
        csv_path=data_paths["test"][0],
        img_dir=data_paths["test"][1],
        is_train=False,
        image_size=image_size
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, 
        persistent_workers=True if num_workers > 0 else False
    )
    # Can often use a larger batch size for validation and testing
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False, 
        num_workers=num_workers, pin_memory=True, 
        persistent_workers=True if num_workers > 0 else False
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=True, 
        persistent_workers=True if num_workers > 0 else False
    )
    
    print(f"✅ DataLoaders created with batch_size={batch_size}, num_workers={num_workers}")
    return train_loader, val_loader, test_loader

def compute_class_weights(csv_path=PROCESSED_DIR / "train.csv"):
    """
    Computes class weights for an imbalanced dataset.
    This helps the model pay more attention to rarer classes during training.
    """
    df = pd.read_csv(csv_path)
    class_counts = df['label'].value_counts().sort_index().values
    
    total_samples = sum(class_counts)
    num_classes = len(class_counts)
    
    # Formula: total_samples / (num_classes * class_count_for_each_class)
    weights = total_samples / (num_classes * class_counts)
    
    return torch.tensor(weights, dtype=torch.float)

# --- Self-test block ---
if __name__ == "__main__":
    print("🔄 Testing DataLoader...")
    
    # For local testing, it's safer to use num_workers=0 to avoid multiprocessing issues.
    train_loader, _, _ = get_dataloaders(batch_size=8, num_workers=0)

    # Fetch one batch
    images, labels = next(iter(train_loader))
    print(f"✅ Loaded one batch. Images shape: {images.shape}, Labels shape: {labels.shape}")

    # Visualize 4 sample images from the batch
    fig, axs = plt.subplots(1, 4, figsize=(12, 4))
    fig.suptitle("Sample Augmented Images from DataLoader")
    for i in range(4):
        img = images[i].numpy().transpose((1, 2, 0)) # Convert from (C, H, W) to (H, W, C)
        
        # Denormalize for correct visualization
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img = std * img + mean
        img = img.clip(0, 1)

        axs[i].imshow(img)
        axs[i].set_title(f"Label: {labels[i].item()}")
        axs[i].axis("off")
    plt.show()

    # Test class weight calculation
    print("\n⚖️ Testing class weight calculation...")
    weights = compute_class_weights()
    print(f"✅ Computed class weights: {weights}")