import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import albumentations as A
import cv2 
from pathlib import Path

# Get the absolute path of the project's root directory.
ROOT_DIR = Path(__file__).resolve().parent.parent

# Add the project's root directory to the system path.
sys.path.append(str(ROOT_DIR))

# --- Augmentation Pipeline Definition for Visualization ---
# This pipeline is identical to your training pipeline, minus normalization and ToTensorV2.
def get_visual_transforms(image_size=224):
    """
    Defines the data augmentation pipeline for visualization.
    It omits Normalize and ToTensorV2 for direct plotting.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        # FIX: Replaced ShiftScaleRotate with Affine for future-proofing
        A.Affine(scale=(0.9, 1.1), rotate=(-15, 15), translate_percent=(0.0625, 0.0625), p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        # FIX: Correctly place arguments inside CoarseDropout
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5),
    ])

# --- Visualization Function ---
def visualize_clean_augmentations(image_path, n_samples=6, image_size=224):
    """
    Loads a single image and applies the augmentation pipeline multiple times
    to visualize the random transformations with clean, non-overlapping titles.
    """
    if not Path(image_path).is_file():
        print(f"❌ ERROR: Image file not found at {image_path}. Please check your path.")
        return

    # Load the original image using OpenCV
    original_image = cv2.imread(str(image_path))
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Get the visualization pipeline
    visual_transforms = get_visual_transforms(image_size)

    # Setup the plot
    fig, axs = plt.subplots(2, 4, figsize=(15, 8))
    axs = axs.flatten()
    # Set the main title and adjust its vertical position
    fig.suptitle("Augmentation of a Single Image", fontsize=16, y=1.02)

    # Display the original image (Resized to match the augmented ones)
    resized_original = cv2.resize(original_image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    axs[0].imshow(resized_original)
    axs[0].set_title("Original (Resized)")
    axs[0].axis("off")

    # Display augmented versions
    for i in range(n_samples):
        # Apply the visual transform
        augmented = visual_transforms(image=original_image)
        augmented_image = augmented['image']
        
        axs[i+1].imshow(augmented_image)
        axs[i+1].set_title(f"Augmented {i+1}")
        axs[i+1].axis("off")

    # Hide any unused subplots
    for i in range(n_samples + 1, len(axs)):
        axs[i].axis('off')

    # Adjust spacing between subplots
    fig.subplots_adjust(top=0.9, bottom=0.08, left=0.05, right=0.95, hspace=0.25, wspace=0.15)
    plt.show()

# --- Self-test block ---
if __name__ == "__main__":
    print("🔄 Visualizing data augmentations...")

    # Define paths
    PROCESSED_DIR = ROOT_DIR / "data/processed"
    train_csv_path = PROCESSED_DIR / "train.csv"
    train_img_dir = PROCESSED_DIR / "train"

    if not train_csv_path.is_file():
        print("⚠️ Training CSV not found. Please run preprocess.py first.")
    elif not train_img_dir.is_dir():
        print(f"⚠️ Training image directory not found at {train_img_dir}.")
    else:
        # Load the CSV to get a list of image names
        df = pd.read_csv(train_csv_path)
        
        # Pick a random image from the DataFrame for visualization
        random_row = df.sample(n=1).iloc[0]
        img_id = random_row['image_id']
        sample_image_path = train_img_dir / f"{img_id}.jpg"
        
        print(f"✅ Selected a random sample from the training set: {sample_image_path.name}")
        visualize_clean_augmentations(sample_image_path)
        print("\n✅ Augmentation visualization complete. A plot window should be open.")