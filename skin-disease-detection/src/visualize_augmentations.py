import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from PIL import Image
from pathlib import Path

# Get the absolute path of the project's root directory.
ROOT_DIR = Path(__file__).resolve().parent.parent

# Add the project's root directory to the system path.
sys.path.append(str(ROOT_DIR))

# --- Augmentation Pipeline Definition ---
# This pipeline must match the one used inside your SkinDataset class for training.
def get_train_transforms(image_size=224):
    """
    Defines the data augmentation pipeline for training.
    """
    return A.Compose([
        # FIX: Changed from height/width to the single 'size' parameter
        A.RandomResizedCrop(
            size=(image_size, image_size),
            scale=(0.8, 1.0), ratio=(0.75, 1.333), p=0.5
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

# --- Visualization Function ---
def visualize_augmentations(image_path, n_samples=5, image_size=224):
    """
    Loads a single image and applies the augmentation pipeline multiple times
    to visualize the random transformations.

    Args:
        image_path (Path or str): The path to a sample image.
        n_samples (int): The number of augmented images to display.
        image_size (int): The size to which images are resized.
    """
    if not Path(image_path).is_file():
        print(f"ERROR: Image file not found at {image_path}. Please check your path.")
        return

    # Load the original image
    original_image = np.array(Image.open(image_path).convert("RGB"))

    # Get the augmentation pipeline
    train_transforms = get_train_transforms(image_size)

    # Setup the plot
    fig, axs = plt.subplots(1, n_samples + 1, figsize=(3 * (n_samples + 1), 3))
    fig.suptitle(f"Augmentation of a Single Image", fontsize=16)

    # Display the original image
    axs[0].imshow(original_image)
    axs[0].set_title("Original")
    axs[0].axis("off")

    # Display augmented versions
    for i in range(n_samples):
        # Apply the full transform and convert to NumPy array
        transformed_image = train_transforms(image=original_image)['image']

        # Denormalize the image for correct visualization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_denorm = std * transformed_image + mean
        img_denorm = np.clip(img_denorm, 0, 1)

        # Plot the augmented image
        axs[i+1].imshow(img_denorm)
        axs[i+1].set_title(f"Augmented {i+1}")
        axs[i+1].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# --- Self-test block ---
if __name__ == "__main__":
    print("🔄 Visualizing data augmentations...")

    # --- IMPORTANT ---
    # Please replace this with the actual path to one of your image files.
    # A sample image from your 'data/processed/train' directory.
    sample_image_path = ROOT_DIR / "data/processed/train/ISIC_0024310.jpg" 

    # Check if the sample data exists before attempting to visualize
    if not sample_image_path.is_file():
        print(f"\n⚠️  Sample image not found at {sample_image_path}.")
        print("Please ensure you have run the data processing script and that the path is correct.")
        print("This script will not run until a valid image path is provided.")
    else:
        visualize_augmentations(sample_image_path)
        print("\n✅ Augmentation visualization complete. A plot window should be open.")
