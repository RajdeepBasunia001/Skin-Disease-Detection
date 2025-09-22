import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class SkinDataset(Dataset):
    def __init__(self, csv_path, img_dir, is_train=True, image_size=224):
        """
        Args:
            csv_path (str or Path): Path to CSV file containing metadata.
            img_dir (str or Path): Directory containing images.
            is_train (bool): Whether dataset is for training (applies augmentations).
            image_size (int): Size to resize images to (image_size x image_size).
        """
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.is_train = is_train
        self.image_size = image_size
        self.transform = self._get_transforms()

    def _get_transforms(self):
        """Returns the appropriate image transformations based on train/val/test."""
        if self.is_train:
            return A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, f"{row['image_id']}.jpg")

        try:
            # Load image and convert BGR -> RGB
            image = cv2.imread(img_path)
            if image is None:
                raise FileNotFoundError(f"Image not found or corrupted: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"❌ Error loading image {img_path}: {e}")
            # Return dummy tensor and invalid label to avoid breaking training
            return torch.randn(3, self.image_size, self.image_size), torch.tensor(-1).long()

        # Apply augmentations
        augmented = self.transform(image=image)
        image = augmented["image"]

        label = torch.tensor(row['label']).long()
        return image, label
