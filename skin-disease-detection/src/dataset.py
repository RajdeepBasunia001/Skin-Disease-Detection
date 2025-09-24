# =======================================================
# src/dataset.py
# =======================================================

import os
import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
import json
import warnings

class SkinDataset(Dataset):
    def __init__(self, csv_path, processed_dir=None, split_name="train", is_train=True,
                 image_size=224, already_resized=True, use_without_hair=False):
        """
        Args:
            csv_path (str or Path): Path to CSV file (train.csv / val.csv / test.csv)
            processed_dir (str or Path): Path to processed data directory
            split_name (str): One of 'train', 'val', 'test'
            is_train (bool): Whether dataset is for training (apply augmentation)
            image_size (int): Target image size
            already_resized (bool): Skip resize if preprocessing already resized
            use_without_hair (bool): Use 'without_hair' images instead of 'with_hair'
        """
        self.df = pd.read_csv(csv_path)
        self.split_name = split_name
        self.is_train = is_train
        self.image_size = image_size
        self.already_resized = already_resized
        self.use_without_hair = use_without_hair

        # --- Resolve processed_dir automatically if not provided ---
        if processed_dir is None:
            ROOT_DIR = Path(__file__).resolve().parent.parent
            self.processed_dir = ROOT_DIR / "data" / "processed"
        else:
            self.processed_dir = Path(processed_dir)

        # --- Load dataset-specific normalization stats ---
        stats_path = self.processed_dir / "dataset_stats.json"
        if stats_path.exists():
            with open(stats_path, "r") as f:
                stats = json.load(f)
                self.mean = tuple(stats.get("mean", (0.485, 0.456, 0.406)))
                self.std = tuple(stats.get("std", (0.229, 0.224, 0.225)))
        else:
            warnings.warn(f"⚠️ dataset_stats.json not found at {stats_path.resolve()}. Using ImageNet defaults.")
            self.mean = (0.485, 0.456, 0.406)
            self.std = (0.229, 0.224, 0.225)

        # --- Transforms ---
        self.transform = self._get_transforms()
        self.transform_minority = self._get_minority_transforms()
        self.minority_classes = [0, 1, 3, 6]  # akiec, bcc, df, vasc

    def _get_transforms(self):
        resize_op = [] if self.already_resized else [A.Resize(self.image_size, self.image_size)]
        if self.is_train:
            return A.Compose(resize_op + [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=180, p=0.7),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.CLAHE(p=0.5),
                A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5),
                A.Normalize(mean=self.mean, std=self.std),
                ToTensorV2()
            ])
        else:
            return A.Compose(resize_op + [
                A.Normalize(mean=self.mean, std=self.std),
                ToTensorV2()
            ])

    def _get_minority_transforms(self):
        return A.Compose([
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120*0.05, alpha_affine=120*0.03, p=0.5),
                A.GridDistortion(p=0.5)
            ], p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.7),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2, p=0.7),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        split_folder = "without_hair" if self.use_without_hair else "with_hair"
        img_path = self.processed_dir / self.split_name / split_folder / f"{row['image_id']}.jpg"

        # --- Load image ---
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                raise FileNotFoundError(f"Image not found or corrupted: {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            warnings.warn(f"❌ Error loading {img_path}: {e}")
            return torch.zeros(3, self.image_size, self.image_size), torch.tensor(-1).long()

        label = torch.tensor(row["label"]).long()

        # --- Apply minority class augmentation first if applicable ---
        if self.is_train and label.item() in self.minority_classes:
            img = self.transform_minority(image=img)["image"]

        # --- Apply standard augmentations and normalization ---
        augmented = self.transform(image=img)
        img_tensor = augmented["image"]

        return img_tensor, label

    def denormalize(self, tensor):
        """Convert normalized tensor back to [0,1] image for visualization."""
        mean = torch.tensor(self.mean).view(3, 1, 1)
        std = torch.tensor(self.std).view(3, 1, 1)
        return tensor * std + mean