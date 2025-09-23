import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class SkinDataset(Dataset):
    def __init__(self, csv_path, img_dir, is_train=True, image_size=224, already_resized=True):
        """
        Args:
            csv_path (str): Path to CSV with metadata.
            img_dir (str): Directory with processed images.
            is_train (bool): Apply augmentation if True.
            image_size (int): Target size (used only if not already_resized).
            already_resized (bool): Skip A.Resize if preprocessing already resized.
        """
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.is_train = is_train
        self.image_size = image_size
        self.already_resized = already_resized
        self.transform = self._get_transforms()

    def _get_transforms(self):
        resize_op = [] if self.already_resized else [A.Resize(self.image_size, self.image_size)]
        
        if self.is_train:
            return A.Compose(resize_op + [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5),  # NEW
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
        else:
            return A.Compose(resize_op + [
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, f"{row['image_id']}.jpg")

        try:
            image = cv2.imread(img_path)
            if image is None:
                raise FileNotFoundError(f"Image not found or corrupted: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"❌ Error loading image {img_path}: {e}")
            # Return black placeholder to keep batch shape consistent
            return torch.zeros(3, self.image_size, self.image_size), torch.tensor(-1).long()

        augmented = self.transform(image=image)
        image = augmented["image"]
        label = torch.tensor(row['label']).long()
        return image, label
