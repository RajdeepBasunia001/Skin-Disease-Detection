import os
import shutil
import pandas as pd
import json
from sklearn.model_selection import GroupShuffleSplit
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import cv2
import numpy as np

# --- Configuration ---
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "HAM10000"
PROCESSED_PATH = DATA_DIR / "processed"

METADATA_FILE = RAW_DATA_PATH / "HAM10000_metadata.csv"
IMG_FOLDER_1 = RAW_DATA_PATH / "HAM10000_images_part_1"
IMG_FOLDER_2 = RAW_DATA_PATH / "HAM10000_images_part_2"

# Preprocessing options
HAIR_REMOVAL_KERNEL_SIZE = (17, 17)
RESIZE_STRATEGY = 'shortest_side_crop'
RESIZE_TO = (224, 224)
APPLY_CLAHE = True
USE_SYMLINK = False

# --- Helper functions ---
def get_image_path(image_id):
    """Finds the path for a given image_id across both folders."""
    img_name = f"{image_id}.jpg"
    path = IMG_FOLDER_1 / img_name
    if not path.exists():
        path = IMG_FOLDER_2 / img_name
    return path

def remove_hair(image, kernel_size=(17, 17)):
    """Removes hair from an image using the black-hat morphology trick."""
    if not image.any():
        return image
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    return cv2.inpaint(image, mask, 1, cv2.INPAINT_TELEA)

def apply_clahe(image):
    """Applies Contrast Limited Adaptive Histogram Equalization."""
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

def resize_and_crop(image, size):
    """Resizes shortest side to `size` and then center-crops to `size`."""
    h, w = image.shape[:2]
    target_h, target_w = size

    if h < w:
        new_h = target_h
        new_w = int(w * (new_h / h))
    else:
        new_w = target_w
        new_h = int(h * (new_w / w))
    
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    start_x = (new_w - target_w) // 2
    start_y = (new_h - target_h) // 2
    cropped_image = resized_image[start_y:start_y + target_h, start_x:start_x + target_w]

    return cropped_image

# --- Main execution ---
if __name__ == "__main__":
    print(f"🔎 Looking for metadata at: {METADATA_FILE}")
    if not METADATA_FILE.exists():
        raise FileNotFoundError(f"❌ Metadata file not found: {METADATA_FILE}")

    df = pd.read_csv(METADATA_FILE)
    print(f"✅ Metadata loaded. Shape: {df.shape}")

    duplicates = df['image_id'].duplicated().sum()
    if duplicates > 0:
        print(f"⚠️ Found {duplicates} duplicate image_ids. Removing duplicates.")
        df = df.drop_duplicates(subset=['image_id'])

    expected_classes = sorted(df['dx'].unique())
    print(f"🏷️ Detected classes: {expected_classes}")
    label_mapping = {name: i for i, name in enumerate(expected_classes)}
    df['label'] = df['dx'].map(label_mapping)

    # --- Corrected Patient-level Train/Val/Test Split ---
    print("🎬 Performing patient-level split using lesion_id...")
    
    lesion_ids = df['lesion_id'].unique()
    
    # Step 1: Split lesion_ids into train and a temporary set (for val/test)
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_lesion_indices, temp_lesion_indices = next(gss1.split(lesion_ids, groups=lesion_ids))
    
    train_lesion_ids = lesion_ids[train_lesion_indices]
    temp_lesion_ids = lesion_ids[temp_lesion_indices]

    # Step 2: Split the temporary lesion_ids into val and test sets
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    val_lesion_indices, test_lesion_indices = next(gss2.split(temp_lesion_ids, groups=temp_lesion_ids))
    
    val_lesion_ids = temp_lesion_ids[val_lesion_indices]
    test_lesion_ids = temp_lesion_ids[test_lesion_indices]

    # Create the dataframes
    train_df = df[df['lesion_id'].isin(train_lesion_ids)]
    val_df = df[df['lesion_id'].isin(val_lesion_ids)]
    test_df = df[df['lesion_id'].isin(test_lesion_ids)]

    # Sanity check for leakage
    train_patients = set(train_df['lesion_id'])
    val_patients = set(val_df['lesion_id'])
    test_patients = set(test_df['lesion_id'])
    assert not train_patients.intersection(val_patients)
    assert not train_patients.intersection(test_patients)
    assert not val_patients.intersection(test_patients)
    
    print(f"📊 Patient-level split -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # --- Create output folders ---
    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
    for split_name in ["train", "val", "test"]:
        (PROCESSED_PATH / split_name).mkdir(exist_ok=True)
        (PROCESSED_PATH / split_name / "with_hair").mkdir(exist_ok=True)
        (PROCESSED_PATH / split_name / "without_hair").mkdir(exist_ok=True)
        
    def process_and_copy_image(row, split_dir):
        img_id = row['image_id']
        src_path = get_image_path(img_id)

        if not src_path.exists():
            return img_id, False

        try:
            with Image.open(src_path) as img:
                img.verify()
        except Exception:
            return img_id, False

        img = cv2.imread(str(src_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # With hair
        img_with_hair = img.copy()
        if APPLY_CLAHE:
            img_with_hair = apply_clahe(img_with_hair)
        if RESIZE_STRATEGY == 'shortest_side_crop' and RESIZE_TO:
            img_with_hair = resize_and_crop(img_with_hair, RESIZE_TO)
        elif RESIZE_TO:
            img_with_hair = cv2.resize(img_with_hair, RESIZE_TO, interpolation=cv2.INTER_AREA)

        cv2.imwrite(str(split_dir / "with_hair" / f"{img_id}.jpg"), cv2.cvtColor(img_with_hair, cv2.COLOR_RGB2BGR))

        # Without hair
        img_without_hair = img.copy()
        if HAIR_REMOVAL_KERNEL_SIZE:
            img_without_hair = remove_hair(img_without_hair, kernel_size=HAIR_REMOVAL_KERNEL_SIZE)
        if APPLY_CLAHE:
            img_without_hair = apply_clahe(img_without_hair)
        if RESIZE_STRATEGY == 'shortest_side_crop' and RESIZE_TO:
            img_without_hair = resize_and_crop(img_without_hair, RESIZE_TO)
        elif RESIZE_TO:
            img_without_hair = cv2.resize(img_without_hair, RESIZE_TO, interpolation=cv2.INTER_AREA)

        cv2.imwrite(str(split_dir / "without_hair" / f"{img_id}.jpg"), cv2.cvtColor(img_without_hair, cv2.COLOR_RGB2BGR))
        
        return img_id, True

    def setup_split_directory(df_split, split_name):
        split_dir = PROCESSED_PATH / split_name
        stats = {"copied": 0, "skipped": 0}
        
        records = df_split.to_dict('records')
        with ThreadPoolExecutor() as executor:
            results = list(tqdm(executor.map(lambda row: process_and_copy_image(row, split_dir), records),
                                total=len(records),
                                desc=f"Processing {split_name}"))

        for _, success in results:
            if success: stats["copied"] += 1
            else: stats["skipped"] += 1
        return stats

    report = {"splits": {}, "label_mapping": label_mapping}
    for split_name, df_split in zip(["train","val","test"], [train_df, val_df, test_df]):
        stats = setup_split_directory(df_split, split_name)
        class_dist = df_split['label'].value_counts().to_dict()
        report["splits"][split_name] = {
            "num_images": len(df_split),
            "copied": stats["copied"],
            "skipped": stats["skipped"],
            "class_distribution": class_dist
        }

    # --- Save CSVs ---
    train_df.to_csv(PROCESSED_PATH / "train.csv", index=False)
    val_df.to_csv(PROCESSED_PATH / "val.csv", index=False)
    test_df.to_csv(PROCESSED_PATH / "test.csv", index=False)

    with open(PROCESSED_PATH / "label_mapping.json", 'w') as f:
        json.dump(label_mapping, f, indent=4)

    with open(PROCESSED_PATH / "preprocess_report.json", 'w') as f:
        json.dump(report, f, indent=4)

    print(f"\n✅ Preprocessing complete! Data ready at {PROCESSED_PATH}")
    print("📑 Integrity report saved at preprocess_report.json")

    # --- Dataset-specific normalization ---
    print("🧮 Calculating dataset-specific mean and std for normalization...")

    train_with_hair_path = PROCESSED_PATH / "train" / "with_hair"
    image_paths = [train_with_hair_path / f for f in os.listdir(train_with_hair_path) if f.endswith('.jpg')]

    total_sum = np.zeros(3, dtype=np.float64)
    total_sum_sq = np.zeros(3, dtype=np.float64)
    pixel_count = 0

    for path in tqdm(image_paths, desc="Calculating mean and std"):
        img = cv2.imread(str(path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float64) / 255.0

        total_sum += np.sum(img, axis=(0, 1))
        total_sum_sq += np.sum(img ** 2, axis=(0, 1))
        pixel_count += img.shape[0] * img.shape[1]

    mean = (total_sum / pixel_count).tolist()
    std = (np.sqrt((total_sum_sq / pixel_count) - np.array(mean) ** 2)).tolist()

    print(f"✅ Dataset Mean: {mean}")
    print(f"✅ Dataset Std: {std}")

    with open(PROCESSED_PATH / "dataset_stats.json", 'w') as f:
        json.dump({"mean": mean, "std": std}, f, indent=4)

    print("📊 Dataset-specific stats saved to dataset_stats.json")
    print("\n💡 For skin disease detection (dermoscopic images), dataset-specific normalization is preferred. ImageNet normalization is fine for a quick prototype, but dataset-specific stats give the best accuracy.")
