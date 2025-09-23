import os
import shutil
import pandas as pd
import json
from sklearn.model_selection import train_test_split
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
REMOVE_HAIR = True
RESIZE_TO = (224, 224)   # set None to keep original
USE_SYMLINK = False      # True if you prefer symlinks

# --- Load metadata ---
print(f"🔎 Looking for metadata at: {METADATA_FILE}")
if not METADATA_FILE.exists():
    raise FileNotFoundError(f"❌ Metadata file not found: {METADATA_FILE}")

df = pd.read_csv(METADATA_FILE)
print(f"✅ Metadata loaded. Shape: {df.shape}")

# --- Handle duplicates ---
duplicates = df['image_id'].duplicated().sum()
if duplicates > 0:
    print(f"⚠️ Found {duplicates} duplicate image_ids. Removing duplicates.")
    df = df.drop_duplicates(subset=['image_id'])

expected_classes = sorted(df['dx'].unique())
print(f"🏷️ Detected classes: {expected_classes}")

# --- Map class labels ---
label_mapping = {name: i for i, name in enumerate(expected_classes)}
df['label'] = df['dx'].map(label_mapping)

# --- Stratified Train/Val/Test Split ---
train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df['label'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df['label'], random_state=42)
print(f"📊 Dataset split -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# --- Create output folders ---
PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
for split_name in ["train", "val", "test"]:
    (PROCESSED_PATH / split_name).mkdir(exist_ok=True)

# --- Hair removal helper ---
def remove_hair(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(1, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    return cv2.inpaint(image, mask, 1, cv2.INPAINT_TELEA)

# --- Image copy & preprocess ---
def process_and_copy_image(row, split_dir, use_symlink=False):
    img_name = f"{row['image_id']}.jpg"
    src_path = IMG_FOLDER_1 / img_name
    if not src_path.exists():
        src_path = IMG_FOLDER_2 / img_name

    if not src_path.exists():
        print(f"⚠️ Missing image: {img_name}")
        return img_name, False

    try:
        with Image.open(src_path) as img:
            img.verify()
    except Exception:
        print(f"⚠️ Corrupted image skipped: {img_name}")
        return img_name, False

    dest_path = split_dir / img_name
    if use_symlink:
        if not dest_path.exists():
            os.symlink(src_path.resolve(), dest_path)
        return img_name, True

    # Load image with cv2 for preprocessing
    img = cv2.imread(str(src_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if REMOVE_HAIR:
        img = remove_hair(img)
    if RESIZE_TO:
        img = cv2.resize(img, RESIZE_TO, interpolation=cv2.INTER_AREA)

    cv2.imwrite(str(dest_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return img_name, True

def setup_split_directory(df_split, split_name, use_symlink=False):
    split_dir = PROCESSED_PATH / split_name
    stats = {"copied": 0, "skipped": 0}
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda row: process_and_copy_image(row, split_dir, use_symlink),
                                    df_split.to_dict('records')))
    for _, success in results:
        if success: stats["copied"] += 1
        else: stats["skipped"] += 1
    return stats

# --- Process splits ---
report = {"splits": {}, "label_mapping": label_mapping}
for split_name, df_split in zip(["train","val","test"], [train_df,val_df,test_df]):
    stats = setup_split_directory(df_split, split_name, USE_SYMLINK)
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

# --- Save preprocess integrity report ---
with open(PROCESSED_PATH / "preprocess_report.json", 'w') as f:
    json.dump(report, f, indent=4)

print(f"\n✅ Preprocessing complete! Data ready at {PROCESSED_PATH}")
print("📑 Integrity report saved at preprocess_report.json")
