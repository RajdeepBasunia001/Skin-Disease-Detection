import os
import shutil
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

# --- Configuration ---
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "HAM10000"
PROCESSED_PATH = DATA_DIR / "processed"

METADATA_FILE = RAW_DATA_PATH / "HAM10000_metadata.csv"
IMG_FOLDER_1 = RAW_DATA_PATH / "HAM10000_images_part_1"
IMG_FOLDER_2 = RAW_DATA_PATH / "HAM10000_images_part_2"

# --- Load metadata ---
print(f"🔎 Looking for metadata at: {METADATA_FILE}")
if not METADATA_FILE.exists():
    raise FileNotFoundError(f"❌ Metadata file not found: {METADATA_FILE}")

df = pd.read_csv(METADATA_FILE)
print(f"✅ Metadata loaded. Shape: {df.shape}")

# --- Handle duplicates and validate metadata ---
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
train_df, temp_df = train_test_split(
    df, test_size=0.30, stratify=df['label'], random_state=42
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, stratify=temp_df['label'], random_state=42
)
print(f"📊 Dataset split -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# --- Create output folders ---
PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
for split_name in ["train", "val", "test"]:
    (PROCESSED_PATH / split_name).mkdir(exist_ok=True)

# --- Image copy function with validation and optional symlink ---
def copy_image(row, split_dir, use_symlink=False):
    img_name = f"{row['image_id']}.jpg"
    src_path = IMG_FOLDER_1 / img_name
    if not src_path.exists():
        src_path = IMG_FOLDER_2 / img_name

    if src_path.exists():
        try:
            # Validate image
            with Image.open(src_path) as img:
                img.verify()
        except Exception:
            print(f"⚠️ Corrupted image skipped: {img_name}")
            return img_name, False

        dest_path = split_dir / img_name
        if use_symlink:
            if not dest_path.exists():
                os.symlink(src_path.resolve(), dest_path)
        else:
            shutil.copy(src_path, dest_path)
        return img_name, True
    else:
        print(f"⚠️ Missing image: {img_name}")
        return img_name, False

def setup_split_directory(df_split, split_name, use_symlink=False):
    split_dir = PROCESSED_PATH / split_name
    missing_files = 0

    # Parallel copy
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda row: copy_image(row, split_dir, use_symlink), df_split.to_dict('records')))

    missing_files = sum(1 for _, success in results if not success)
    if missing_files > 0:
        print(f"⚠️ {missing_files} missing or corrupted files in {split_name}")

# --- Copy images for each split ---
setup_split_directory(train_df, "train")
setup_split_directory(val_df, "val")
setup_split_directory(test_df, "test")

# --- Save CSVs and label mapping ---
train_df.to_csv(PROCESSED_PATH / "train.csv", index=False)
val_df.to_csv(PROCESSED_PATH / "val.csv", index=False)
test_df.to_csv(PROCESSED_PATH / "test.csv", index=False)

with open(PROCESSED_PATH / "label_mapping.json", 'w') as f:
    json.dump(label_mapping, f, indent=4)

# --- Save image metadata (size, mode) ---
def save_image_metadata(df_split, split_name):
    metadata = []
    for img_id in df_split['image_id']:
        for folder in [IMG_FOLDER_1, IMG_FOLDER_2]:
            img_path = folder / f"{img_id}.jpg"
            if img_path.exists():
                try:
                    with Image.open(img_path) as img:
                        metadata.append({
                            'image_id': img_id,
                            'split': split_name,
                            'width': img.width,
                            'height': img.height,
                            'mode': img.mode
                        })
                except:
                    continue
                break
    return metadata

all_metadata = []
for split_name, df_split in zip(["train","val","test"], [train_df,val_df,test_df]):
    all_metadata.extend(save_image_metadata(df_split, split_name))

pd.DataFrame(all_metadata).to_csv(PROCESSED_PATH / "image_metadata.csv", index=False)

print(f"\n✅ Preprocessing complete! Data ready at {PROCESSED_PATH}")
