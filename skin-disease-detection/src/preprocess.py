import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# --- Build absolute paths relative to this script ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # root folder
DATASET_PATH = os.path.join(BASE_DIR, "data", "HAM10000")
METADATA_FILE = os.path.join(DATASET_PATH, "HAM10000_metadata.csv")
IMG_FOLDER_1 = os.path.join(DATASET_PATH, "HAM10000_images_part_1")
IMG_FOLDER_2 = os.path.join(DATASET_PATH, "HAM10000_images_part_2")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "processed")

print("🔎 Looking for metadata at:", METADATA_FILE)

if not os.path.exists(METADATA_FILE):
    raise FileNotFoundError(f"❌ Metadata file not found: {METADATA_FILE}")

# --- Load metadata ---
df = pd.read_csv(METADATA_FILE)
print("✅ Metadata loaded. Shape:", df.shape)

# --- Map class labels ---
label_mapping = {label: idx for idx, label in enumerate(sorted(df['dx'].unique()))}
df['label'] = df['dx'].map(label_mapping)

# --- Train/Val/Test Split ---
train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df['label'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df['label'], random_state=42)

print(f"📊 Split -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# --- Create output folders ---
os.makedirs(OUTPUT_PATH, exist_ok=True)
for split_name in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUTPUT_PATH, split_name), exist_ok=True)

# --- Copy images ---
def copy_images(df, split_name):
    split_path = os.path.join(OUTPUT_PATH, split_name)
    for _, row in df.iterrows():
        img_name = row['image_id'] + ".jpg"
        src_path = os.path.join(IMG_FOLDER_1, img_name)
        if not os.path.exists(src_path):
            src_path = os.path.join(IMG_FOLDER_2, img_name)
        if os.path.exists(src_path):
            shutil.copy(src_path, os.path.join(split_path, img_name))
        else:
            print(f"⚠ Missing image: {img_name}")

copy_images(train_df, "train")
copy_images(val_df, "val")
copy_images(test_df, "test")

# --- Save CSVs ---
train_df.to_csv(os.path.join(OUTPUT_PATH, "train.csv"), index=False)
val_df.to_csv(os.path.join(OUTPUT_PATH, "val.csv"), index=False)
test_df.to_csv(os.path.join(OUTPUT_PATH, "test.csv"), index=False)

print("✅ Preprocessing complete! Check data/processed/")
