# src/evaluate.py
import os
import sys
import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import numpy as np

# --- Setup ---
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.dataloader import get_dataloaders
from src.models import EfficientNetB0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = ROOT_DIR / "outputs" / "checkpoints" / "best_model.pth"

def main():
    # --- Load Data ---
    _, val_loader, test_loader = get_dataloaders(batch_size=32, num_workers=2)

    # --- Load class names ---
    import json
    label_mapping = json.load(open(ROOT_DIR / "data/processed/label_mapping.json"))
    class_names = [k for k in sorted(label_mapping.keys(), key=lambda x: label_mapping[x])]

    # --- Load Model ---
    model = EfficientNetB0(num_classes=len(class_names))
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # --- Evaluation ---
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # --- Metrics ---
    print("\n📊 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    # --- Confusion Matrix ---
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(ROOT_DIR / "outputs" / "reports" / "confusion_matrix_eval.png")
    plt.close()

    # --- ROC-AUC ---
    try:
        auc = roc_auc_score(all_labels, np.array(all_probs), multi_class="ovr")
        print(f"🔹 Macro ROC-AUC: {auc:.4f}")
    except Exception as e:
        print(f"⚠️ ROC-AUC calculation failed: {e}")

if __name__ == "__main__":
    main()
