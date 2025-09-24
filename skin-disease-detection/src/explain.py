# =======================================================
# src/evaluate.py
# =======================================================

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns
import multiprocessing

from src.dataset import SkinDataset
from src.dataloader import get_dataloaders

# --- Configuration ---
ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
REPORTS_DIR = ROOT_DIR / "outputs" / "reports"
MISCLASSIFIED_DIR = REPORTS_DIR / "misclassified"

BATCH_SIZE = 32

# =======================================================
# Helper functions
# =======================================================
def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"🖼 Confusion matrix saved to: {save_path}")

def plot_roc_curves(y_true_one_hot, y_score, class_names, save_path):
    plt.figure(figsize=(12, 10))
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_one_hot[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{class_name} (AUC = {roc_auc:.2f})")
    plt.plot([0,1], [0,1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"🖼 ROC-AUC curves saved to: {save_path}")

def visualize_misclassified_samples(misclassified_list, class_names, max_samples=25):
    print(f"🖼 Visualizing {min(len(misclassified_list), max_samples)} misclassified samples...")
    temp_dataset = SkinDataset(
        csv_path=PROCESSED_DIR / "test.csv",
        processed_dir=PROCESSED_DIR,
        split_name="test",
        is_train=False,
        already_resized=True,
        use_without_hair=True
    )
    fig, axes = plt.subplots(
        nrows=int(np.ceil(min(len(misclassified_list), max_samples) / 5)),
        ncols=5, figsize=(20, 15)
    )
    axes = axes.flatten()
    for i, (idx, true_label, pred_label) in enumerate(misclassified_list[:max_samples]):
        img_tensor, _ = temp_dataset[idx]
        img = img_tensor.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        axes[i].imshow(img)
        axes[i].set_title(f"T: {class_names[true_label]}\nP: {class_names[pred_label]}")
        axes[i].axis("off")
    for j in range(i+1, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    save_path = MISCLASSIFIED_DIR / "misclassified_samples.png"
    plt.savefig(save_path)
    plt.close()
    print(f"🖼 Misclassified samples saved to: {save_path}")

# =======================================================
# Main evaluation
# =======================================================
def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MISCLASSIFIED_DIR.mkdir(parents=True, exist_ok=True)

    # --- Windows multiprocessing fix ---
    if os.name == "nt":
        multiprocessing.freeze_support()
        num_workers = 0
    else:
        num_workers = min(multiprocessing.cpu_count() // 2, 4)

    # --- DataLoader ---
    _, _, test_loader, _ = get_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=num_workers,
        use_sampler=False
    )

    # --- Class names ---
    label_mapping = json.load(open(PROCESSED_DIR / "label_mapping.json"))
    class_names = [k for k, v in sorted(label_mapping.items(), key=lambda item: item[1])]
    num_classes = len(class_names)

    # --- Load model ---
    from src.models import EfficientNetB0
    model = EfficientNetB0(num_classes=num_classes, pretrained=False).cuda()
    checkpoint_path = ROOT_DIR / "outputs" / "checkpoints"
    # Load the latest checkpoint
    ckpts = sorted(checkpoint_path.glob("*.pth"))
    if len(ckpts) == 0:
        raise FileNotFoundError("No checkpoint found in outputs/checkpoints")
    model.load_state_dict(torch.load(ckpts[-1]))
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    print("⏳ Running evaluation...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            mask = labels != -1
            images, labels = images[mask].cuda(), labels[mask].cuda()
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels_np = np.array(all_labels)
    all_preds_np = np.array(all_preds)
    all_probs_np = np.array(all_probs)

    # --- Metrics ---
    balanced_acc = balanced_accuracy_score(all_labels_np, all_preds_np)
    print(f"\n📊 Balanced Accuracy: {balanced_acc:.4f}")
    print("📋 Classification Report:")
    print(classification_report(all_labels_np, all_preds_np, target_names=class_names))

    # --- Plots ---
    plot_confusion_matrix(all_labels_np, all_preds_np, class_names, REPORTS_DIR / "confusion_matrix_eval.png")

    y_true_one_hot = np.eye(num_classes)[all_labels_np]
    plot_roc_curves(y_true_one_hot, all_probs_np, class_names, REPORTS_DIR / "roc_curves.png")

    # --- Misclassified samples ---
    misclassified_samples = [(i, t, p) for i, (t, p) in enumerate(zip(all_labels_np, all_preds_np)) if t != p]
    visualize_misclassified_samples(misclassified_samples, class_names)

    print("\n✅ Evaluation complete. Reports saved to:", REPORTS_DIR)

# =======================================================
if __name__ == "__main__":
    main()
