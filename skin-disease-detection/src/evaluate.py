# =======================================================
# src/evaluate.py
# =======================================================

import os
import sys
from pathlib import Path
import torch
import numpy as np
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import json
from tqdm import tqdm
import multiprocessing
from itertools import cycle
import torch.nn.functional as F

# --- Project root ---
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.dataloader import get_dataloaders
from src.models import EfficientNetB0
from src.dataset import SkinDataset

# --- Config ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
NUM_WORKERS = 0  # safer on Windows
CKPT_DIR = ROOT_DIR / "outputs" / "checkpoints"
REPORTS_DIR = ROOT_DIR / "outputs" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MISCLASSIFIED_DIR = REPORTS_DIR / "misclassified_samples"
MISCLASSIFIED_DIR.mkdir(parents=True, exist_ok=True)

# --- Helper functions ---
def find_latest_checkpoint(ckpt_dir: Path):
    ckpts = sorted([p for p in ckpt_dir.glob("*.pth")])
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    return ckpts[-1]

def plot_roc_curves(y_true_one_hot, y_score, classes, save_path):
    n_classes = len(classes)
    fpr, tpr, roc_auc = {}, {}, {}

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_one_hot[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    colors = cycle(plt.cm.jet(np.linspace(0, 1, n_classes)))
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f"{classes[i]} (AUC = {roc_auc[i]:.2f})")

    plt.plot([0, 1], [0, 1], "k--", lw=2, label="Random Guessing")
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"🖼 ROC-AUC curves saved to: {save_path}")

def visualize_misclassified_samples(misclassified_list, class_names, max_samples=25, save_dir=MISCLASSIFIED_DIR):
    print(f"🖼 Visualizing {min(len(misclassified_list), max_samples)} misclassified samples...")
    temp_dataset = SkinDataset(
        csv_path=ROOT_DIR / "data/processed/test.csv",
        processed_dir=ROOT_DIR / "data/processed",
        split_name="test",
        is_train=False,
        already_resized=True,
        use_without_hair=True  # change to False if you want "with_hair"
    )

    fig, axes = plt.subplots(
        nrows=int(np.ceil(min(len(misclassified_list), max_samples) / 5)),
        ncols=5, figsize=(20, 15)
    )
    axes = axes.flatten()

    for i, (img_tensor, true_label, pred_label) in enumerate(misclassified_list[:max_samples]):
        img = temp_dataset.denormalize(img_tensor).permute(1, 2, 0)
        ax = axes[i]
        ax.imshow(img.cpu().numpy())
        ax.set_title(f"True: {class_names[true_label]}\nPred: {class_names[pred_label]}",
                     fontsize=10, color='red')
        ax.axis("off")

    for j in range(len(misclassified_list), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(save_dir / "misclassified_samples.png")
    plt.close()
    print(f"🖼 Misclassified samples saved to: {save_dir / 'misclassified_samples.png'}")

# --- Main evaluation ---
def main():
    # --- Data ---
    _, _, test_loader, _ = get_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        use_sampler=False
    )

    # --- Classes ---
    label_mapping = json.load(open(ROOT_DIR / "data/processed/label_mapping.json"))
    class_names = [k for k in sorted(label_mapping.keys(), key=lambda x: label_mapping[x])]
    n_classes = len(class_names)

    # --- Checkpoint ---
    checkpoint_path = find_latest_checkpoint(CKPT_DIR)
    print("📂 Using checkpoint:", checkpoint_path)

    # --- Model ---
    model = EfficientNetB0(num_classes=n_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print("✅ Model loaded. Running evaluation on test set...")

    all_preds, all_labels, all_probs = [], [], []
    misclassified_samples = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # use correct autocast
            if DEVICE.type == "cuda":
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    probs = F.softmax(outputs, dim=1)
            else:
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)

            preds = torch.argmax(probs, dim=1)

            incorrect_indices = (preds != labels).nonzero(as_tuple=True)[0]
            for idx in incorrect_indices:
                if len(misclassified_samples) < 25:
                    misclassified_samples.append((images[idx].detach().cpu(),
                                                  labels[idx].item(),
                                                  preds[idx].item()))

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())

    all_labels_np = np.array(all_labels)
    all_preds_np = np.array(all_preds)
    all_probs_np = np.array(all_probs)

    # --- Metrics ---
    balanced_acc = balanced_accuracy_score(all_labels_np, all_preds_np)
    print(f"\n📊 Balanced Accuracy: {balanced_acc:.4f}")
    print("📋 Classification Report:")
    print(classification_report(all_labels_np, all_preds_np,
                                target_names=class_names, digits=4))

    # --- Confusion matrix ---
    cm = confusion_matrix(all_labels_np, all_preds_np)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(REPORTS_DIR / "confusion_matrix_eval.png")
    plt.close()
    print(f"🖼 Confusion matrix saved to: {REPORTS_DIR / 'confusion_matrix_eval.png'}")

    # --- ROC Curves ---
    y_true_one_hot = np.eye(n_classes)[all_labels_np]
    plot_roc_curves(y_true_one_hot, all_probs_np, class_names, REPORTS_DIR / "roc_curves.png")

    # --- Misclassified samples ---
    visualize_misclassified_samples(misclassified_samples, class_names)

    # --- Save metrics ---
    metrics = {
        "balanced_accuracy": float(balanced_acc),
        "classification_report": classification_report(
            all_labels_np, all_preds_np, target_names=class_names, output_dict=True),
        "n_test_samples": len(all_labels_np)
    }
    (REPORTS_DIR / "eval_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"📁 Metrics saved to: {REPORTS_DIR / 'eval_metrics.json'}")
    print("\n✅ Evaluation complete.")

if __name__ == "__main__":
    multiprocessing.freeze_support()  # Windows fix
    main()
