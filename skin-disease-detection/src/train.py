# train.py
import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from tqdm import tqdm # Changed from tqdm.notebook to tqdm for broader compatibility
import numpy as np
import multiprocessing # Added for os-agnostic cpu count

# --- Project root ---
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

# --- Imports ---
from src.dataloader import get_dataloaders, compute_class_weights, FocalLoss, mixup_data, mixup_criterion
from src.models import EfficientNetB0

# --- Configuration ---
EPOCHS = 50
EARLY_STOP_PATIENCE = 10
BASE_BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 2
MAX_LR = 3e-4
MODEL_NAME = "efficientnet_b0"

OUTPUT_DIR = ROOT_DIR / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
REPORTS_DIR = OUTPUT_DIR / "reports"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Run name generator ---
def get_next_run_name(model_name: str):
    date_str = datetime.now().strftime("%Y-%m-%d")
    existing = [f.name for f in CHECKPOINT_DIR.glob(f"{model_name}_run*_*.pth")]
    run_nums = []
    for f in existing:
        try:
            part = f.split("_")[1] # runX
            run_nums.append(int(part.replace("run", "")))
        except (IndexError, ValueError):
            continue
    next_run = max(run_nums) + 1 if run_nums else 1
    return f"{model_name}_run{next_run}_{date_str}.pth"

# --- Training ---
def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, scaler, epoch, use_mixup):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1} Training")

    for i, (images, labels) in progress_bar:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        with torch.cuda.amp.autocast():
            if use_mixup:
                # MixUp
                mixed_images, y_a, y_b, lam = mixup_data(images, labels, alpha=0.2)
                outputs = model(mixed_images)
                loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)


        scaler.scale(loss / GRADIENT_ACCUMULATION_STEPS).backward()

        if (i + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Removed scheduler step from inside the accumulation loop as it's typically per-batch
        # and OneCycleLR handles this correctly. If your learning rate changes too fast,
        # you might want to consider putting it after optimizer.step()

        running_loss += loss.item()

        # Note: Accuracy during a mixup epoch is not a reliable metric
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        progress_bar.set_postfix(loss=running_loss / (i + 1), accuracy=correct / total,
                                 lr=optimizer.param_groups[0]['lr'])

    # Step the scheduler at the end of the epoch
    scheduler.step()
    return running_loss / len(dataloader), correct / total

# --- Validation ---
def validate(model, dataloader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_labels, all_preds = [], []

    progress_bar = tqdm(dataloader, desc="Validating", unit="batch")
    with torch.no_grad():
        for images, labels in progress_bar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            progress_bar.set_postfix(loss=running_loss / (progress_bar.n + 1), accuracy=correct / total)

    return running_loss / len(dataloader), correct / total, all_labels, all_preds

# --- Final Test Evaluation ---
def final_evaluation(model, dataloader, class_names, checkpoint_path):
    print("\n" + "="*50)
    print("🏆 Starting Final Evaluation on the Test Set 🏆")
    print("="*50)

    # Load the best model weights
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        print(f"✅ Successfully loaded best model from: {checkpoint_path}")
    except FileNotFoundError:
        print(f"❌ Error: Checkpoint file not found at {checkpoint_path}. Skipping test evaluation.")
        return

    model.to(DEVICE)
    model.eval()

    all_labels, all_preds = [], []
    progress_bar = tqdm(dataloader, desc="Testing", unit="batch")

    with torch.no_grad():
        for images, labels in progress_bar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            with torch.cuda.amp.autocast():
                outputs = model(images)

            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    all_labels_np = np.array(all_labels)
    all_preds_np = np.array(all_preds)

    # --- Metrics ---
    print("\n📊 Test Set Performance Metrics:")
    balanced_acc = balanced_accuracy_score(all_labels_np, all_preds_np)
    print(f"  - Balanced Accuracy: {balanced_acc:.4f}")

    print("\n📋 Classification Report:")
    print(classification_report(all_labels_np, all_preds_np, target_names=class_names, digits=4))

    # --- Confusion Matrix ---
    cm = confusion_matrix(all_labels_np, all_preds_np)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.title("Confusion Matrix - Test Set", fontsize=14)
    
    report_path = REPORTS_DIR / "test_confusion_matrix.png"
    plt.savefig(report_path)
    plt.close()
    print(f"\n🖼 Confusion matrix saved to: {report_path}")
    print("="*50)


# --- Main ---
def main():
    print("✅ Using device:", DEVICE)
    if DEVICE.type == "cpu":
        print("⚠️ Warning: Training on CPU. It will be very slow.")

    # --- Data ---
    num_workers = 0 if os.name == 'nt' else multiprocessing.cpu_count() // 2
    print(f"Using {num_workers} workers.")

    train_loader, val_loader, test_loader, use_mixup = get_dataloaders(
        batch_size=BASE_BATCH_SIZE,
        num_workers=num_workers,
        use_sampler=True,
        use_mixup=True # Set to False if you want to disable mixup
    )

    # --- Classes ---
    label_mapping = json.load(open(ROOT_DIR / "data/processed/label_mapping.json"))
    class_names = [k for k, v in sorted(label_mapping.items(), key=lambda item: item[1])]
    num_classes = len(class_names)

    # --- Model ---
    model = EfficientNetB0(num_classes=num_classes, pretrained=True).to(DEVICE)

    # Freeze backbone for the first few epochs
    for param in model.model.parameters():
        param.requires_grad = False
    for param in model.model.classifier.parameters():
        param.requires_grad = True

    # --- Loss / Optimizer / Scheduler ---
    class_weights = compute_class_weights().to(DEVICE)
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=MAX_LR)
    scheduler = OneCycleLR(optimizer, max_lr=MAX_LR, epochs=EPOCHS, steps_per_epoch=len(train_loader))
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == 'cuda'))

    # --- Training Loop ---
    best_acc, patience_counter = 0.0, 0
    checkpoint_name = get_next_run_name(MODEL_NAME)
    
    # Variables to store final epoch results
    final_epoch, train_loss, train_acc, val_loss, val_acc = 0, 0, 0, 0, 0

    for epoch in range(EPOCHS):
        final_epoch = epoch
        if epoch == 5:
            print("🔓 Unfreezing all layers for fine-tuning.")
            for param in model.parameters():
                param.requires_grad = True

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, epoch, use_mixup)
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion)

        print(f"Epoch {epoch+1}/{EPOCHS} -> "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        # Early stopping & checkpoint
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), CHECKPOINT_DIR / checkpoint_name)
            print(f"✅ Checkpoint saved: {checkpoint_name}")
        else:
            patience_counter += 1
            print(f"⏳ Patience: {patience_counter}/{EARLY_STOP_PATIENCE}")
            if patience_counter >= EARLY_STOP_PATIENCE:
                print("⏹ Early stopping triggered.")
                break

    # --- Training finished ---
    print("\n📊 Training Summary")
    print(f"  Total Epochs Run      : {final_epoch + 1}")
    print(f"  Best Validation Acc   : {best_acc:.4f}")
    print(f"  Best Checkpoint       : {checkpoint_name}")
    print(f"  Last Train Loss       : {train_loss:.4f}")
    print(f"  Last Train Accuracy   : {train_acc:.4f}")
    print(f"  Last Validation Loss  : {val_loss:.4f}")
    print(f"  Last Validation Accuracy: {val_acc:.4f}")
    print("✅ Training complete.")

    # --- Final Evaluation on Test Set ---
    # Re-initialize a model instance for clean evaluation
    eval_model = EfficientNetB0(num_classes=num_classes, pretrained=False)
    final_evaluation(eval_model, test_loader, class_names, CHECKPOINT_DIR / checkpoint_name)


if __name__ == "__main__":
    main()