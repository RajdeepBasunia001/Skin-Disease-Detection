# src/train.py
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- Project root ---
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.dataloader import get_dataloaders, compute_class_weights
from src.models import EfficientNetB0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-4

OUTPUT_DIR = ROOT_DIR / "outputs"
(OUTPUT_DIR / "checkpoints").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "reports").mkdir(parents=True, exist_ok=True)

def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in dataloader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / len(dataloader), correct / total

def validate(model, dataloader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_labels, all_preds = [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    return (running_loss / len(dataloader),
            correct / total,
            all_labels,
            all_preds)

def plot_confusion_matrix(y_true, y_pred, classes, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    if save_path:
        plt.savefig(save_path)
    plt.close()

def main():
    # --- Load Data ---
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE, use_sampler=True)

    # Get number of classes
    import json
    label_mapping = json.load(open(ROOT_DIR / "data/processed/label_mapping.json"))
    class_names = [k for k in sorted(label_mapping.keys(), key=lambda x: label_mapping[x])]

    # --- Model ---
    model = EfficientNetB0(num_classes=len(class_names)).to(DEVICE)

    # Class weights for imbalance
    class_weights = compute_class_weights().to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # FIX: Use AdamW optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    
    # FIX: Add a CosineAnnealingLR scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # --- Training Loop ---
    best_acc = 0.0
    for epoch in range(EPOCHS):
        start = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, y_true, y_pred = validate(model, val_loader, criterion)
        
        # FIX: Step the scheduler after each epoch
        scheduler.step()

        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} | "
              f"Time: {time.time()-start:.2f}s | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}") # Print current learning rate

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), OUTPUT_DIR / "checkpoints/best_model.pth")

    # --- Final Evaluation ---
    model.load_state_dict(torch.load(OUTPUT_DIR / "checkpoints/best_model.pth"))
    _, test_acc, y_true, y_pred = validate(model, test_loader, criterion)

    print("\n✅ Test Accuracy:", test_acc)
    print("\n📑 Classification Report:\n", classification_report(y_true, y_pred, target_names=class_names))

    plot_confusion_matrix(y_true, y_pred, class_names,
                          save_path=OUTPUT_DIR / "reports/confusion_matrix.png")

if __name__ == "__main__":
    main()