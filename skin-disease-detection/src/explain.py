# src/explain.py
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import json

# --- Project root ---
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.models import EfficientNetB0
from src.dataloader import get_dataloaders

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_image(image_tensor):
    """
    Convert tensor to numpy image for visualization
    """
    # Denormalize the image tensor
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    
    image = image_tensor.cpu().numpy()
    image = std * image + mean
    image = np.clip(image, 0, 1)
    
    # Convert to HWC format for plotting
    image = image.transpose(0, 2, 3, 1).squeeze()
    
    return np.uint8(255 * image)

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        # FIX: Replace register_backward_hook with register_full_backward_hook
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        self.model.zero_grad()
        output[0, class_idx].backward(retain_graph=True)

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)

        cam = cam[0].squeeze().cpu().numpy()
        cam = cv2.resize(cam, (input_tensor.size(2), input_tensor.size(3)))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

def visualize_gradcam(model, dataloader, class_names, save_dir):
    model.eval()
    gradcam = GradCAM(model, model.model.conv_head)  # Last conv layer of EfficientNetB0

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Take a few samples from test set
    count = 0
    for images, labels in dataloader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        for i in range(len(images)):
            if count >= 10:  # save 10 examples
                return

            input_tensor = images[i].unsqueeze(0)
            cam = gradcam.generate(input_tensor)

            img = preprocess_image(images[i].unsqueeze(0))
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            overlay = np.uint8(0.5 * img + 0.5 * heatmap)

            pred = torch.argmax(model(input_tensor)).item()
            true = labels[i].item()

            fig, ax = plt.subplots(1, 3, figsize=(12, 4))
            ax[0].imshow(img)
            ax[0].set_title(f"Original (True: {class_names[true]})")
            ax[0].axis("off")

            ax[1].imshow(heatmap)
            ax[1].set_title("Grad-CAM Heatmap")
            ax[1].axis("off")

            ax[2].imshow(overlay)
            ax[2].set_title(f"Overlay (Pred: {class_names[pred]})")
            ax[2].axis("off")

            plt.tight_layout()
            plt.savefig(save_dir / f"gradcam_{count}.png")
            plt.close()
            count += 1

def main():
    # --- Load Data ---
    _, _, test_loader = get_dataloaders(batch_size=1, use_sampler=False)

    # Load class names
    label_mapping = json.load(open(ROOT_DIR / "data/processed/label_mapping.json"))
    class_names = [k for k in sorted(label_mapping.keys(), key=lambda x: label_mapping[x])]

    # --- Model ---
    model = EfficientNetB0(num_classes=len(class_names)).to(DEVICE)
    checkpoint = ROOT_DIR / "outputs/checkpoints/best_model.pth"
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))

    # --- Generate Grad-CAMs ---
    visualize_gradcam(model, test_loader, class_names,
                      save_dir=ROOT_DIR / "outputs/gradcam")

    print(f"✅ Grad-CAM visualizations saved in: {ROOT_DIR / 'outputs/gradcam'}")

if __name__ == "__main__":
    main()