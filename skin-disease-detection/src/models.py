# src/models.py
import torch
import torch.nn as nn
import timm

class EfficientNetB0(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True):
        """
        EfficientNet-B0 with custom classifier head.
        """
        super(EfficientNetB0, self).__init__()
        self.model = timm.create_model("efficientnet_b0", pretrained=pretrained)
        
        # Replace classifier with a 2-layer head
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)
