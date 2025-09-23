# src/models.py
import torch
import torch.nn as nn
import timm

class EfficientNetB0(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True):
        super(EfficientNetB0, self).__init__()
        self.model = timm.create_model("efficientnet_b0", pretrained=pretrained)
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
