"""
Model architecture: Transfer learning on DenseNet121 (pretrained on ImageNet)
fine-tuned for binary chest X-ray classification (NORMAL vs PNEUMONIA).

Why DenseNet121?
- It is the backbone most widely validated on chest X-ray tasks in the
  medical-imaging literature (e.g. CheXNet, Rajpurkar et al. 2017).
- Its dense connectivity gives strong gradient flow, which also makes
  Grad-CAM visualizations on its last conv block very clean.
"""
import torch
import torch.nn as nn
from torchvision import models

from src.config import NUM_CLASSES


class PneumoniaDenseNet(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True,
                 freeze_backbone: bool = False):
        super().__init__()
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.densenet121(weights=weights)

        if freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False

        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)

    @property
    def target_layer(self):
        """Last convolutional layer — used as the Grad-CAM target layer."""
        return self.backbone.features.denseblock4.denselayer16.conv2


def build_model(pretrained: bool = True, freeze_backbone: bool = False) -> PneumoniaDenseNet:
    return PneumoniaDenseNet(pretrained=pretrained, freeze_backbone=freeze_backbone)


def load_trained_model(model_path: str, device: str = "cpu") -> PneumoniaDenseNet:
    """Loads a model checkpoint saved by src/train.py."""
    model = build_model(pretrained=False)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
