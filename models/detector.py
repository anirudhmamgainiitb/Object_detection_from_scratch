from models.backbone import CustomBackbone
import torch
import torch.nn as nn

class Detector(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.backbone = CustomBackbone()
        self.head = nn.Conv2d(512, 5 + num_classes, 1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x
