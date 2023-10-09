import timm
import torch.nn as nn
import torch
import torch.nn.functional as F


class TimmModel(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        for param in self.model.parameters():
            param.requires_grad = False
        self.fc = nn.Linear(self.model.num_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        return x
