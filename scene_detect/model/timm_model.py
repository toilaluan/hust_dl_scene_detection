import timm
import torch.nn as nn
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model


class TimmModel(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=True, lora_cfg=None):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        if lora_cfg:
            config = LoraConfig(
                r=lora_cfg.r,
                lora_alpha=lora_cfg.alpha,
                lora_dropout=lora_cfg.dropout,
                target_modules=lora_cfg.target_modules,
            )
            self.model = get_peft_model(self.model, config)
            self.model.print_trainable_parameters()
        self.fc = nn.Linear(self.model.num_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        return x
