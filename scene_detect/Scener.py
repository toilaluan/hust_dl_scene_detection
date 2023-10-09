import pytorch_lightning as pl
from scene_detect.model.timm_model import TimmModel
import torch.nn.functional as F
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.classification import (
    MultilabelPrecision,
    MultilabelRecall,
    MultilabelAccuracy,
)


class Scener(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = TimmModel(**cfg.model)
        self.save_hyperparameters()
        self.precision_calc = MultilabelPrecision(
            num_labels=cfg.model.num_classes,
        )
        self.recall_calc = MultilabelRecall(num_labels=cfg.model.num_classes)
        self.acc_calc = MultilabelAccuracy(num_labels=cfg.model.num_classes)
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        y_hat = torch.sigmoid(y_hat)
        self.precision_calc.update(y_hat, y)
        self.recall_calc.update(y_hat, y)
        self.acc_calc.update(y_hat, y)
        return loss

    def on_validation_epoch_end(self):
        self.log("val_precision", self.precision_calc.compute(), prog_bar=True)
        self.log("val_recall", self.recall_calc.compute(), prog_bar=True)
        self.log("val_acc", self.acc_calc.compute(), prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.optimizer.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return [optimizer], [scheduler]
