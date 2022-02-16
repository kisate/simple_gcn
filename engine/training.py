"""

PyTorch Lightning training code.

"""

from typing import List, Dict, Optional

import torch.nn.functional as F
from pytorch_lightning import LightningModule, LightningDataModule
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_mean

from engine.model import GCN


def calculate_metrics(outputs: List[Dict]):
    targets = [elem.item() for batch in outputs for elem in batch["input"].target]
    preds = [elem.item() for batch in outputs for elem in batch["preds"]]

    return {"MSE": mean_squared_error(targets, preds), "R2": r2_score(targets, preds)}


class TrainingModule(LightningModule):
    def __init__(self, model: GCN):
        super().__init__()
        self.model = model
        self.training_metrics: List[Dict] = []
        self.validation_metrics: List[Dict] = []

    def training_step(self, batch: Data, *args):
        out = self.model(batch)
        out = scatter_mean(out, batch.batch, dim=0)
        loss = F.mse_loss(out, batch.target)

        return {"loss": loss, "input": batch, "preds": out.detach()}

    def validation_step(self, batch: Data, *args):
        out = self.model(batch)
        out = scatter_mean(out, batch.batch, dim=0)
        loss = F.mse_loss(out, batch.target)

        return {"val_loss": loss, "input": batch, "preds": out.detach()}

    def training_epoch_end(self, outputs: List[Dict]):
        metrics = calculate_metrics(outputs)
        self.training_metrics.append(metrics)

        self.log("train_loss", metrics["MSE"])
        self.log("train_r2", metrics["R2"])

    def validation_epoch_end(self, outputs: List[Dict]):
        metrics = calculate_metrics(outputs)
        self.validation_metrics.append(metrics)

        self.log("val_loss", metrics["MSE"])
        self.log("val_r2", metrics["R2"])

    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr=0.01, weight_decay=5e-4)


class TrainingDataModule(LightningDataModule):
    def __init__(self, data: List[Data], batch_size=32, test_size=0.1):
        super().__init__()
        self.data = data
        self.test_size = test_size
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None):
        self.train_data, self.val_data = train_test_split(self.data, test_size=self.test_size)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_data, batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_data, batch_size=self.batch_size)
