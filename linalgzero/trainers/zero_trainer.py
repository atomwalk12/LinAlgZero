import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision import transforms  # type: ignore[import-untyped]

from linalgzero.data.cifar_dataset import CifarDataset
from linalgzero.experiments.config import ZeroConfig
from linalgzero.metrics.metrics import AccuracyMetric, Metric
from linalgzero.models.SimpleCNN import SimpleCNN
from linalgzero.trainers.trainer import ZeroTrainer
from linalgzero.utils.helpers import UninitializedError


class LinAlgTrainer(ZeroTrainer):
    def __init__(self, config: ZeroConfig):
        super().__init__(config)

    def _create_model(self) -> nn.Module:
        return SimpleCNN()

    def _create_optimizer(self) -> Optimizer:
        if self.model is None:
            raise UninitializedError("Optimizer")

        parameters_with_grad = filter(lambda p: p.requires_grad, self.model.parameters())
        return torch.optim.AdamW(
            parameters_with_grad,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def _create_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        train_dataset = CifarDataset(
            root="./cifar10/dataset", train=True, download=True, transform=transform
        )
        val_dataset = CifarDataset(
            root="./cifar10/dataset", train=False, download=True, transform=transform
        )
        train_dataloader: DataLoader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.n_workers,
        )
        val_dataloader: DataLoader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.n_workers,
        )
        return train_dataloader, val_dataloader

    def _create_loss(self) -> nn.Module:
        return nn.CrossEntropyLoss()

    def _create_metrics(self) -> tuple[dict[str, Metric], dict[str, Metric]]:
        train_metrics: dict[str, Metric] = {"accuracy": AccuracyMetric()}
        val_metrics: dict[str, Metric] = {"accuracy": AccuracyMetric()}
        return train_metrics, val_metrics

    def _forward_pass(self, batch: dict) -> dict:
        images = batch["image"]
        if self.model is None:
            raise UninitializedError("Model")

        logits = self.model(images)
        return {"logits": logits}

    def _compute_loss(self, model_output: dict, batch: dict) -> torch.Tensor:
        labels = batch["label"]
        if self.criterion is None:
            raise UninitializedError("Criterion")

        loss: torch.Tensor = self.criterion(model_output["logits"], labels)
        return loss
