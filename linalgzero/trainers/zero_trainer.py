from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision import transforms  # type: ignore[import-untyped]

import wandb
from linalgzero.data.cifar_dataset import CifarDataset
from linalgzero.experiments.config import ZeroConfig
from linalgzero.metrics.accuracy import AccuracyMetric
from linalgzero.metrics.metrics import Metric
from linalgzero.models.simple_cnn import SimpleCNN
from linalgzero.trainers.trainer import ZeroTrainer
from linalgzero.utils.helpers import UninitializedError


class LinAlgTrainer(ZeroTrainer):
    def __init__(self, config: ZeroConfig):
        super().__init__(config)

    def create_model(self) -> nn.Module:
        return SimpleCNN()

    def create_optimizer(self) -> Optimizer:
        if self.model is None:
            raise UninitializedError("Model")

        parameters_with_grad = filter(lambda p: p.requires_grad, self.model.parameters())
        return torch.optim.AdamW(
            parameters_with_grad,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def create_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        train_dataset = CifarDataset(
            root=Path(self.config.output_path) / "dataset",
            train=True,
            download=True,
            transform=transform,
        )
        val_dataset = CifarDataset(
            root=Path(self.config.output_path) / "dataset",
            train=False,
            download=True,
            transform=transform,
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

    def create_loss(self) -> nn.Module:
        return nn.CrossEntropyLoss()

    def create_metrics(self) -> tuple[dict[str, Metric], dict[str, Metric]]:
        train_metrics: dict[str, Metric] = {"accuracy": AccuracyMetric()}
        val_metrics: dict[str, Metric] = {"accuracy": AccuracyMetric()}
        return train_metrics, val_metrics

    def forward_pass(self, batch: dict) -> dict:
        images = batch["image"]
        if self.model is None:
            raise UninitializedError("Model")

        logits = self.model(images)
        return {"logits": logits}

    def compute_loss(self, model_output: dict, batch: dict) -> torch.Tensor:
        labels = batch["label"]
        if self.criterion is None:
            raise UninitializedError("Criterion")

        loss: torch.Tensor = self.criterion(model_output["logits"], labels)
        return loss

    def visualize_batch(self, model_output: dict, batch: dict, mode: str) -> None:
        images = batch["image"].cpu()
        labels = batch["label"].cpu()
        logits = model_output["logits"].cpu()
        preds = torch.argmax(logits, dim=-1)

        # Log a table of image predictions to W&B
        class_names = [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]
        table = wandb.Table(columns=["image", "ground_truth", "prediction"])
        for img, gt, pred in zip(images, labels, preds):
            table.add_data(wandb.Image(img), class_names[gt], class_names[pred])
        self.wandb_logger.log({f"{mode}/predictions": table}, step=self.global_step)
