import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Optional

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from linalgzero.experiments.config import ZeroConfig


class ZeroTrainer(ABC):
    def __init__(self, config: ZeroConfig):
        """Base class for all trainers.

        The trainer is responsible for creating the data, model, loss, and optimiser.
        It also handles the training and evaluation of the model.

        Args:
            config (ZeroConfig): Config object.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.output_dir = Path("outputs")  # Or get from config

        # Training state
        self.global_step = 0
        self.best_score: float = -float("inf")

        # Setup
        self.device = self._setup_device()

        # Components to be created by subclasses
        self.model: Optional[torch.nn.Module] = None
        self.optimizer: Optional[Optimizer] = None
        self.train_dataloader: Optional[DataLoader] = None
        self.val_dataloader: Optional[DataLoader] = None
        self._train_dataloader_iter: Optional[Iterator[dict]] = None

    def _setup_device(self) -> torch.device:
        """Sets up the device for training."""
        if torch.cuda.is_available():
            self.logger.info("Using CUDA device.")
            return torch.device("cuda")
        self.logger.info("Using CPU device.")
        return torch.device("cpu")

    @abstractmethod
    def _create_model(self) -> torch.nn.Module:
        """Creates the model."""

    @abstractmethod
    def _create_optimizer(self) -> Optimizer:
        """Creates the optimizer."""

    @abstractmethod
    def _create_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        """Creates the train and validation dataloaders."""
