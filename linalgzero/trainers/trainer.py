import logging
from abc import ABC, abstractmethod
from logging import Logger
from typing import Optional

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from linalgzero.experiments.config import ZeroConfig
from linalgzero.metrics.metrics import Metric
from linalgzero.utils.session import SessionManager
from linalgzero.utils.wandb_logger import WandbLogger


class ZeroTrainer(ABC):
    def __init__(self, config: ZeroConfig):
        """Base class for all trainers.

        The trainer is responsible for creating the data, model, loss, and optimiser.
        It also handles the training and evaluation of the model.

        Args:
            config (ZeroConfig): Config object.
        """
        self.config: ZeroConfig = config
        self.logger: Logger = logging.getLogger(__name__)

        # Training state
        self.global_step = 0
        self.best_score: float = -float("inf")

        # Setup
        self.device = self._setup_device()
        self.session_manager = SessionManager(config)
        self.wandb_logger = WandbLogger(
            config=config,
            project_name=config.wandb_project,
            run_name=self.session_manager.session_path.name,
        )

        # Components to be created by subclasses
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[Optimizer] = None
        self.train_dataloader: Optional[DataLoader] = None
        self.val_dataloader: Optional[DataLoader] = None
        self.train_metrics: dict[str, Metric] = {}
        self.val_metrics: dict[str, Metric] = {}
        self.criterion: Optional[nn.Module] = None

    def _setup_device(self) -> torch.device:
        """Sets up the device for training."""
        if torch.cuda.is_available() and self.config.gpu:
            self.logger.info("Using CUDA device.")
            return torch.device("cuda")
        self.logger.info("Using CPU device.")
        return torch.device("cpu")

    @abstractmethod
    def _create_model(self) -> nn.Module:
        """Creates the model."""
        raise NotImplementedError

    @abstractmethod
    def _create_optimizer(self) -> Optimizer:
        """Creates the optimizer."""
        raise NotImplementedError

    @abstractmethod
    def _forward_pass(self, batch: dict) -> dict:
        """Performs a forward pass through the model."""
        raise NotImplementedError

    @abstractmethod
    def _compute_loss(self, model_output: dict, batch: dict) -> torch.Tensor:
        """Computes the loss."""
        raise NotImplementedError

    @abstractmethod
    def _create_loss(self) -> nn.Module:
        """Creates the loss function."""
        raise NotImplementedError

    @abstractmethod
    def _create_metrics(self) -> tuple[dict[str, Metric], dict[str, Metric]]:
        """Creates the train and validation metrics."""
        raise NotImplementedError
