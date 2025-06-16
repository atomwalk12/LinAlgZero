from torch import nn
from torch.optim import SGD, Optimizer
from torch.utils.data import DataLoader

from linalgzero.experiments.config import ZeroConfig
from linalgzero.trainers.trainer import ZeroTrainer


class LinAlgTrainer(ZeroTrainer):
    def __init__(self, config: ZeroConfig):
        super().__init__(config)

    def _create_model(self) -> nn.Module:
        # Use a real simple model
        return nn.Linear(1, 1)

    def _create_optimizer(self) -> Optimizer:
        model = self._create_model()
        return SGD(model.parameters(), lr=0.01)

    def _create_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        # Create mock dataloaders that satisfy the type requirement
        # This is a placeholder implementation for testing
        raise NotImplementedError("Mock dataloaders not implemented")
