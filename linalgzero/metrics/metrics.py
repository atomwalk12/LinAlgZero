import abc

import torch


class Metric(abc.ABC):
    """Abstract base class for a metric calculation."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """The name of the metric."""

    @abc.abstractmethod
    def update(self, model_output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> None:
        """Update the metric's state with a new batch of predictions and targets."""

    @abc.abstractmethod
    def compute(self) -> float:
        """Compute the final metric value."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset the metric's state."""
