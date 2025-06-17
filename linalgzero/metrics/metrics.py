import abc

import torch

from linalgzero.utils.helpers import IncompatibleShapesError


class Metric(abc.ABC):
    """Abstract base class for a metric calculation."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """The name of the metric."""
        raise NotImplementedError

    @abc.abstractmethod
    def update(self, model_output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> None:
        """Update the metric's state with a new batch of predictions and targets."""
        raise NotImplementedError

    @abc.abstractmethod
    def compute(self) -> float:
        """Compute the final metric value."""
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset the metric's state."""
        raise NotImplementedError


class AccuracyMetric(Metric):
    """Computes accuracy for classification tasks."""

    def __init__(self) -> None:
        self.correct = 0.0
        self.total = 0
        self.reset()

    @property
    def name(self) -> str:
        return "accuracy"

    def update(self, model_output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> None:
        """Update accuracy with a new batch of predictions and targets.

        Expects `model_output` to have a 'logits' key and `batch` to have a 'labels' key.
        """
        logits = model_output["logits"]
        labels = batch["label"]

        pred = torch.argmax(logits.detach(), dim=-1)
        if pred.shape != labels.shape:
            raise IncompatibleShapesError(expected_shape=labels.shape, actual_shape=pred.shape)

        self.correct += (pred == labels).sum().item()
        self.total += labels.numel()

    def compute(self) -> float:
        """Compute the final accuracy."""
        if self.total == 0:
            return 0.0
        return self.correct / self.total

    def reset(self) -> None:
        """Reset the accuracy state."""
        self.correct = 0
        self.total = 0
