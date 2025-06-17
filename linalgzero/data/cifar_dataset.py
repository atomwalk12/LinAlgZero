# torchvision does not have type hints. See https://github.com/pytorch/vision/issues/2025
from torchvision.datasets import CIFAR10  # type: ignore[import-untyped]


class CifarDataset(CIFAR10):  # type: ignore[no-any-unimported]
    """
    A wrapper for the CIFAR10 dataset that returns a dictionary
    instead of a tuple.
    """

    def __getitem__(self, index: int) -> dict:
        """
        Args:
            index (int): Index

        Returns:
            dict: {"image": image, "label": label}
        """
        # The transform passed to __init__ has already been applied.
        # We just change the return type from a tuple to a dict.
        image, label = super().__getitem__(index)
        return {"image": image, "label": label}
