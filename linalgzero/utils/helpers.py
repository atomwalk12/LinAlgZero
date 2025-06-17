import logging
from pathlib import Path

import torch.nn as nn


class IncompatibleShapesError(ValueError):
    """Exception raised when a tensor has an incompatible shape."""

    def __init__(self, expected_shape: tuple[int, ...], actual_shape: tuple[int, ...]) -> None:
        super().__init__(f"Incompatible shapes. Expected {expected_shape}, got {actual_shape}")


class UninitializedError(ValueError):
    """Exception raised when a component is not initialized."""

    def __init__(self, component_name: str) -> None:
        super().__init__(f"{component_name} is not initialized")


class FileNotFoundException(FileNotFoundError):
    """Exception raised when a file is not found."""

    def __init__(self, file_path: Path) -> None:
        super().__init__(f"File not found: {file_path}")


class InvalidTypeError(ValueError):
    """Exception raised when a type is invalid."""

    def __init__(self, expected_type: type, actual_type: type) -> None:
        super().__init__(f"Invalid type. Expected {expected_type}, got {actual_type}")


def setup_logging(log_path: Path) -> None:
    """Sets up the root logger to log to a file and the console.

    Args:
        log_path (Path): Path to the log file.
    """
    # Remove all handlers associated with the root logger object.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Basic configuration
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
    )


def get_default_config_path() -> str:
    """Get the default config file path relative to the experiments directory."""
    script_dir = Path(__file__).parent
    return str(script_dir / ".." / "experiments" / "linalgzero.yml")


def format_time(seconds: float) -> str:
    """Formats time in seconds to a string representation HH:MM:SS.

    Args:
        seconds (float): The time in seconds.

    Returns:
        str: The formatted time string.
    """
    s = int(seconds)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def count_n_parameters(model: nn.Module, only_trainable: bool = False) -> float:
    """Counts the number of parameters in a model.

    Args:
        model (torch.nn.Module): The model.
        only_trainable (bool, optional): Whether to count only trainable
            parameters. Defaults to False.

    Returns:
        float: The number of parameters in millions.
    """
    if only_trainable:
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        n_parameters = sum(p.numel() for p in model.parameters())
    return n_parameters / 10**6
