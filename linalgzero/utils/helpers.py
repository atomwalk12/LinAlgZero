import logging
from pathlib import Path


class IncompatibleShapesError(ValueError):
    """Exception raised when a tensor has an incompatible shape."""

    def __init__(self, expected_shape: tuple[int, ...], actual_shape: tuple[int, ...]) -> None:
        self.expected_shape = expected_shape
        self.actual_shape = actual_shape
        super().__init__(f"Incompatible shapes. Expected {expected_shape}, got {actual_shape}")


class UninitializedError(RuntimeError):
    """Exception raised when a component is not initialized."""

    def __init__(self, component_name: str) -> None:
        self.component_name = component_name
        super().__init__(f"{component_name} is not initialized")


class FileNotFoundException(FileNotFoundError):
    """Exception raised when a file is not found."""

    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path
        super().__init__(f"File not found: {file_path}")


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
