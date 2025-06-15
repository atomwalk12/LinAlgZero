from dataclasses import dataclass
from pathlib import Path


@dataclass
class ZeroConfig:
    """Dataclass that stores training configuration parameters.

    Attributes:
        batch_size (int): Number of samples processed in each training batch.
        train_iterations (int): Total number of training iterations.
        print_iterations (int): Frequency (in iterations) at which to print
            training metrics to the console.
        wandb_iterations (int): Frequency (in iterations) at which to log
            metrics to Weights & Biases.
        val_iterations (int): Frequency (in iterations) at which to run the
            validation loop.
    """

    batch_size: int
    train_iterations: int
    print_iterations: int
    wandb_iterations: int
    val_iterations: int


def get_default_config_path() -> str:
    """Get the default config file path relative to the experiments directory."""
    script_dir = Path(__file__).parent

    return str(script_dir / "linalgzero.yml")
