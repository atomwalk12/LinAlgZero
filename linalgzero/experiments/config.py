from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ZeroConfig:
    """Dataclass that stores training configuration parameters.

    Attributes:
        batch_size (int): Number of samples processed in each training batch.
        train_iterations (int): Total number of training iterations.
        n_workers (int): Number of workers for the dataloader.
        gpu (bool): Whether to use the GPU.

        print_iterations (int): Frequency (in iterations) at which to print
            training metrics to the console.
        log_loss_iterations (int): Frequency (in iterations) at which to log
            the training loss.
        log_media_iterations (int): Frequency (in iterations) at which to
            visualize samples and log media.
        val_iterations (int): Frequency (in iterations) at which to run the
            validation loop.

        output_path (str): The root directory for logs.
        tag (Optional[str]): A descriptive tag for the run.
        restore_path (Optional[str]): Path to a session directory to restore.

        main_val_metric (str): The primary validation metric for checkpointing.

        learning_rate (float): The learning rate for the optimizer.
        weight_decay (float): The weight decay for the optimizer.

        wandb_project (str): Project name for Weights & Biases.
        wandb_entity (Optional[str]): Entity for Weights & Biases.
        wandb_run_name (Optional[str]): Run name for Weights & Biases.
    """

    # Core training arguments
    batch_size: int
    train_iterations: int
    n_workers: int
    gpu: bool

    # Logging and validation frequencies
    print_iterations: int
    log_loss_iterations: int
    log_media_iterations: int
    val_iterations: int

    # Metric arguments
    main_val_metric: str

    # Optimizer arguments
    learning_rate: float
    weight_decay: float

    # W&B arguments
    wandb_project: str
    wandb_entity: Optional[str]
    wandb_run_name: Optional[str]

    # Path arguments
    output_path: str
    tag: Optional[str]
    restore_path: Optional[str] = None


def get_default_config_path() -> str:
    """Get the default config file path relative to the experiments directory."""
    script_dir = Path(__file__).parent
    return str(script_dir / "linalgzero.yml")
