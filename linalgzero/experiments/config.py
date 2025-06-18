from dataclasses import dataclass
from typing import Optional


@dataclass
class ZeroConfig:
    """Dataclass that stores training configuration parameters.

    Attributes:
        batch_size (int): Number of samples processed in each training batch.
        train_iterations (int): Total number of training iterations.
        learning_rate (float): The learning rate for the optimizer.
        weight_decay (float): The weight decay for the optimizer.

        gpu (bool): Whether to use the GPU.
        n_workers (int): Number of workers for the dataloader.
        seed (Optional[int]): Random seed for reproducibility. If not provided,
            the experiment will use a random seed.

        val_iterations (int): Frequency (in iterations) at which to run the
            validation loop.
        main_val_metric (str): The primary validation metric for checkpointing.

        print_iterations (int): Frequency (in iterations) at which to print
            training metrics to the console.
        log_loss_iterations (int): Frequency (in iterations) at which to log
            the training loss.
        log_media_iterations (int): Frequency (in iterations) at which to
            visualize samples and log media.

        output_path (str): The root directory for logs.
        restore_path (Optional[str]): Path to a session directory to restore.
        tags (Optional[List[str]]): A list of descriptive tags for the run.

        wandb_project (str): Project name for Weights & Biases.
        wandb_entity (str): Entity for Weights & Biases.
        wandb_run_name (Optional[str]): Run name for Weights & Biases. If None,
            the run name will be the session name where results are stored.
    """

    # Core training arguments
    batch_size: int
    train_iterations: int
    learning_rate: float
    weight_decay: float

    # Infrastructure settings
    gpu: bool
    n_workers: int

    # Validation configuration
    val_iterations: int
    main_val_metric: str

    # Logging frequencies
    print_iterations: int
    log_loss_iterations: int
    log_media_iterations: int

    # Path arguments
    output_path: str

    # W&B arguments
    wandb_project: str
    wandb_entity: str

    # Optional parameters with defaults
    seed: Optional[int] = None
    tags: Optional[list[str]] = None

    # It is recommended to not set this parameter, as it will default to the
    # session directory name.
    wandb_run_name: Optional[str] = None

    # These parameters should never be defined in the config file.
    # They are used internally to restore a session.
    restore_path: Optional[str] = None
    dataset_hash: Optional[str] = None
