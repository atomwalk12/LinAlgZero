from dataclasses import dataclass
from typing import Optional


@dataclass
class ZeroConfig:
    """Dataclass that stores training configuration parameters.

    Attributes:
        batch_size (int): Number of samples processed in each training batch.
        train_iterations (int): Total number of training iterations.
        n_workers (int): Number of workers for the dataloader.
        gpu (bool): Whether to use the GPU.
        seed (Optional[int]): Random seed for reproducibility. If not provided,
            the experiment will use a random seed.

        print_iterations (int): Frequency (in iterations) at which to print
            training metrics to the console.
        log_loss_iterations (int): Frequency (in iterations) at which to log
            the training loss.
        log_media_iterations (int): Frequency (in iterations) at which to
            visualize samples and log media.
        val_iterations (int): Frequency (in iterations) at which to run the
            validation loop.

        output_path (str): The root directory for logs.
        tags (Optional[List[str]]): A list of descriptive tags for the run.
        restore_path (Optional[str]): Path to a session directory to restore.

        main_val_metric (str): The primary validation metric for checkpointing.

        learning_rate (float): The learning rate for the optimizer.
        weight_decay (float): The weight decay for the optimizer.

        wandb_project (str): Project name for Weights & Biases.
        wandb_entity (str): Entity for Weights & Biases.
        wandb_run_name (Optional[str]): Run name for Weights & Biases. If None,
            the run name will be the session name where results are stored.
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

    # Path arguments
    output_path: str

    # W&B arguments
    wandb_project: str
    wandb_entity: str

    # It is recommended to not set this parameter, as it will default to the
    # session directory name.
    wandb_run_name: Optional[str] = None
    tags: Optional[list[str]] = None

    # This parameter should never be defined in the config file.
    # It is used internally to restore a session.
    restore_path: Optional[str] = None
    seed: Optional[int] = None
