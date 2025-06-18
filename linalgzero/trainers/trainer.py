import logging
import time
from abc import ABC, abstractmethod
from logging import Logger
from typing import Optional

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from linalgzero.experiments.config import ZeroConfig
from linalgzero.metrics.metrics import Metric
from linalgzero.utils.helpers import UninitializedError, count_n_parameters, format_time
from linalgzero.utils.session import SessionManager
from linalgzero.utils.wandb_logger import WandbLogger


class ZeroTrainer(ABC):
    def __init__(self, config: ZeroConfig):
        """Initialize the base trainer with configuration and setup core components.

        Sets up device selection, session management, logging, and initializes all trainer
        components to None. Subclasses must implement abstract methods to create model,
        optimizer, dataloaders, metrics, and loss function.

        Args:
            config (ZeroConfig): Configuration object containing training parameters,
                model settings, and experiment configuration.
        """
        self.config: ZeroConfig = config
        self.logger: Logger = logging.getLogger(__name__)

        # Training state
        self.global_step = 0
        self.best_score: float = -float("inf")

        # Setup
        self.device = self._setup_device()
        self.session_manager = SessionManager(config)
        self.wandb_logger = WandbLogger(
            config=config,
            project_name=config.wandb_project,
            run_name=config.wandb_run_name or self.session_manager.session_path.name,
            entity=config.wandb_entity or "",
        )

        # Components to be created by subclasses
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[Optimizer] = None
        self.train_dataloader: Optional[DataLoader] = None
        self.val_dataloader: Optional[DataLoader] = None
        self.train_metrics: dict[str, Metric] = {}
        self.val_metrics: dict[str, Metric] = {}
        self.criterion: Optional[nn.Module] = None

    def train(self) -> None:
        """Execute the complete training loop with validation and checkpointing.

        Runs setup(), then iterates through training steps for config.train_iterations.
        Handles periodic logging, validation, and checkpoint saving based on config
        intervals. Automatically manages model train/eval modes and W&B logging.
        """
        self._setup()

        self.logger.info("-" * 100)
        self.logger.info("Starting training.")
        if self.model:
            self.model.train()

        self.start_time = time.time()
        for step in range(1, self.config.train_iterations + 1):
            self.global_step = step

            step_results = self._train_step()
            self._on_step_end(step_results)

        self.logger.info("Training finished.")
        self.wandb_logger.finish()

    def _train_step(self) -> dict:
        """Execute one training iteration with forward/backward pass and metric updates."""
        if self.model is None:
            raise UninitializedError("Model")
        if self.optimizer is None:
            raise UninitializedError("Optimizer")

        # Track time for data fetching
        t0 = time.time()
        batch = self._get_next_batch()
        batch = self._prepare_batch(batch)
        data_fetch_time = time.time() - t0

        # Track time for model update
        t1 = time.time()
        self.optimizer.zero_grad()
        model_output = self.forward_pass(batch)
        loss = self.compute_loss(model_output, batch)
        loss.backward()
        self.optimizer.step()
        model_update_time = time.time() - t1

        # Update metrics
        for metric in self.train_metrics.values():
            metric.update(model_output, batch)

        # Visualize batch
        if self.global_step % self.config.log_media_iterations == 0:
            self.visualize_batch(model_output, batch, "train")

        return {
            "loss": loss.item(),
            "data_fetch_time": data_fetch_time,
            "model_update_time": model_update_time,
        }

    def _on_step_end(self, step_results: dict) -> None:
        """Handle post-training step operations based on configured intervals."""
        if self.global_step % self.config.print_iterations == 0:
            self._log_metrics(
                loss=step_results["loss"],
                data_fetch_time=step_results["data_fetch_time"],
                model_update_time=step_results["model_update_time"],
            )

        if self.global_step % self.config.log_loss_iterations == 0:
            self.wandb_logger.log({"train/loss": step_results["loss"]}, step=self.global_step)

        if self.global_step % self.config.val_iterations == 0:
            # Run validation
            score = self._validate()

            # Save checkpoint if new best score
            if score > self.best_score:
                self.logger.info(
                    f"New best score: {self.best_score:.3f} -> {score:.3f}. Saving best checkpoint."
                )
                self.best_score = score
                self._save_checkpoint(tag="best")
            else:
                self.logger.info(f"Score {score:.3f} did not improve from {self.best_score:.3f}.")

            self._save_checkpoint(tag="last")

            self.logger.info("-" * 100)

    def _validate(self) -> float:
        """Run full validation loop and compute metrics on validation dataset."""
        self.logger.info("-" * 100)
        self.logger.info("Running validation...")

        if self.model is None:
            raise UninitializedError("Model")
        if self.val_dataloader is None:
            raise UninitializedError("ValDataloader")

        self.model.eval()

        total_loss = 0.0
        # Run through the validation dataset and compute loss
        with torch.no_grad():
            for i, batch in tqdm(enumerate(self.val_dataloader), total=len(self.val_dataloader)):
                loss = self._validate_step(batch, i)
                total_loss += loss

        # Compute average loss
        avg_loss = total_loss / len(self.val_dataloader)
        self.logger.info(f"Validation loss: {avg_loss:.4f}")

        # Compute metrics
        self.logger.info("Metrics:")
        train_scores = {name: metric.compute() for name, metric in self.train_metrics.items()}
        val_scores = {name: metric.compute() for name, metric in self.val_metrics.items()}
        for name, score in train_scores.items():
            self.logger.info(f"Train {name}: {score:.4f}")
        for name, score in val_scores.items():
            self.logger.info(f"Validation {name}: {score:.4f}")

        # Log to W&B
        log_data = {f"val/{name}": score for name, score in val_scores.items()}
        log_data.update({f"train/{name}": score for name, score in train_scores.items()})
        log_data["val/loss"] = avg_loss
        self.wandb_logger.log(log_data, step=self.global_step)

        # Save results to disk
        results_data = {
            "global_step": self.global_step,
            "val_loss": avg_loss,
            **{f"val_{name}": score for name, score in val_scores.items()},
            **{f"train_{name}": score for name, score in train_scores.items()},
        }
        self.session_manager.save_json(results_data, "validation_results.json")

        # Reset metrics
        for metric in self.train_metrics.values():
            metric.reset()
        for metric in self.val_metrics.values():
            metric.reset()

        self.model.train()

        score = val_scores[self.config.main_val_metric]
        return score

    def _validate_step(self, batch: dict, iteration: int) -> float:
        """Process one validation batch and update metrics without gradient computation."""
        if self.model is None:
            raise UninitializedError("Model")

        # Prepare batch
        batch = self._prepare_batch(batch)
        model_output = self.forward_pass(batch)
        loss = self.compute_loss(model_output, batch)

        # Update metrics
        for metric in self.val_metrics.values():
            metric.update(model_output, batch)

        if iteration == 0:
            self.visualize_batch(model_output, batch, "val")

        return loss.item()

    def _save_checkpoint(self, tag: str) -> None:
        """Saves the current training state."""
        if self.model is None:
            raise UninitializedError("Model")
        if self.optimizer is None:
            raise UninitializedError("Optimizer")

        # Save checkpoint
        self.session_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            global_step=self.global_step,
            best_score=self.best_score,
            tag=tag,
        )

    def _load_checkpoint(self) -> None:
        """Loads the training state from a checkpoint."""
        # Load checkpoint
        checkpoint = self.session_manager.load_checkpoint(tag="last")

        if checkpoint is None:
            return

        if self.model is None:
            raise UninitializedError("Model")
        if self.optimizer is None:
            raise UninitializedError("Optimizer")

        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.global_step = checkpoint["global_step"]
        self.best_score = checkpoint["best_score"]

        self.logger.info(
            f"Loaded checkpoint from step {self.global_step} with best score {self.best_score:.3f}."
        )

    def _setup(self) -> None:
        """Initialize all trainer components and prepare for training."""

        # Load model, optimizer, criterion, dataloaders, metrics
        self.logger.info("Setting up trainer.")
        self.model = self.create_model()
        n_params = count_n_parameters(self.model)
        n_trainable_params = count_n_parameters(self.model, only_trainable=True)
        self.logger.info(
            f"Model created: {n_params:.2f}M parameters ({n_trainable_params:.2f}M trainable)."
        )
        self.model.to(self.device)
        self.optimizer = self.create_optimizer()
        self.criterion = self.create_loss()
        self.train_dataloader, self.val_dataloader = self.create_dataloaders()
        self.train_metrics, self.val_metrics = self.create_metrics()

        # Load checkpoint if it exists
        self._load_checkpoint()

        # To get batches from the training dataloader
        self._train_dataloader_iter = iter(self.train_dataloader)

        self.logger.info("Trainer setup complete.")

    def _setup_device(self) -> torch.device:
        """Sets up the device for training."""
        if torch.cuda.is_available() and self.config.gpu:
            self.logger.info("Using CUDA device.")
            return torch.device("cuda")
        self.logger.info("Using CPU device.")
        return torch.device("cpu")

    def _log_metrics(self, loss: float, data_fetch_time: float, model_update_time: float) -> None:
        """Logs metrics for the current step."""

        # Calculate time so far and time left
        time_so_far = time.time() - self.start_time
        training_time_left = (self.config.train_iterations / self.global_step - 1.0) * time_so_far
        step_duration = data_fetch_time + model_update_time
        samples_per_sec = self.config.batch_size / step_duration

        # Log metrics
        self.logger.info(
            f"Step {self.global_step}/{self.config.train_iterations} | "
            f"Loss: {loss:.4f} | "
            f"Step duration: {step_duration * 1000:.0f}ms | "
            f"Samples/s: {samples_per_sec:.1f} | "
            f"Data fetch: {data_fetch_time * 1000:.0f}ms | "
            f"Model update: {model_update_time * 1000:.0f}ms | "
            f"Time elapsed: {format_time(time_so_far)} | "
            f"Time left: {format_time(training_time_left)}"
        )

    def _get_next_batch(self) -> dict:
        """Gets the next batch from the training dataloader."""
        try:
            # Get next batch. This is used during training (not validation), because we track
            # progress in terms of the number of iterations performed.
            batch = next(self._train_dataloader_iter)
        except StopIteration:
            if self.train_dataloader:
                self._train_dataloader_iter = iter(self.train_dataloader)
                batch = next(self._train_dataloader_iter)
        batch_dict: dict = batch
        return batch_dict

    def _prepare_batch(self, batch: dict) -> dict:
        """Prepares a batch for training (e.g., moves to device)."""
        return {k: v.to(self.device) for k, v in batch.items()}

    @abstractmethod
    def create_model(self) -> nn.Module:
        """Create and return the neural network model for training.

        Returns:
            nn.Module: The model instance, ready for training. Should be compatible
                with the expected input/output format defined in forward_pass().
        """

    @abstractmethod
    def create_optimizer(self) -> Optimizer:
        """Create and return the optimizer for model parameter updates.

        Returns:
            Optimizer: Configured optimizer instance (e.g., Adam, SGD) with appropriate
                learning rate and parameters from self.model.parameters().
        """

    @abstractmethod
    def forward_pass(self, batch: dict) -> dict:
        """Execute forward pass through the model with a batch of data.

        Args:
            batch (dict): Batch data dictionary containing inputs and targets,
                already moved to the appropriate device.

        Returns:
            dict: Model outputs dictionary containing predictions and any intermediate
                results needed for loss computation and metrics.
        """

    @abstractmethod
    def compute_loss(self, model_output: dict, batch: dict) -> torch.Tensor:
        """Calculate the loss value for backpropagation.

        Args:
            model_output (dict): Dictionary containing model predictions and outputs
                from forward_pass().
            batch (dict): Original batch data containing targets and inputs.

        Returns:
            torch.Tensor: Scalar loss tensor ready for backward() call.
        """

    @abstractmethod
    def create_loss(self) -> nn.Module:
        """Create and return the loss function for training.

        Returns:
            nn.Module: Loss function instance (e.g., CrossEntropyLoss, MSELoss)
                compatible with model outputs and targets.
        """

    @abstractmethod
    def create_metrics(self) -> tuple[dict[str, Metric], dict[str, Metric]]:
        """Create training and validation metrics for performance tracking.

        Returns:
            tuple[dict[str, Metric], dict[str, Metric]]: Two dictionaries containing
                (train_metrics, val_metrics). Each metric should implement update(),
                compute(), and reset() methods.
        """

    @abstractmethod
    def create_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        """Create training and validation dataloaders.

        Returns:
            tuple[DataLoader, DataLoader]: Training and validation dataloaders
                (train_dataloader, val_dataloader) with appropriate batch sizes
                and data preprocessing.
        """

    @abstractmethod
    def visualize_batch(self, model_output: dict, batch: dict, mode: str) -> None:
        """Log visualizations of model predictions and data to W&B.

        Called periodically during training and once per validation to provide
        visual feedback on model performance.

        Args:
            model_output (dict): Model predictions and outputs from forward_pass().
            batch (dict): Input batch data and targets.
            mode (str): Either "train" or "val" to indicate the current phase.
        """
