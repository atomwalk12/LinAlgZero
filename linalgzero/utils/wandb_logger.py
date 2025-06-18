import logging
from dataclasses import asdict
from typing import Any

import wandb
from linalgzero.experiments.config import ZeroConfig


class WandbLogger:
    """Logs metrics to Weights & Biases."""

    def __init__(self, config: ZeroConfig, project_name: str, run_name: str, entity: str):
        """
        Args:
            config (ZeroConfig): The training configuration.
            project_name (str): The name of the W&B project.
            run_name (str): The name of the W&B run.
        """
        self.logger = logging.getLogger(__name__)

        # Prepare wandb tags
        tags: list[str] = []
        if config.tags:
            tags.extend(config.tags)

        try:
            wandb.init(
                project=project_name,
                entity=entity,
                name=run_name,
                config=asdict(config),
                tags=tags,
            )
            self.logger.info("Initialized W&B logger.")

        except wandb.errors.UsageError:
            self.logger.exception("Failed to initialize W&B")
            self.logger.exception(
                "Please make sure you are logged in to W&B. Run `wandb login` in your terminal."
            )
            # Handle the case where wandb is not available
            self._wandb_available = False
        else:
            self._wandb_available = True

    def log(self, data: dict[str, Any], step: int) -> None:
        """Logs a dictionary of metrics to W&B.

        Args:
            data (Dict[str, Any]): The metrics to log.
            step (int): The current training step.
        """
        if self._wandb_available:
            wandb.log(data, step=step)

    def finish(self) -> None:
        """Finishes the W&B run."""
        if self._wandb_available:
            wandb.finish()
            self.logger.info("Finished W&B run.")
