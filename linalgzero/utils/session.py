import datetime
import json
import logging
import socket
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any, Union

import torch
import yaml
from torch.optim import Optimizer

from linalgzero.experiments.config import ZeroConfig
from linalgzero.utils.helpers import FileNotFoundException, setup_logging, xxhash_dir


class SessionManager:
    """Manages the training session, including logging and checkpointing."""

    def __init__(self, config: ZeroConfig):
        """
        Args:
            config (ZeroConfig): The training configuration.
        """
        self.config = config

        # 1. Determine the session path
        if config.restore_path:
            self.session_path = Path(config.restore_path)
            if not self.session_path.exists():
                raise FileNotFoundException(self.session_path)
        else:
            self.log_dir = Path(config.output_path)
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.session_path = self._create_session()

        # 2. Now that the path is known, set up logging
        setup_logging(self.session_path / "logs.txt")
        self.logger = logging.getLogger(__name__)

        # 3. Perform context-specific actions
        self.logger.info("#" * 100)
        if config.restore_path:
            self.logger.info(f"Restoring session from {self.session_path}")
            self._compare_git_hash()
        else:
            self.logger.info(f"Starting new session in {self.session_path}")
            self._save_config()
            self._save_git_hash()

        # Print configuration in both cases
        self._print_config()

    def _create_session(self) -> Path:
        """Creates a unique session directory with a descriptive name."""
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        hostname = socket.gethostname()

        session_name_parts = [f"session_{timestamp}", hostname]
        if self.config.tags:
            session_name_parts.extend(self.config.tags)

        session_name = "_".join(session_name_parts)
        session_path = self.log_dir / session_name
        session_path.mkdir(parents=True, exist_ok=True)

        return session_path

    def _print_config(self) -> None:
        """Prints the current configuration to the log."""
        config = yaml.dump(asdict(self.config), default_flow_style=False, indent=4, sort_keys=True)
        self.logger.info("Configuration:")
        for line in config.strip().split("\n"):
            self.logger.info(f"    {line}")

    def _save_config(self) -> None:
        """Saves the config to a YAML file in the session directory."""
        config_path = self.session_path / "config.yml"
        config = yaml.dump(asdict(self.config), default_flow_style=False, indent=4, sort_keys=True)

        with open(config_path, "w") as f:
            f.write(config)

        self.logger.info(f"Saved config to {config_path}")

    def _save_git_hash(self) -> None:
        """Saves the current git hash to a file in the session directory."""
        git_hash_path = self.session_path / "git_hash.txt"
        try:
            git_hash = subprocess.check_output(  # noqa: S603
                ["git", "rev-parse", "HEAD"],  # noqa: S607
                universal_newlines=True,
            ).strip()
            with open(git_hash_path, "w") as f:
                f.write(git_hash)
            self.logger.info(f"Saved git hash {git_hash} to {git_hash_path}")
        except subprocess.CalledProcessError:
            self.logger.warning("Could not get git hash. Not a git repository?")

    def _compare_git_hash(self) -> None:
        """Compares the current git hash with the one in the session directory."""
        git_hash_path = self.session_path / "git_hash.txt"
        if not git_hash_path.exists():
            self.logger.warning("No git hash found in session directory.")
            return

        with open(git_hash_path) as f:
            previous_git_hash = f.read().strip()

        try:
            current_git_hash = subprocess.check_output(  # noqa: S603
                ["git", "rev-parse", "HEAD"],  # noqa: S607
                universal_newlines=True,
            ).strip()
        except subprocess.CalledProcessError:
            self.logger.warning("Could not get current git hash.")
            return

        if current_git_hash != previous_git_hash:
            self.logger.warning("Restoring model with a different git hash.")
            self.logger.warning(f"Previous: {previous_git_hash}")
            self.logger.warning(f"Current:  {current_git_hash}")

    def manage_dataset_hash(self, dataset_path: Path) -> None:
        """Manages the dataset hash for reproducibility.

        For a new run, it computes and saves the dataset hash to the config.
        For a restored run, it compares the current hash with the stored one.

        Args:
            dataset_path (Path): The path to the dataset directory.
        """
        if not dataset_path.exists() or not dataset_path.is_dir():
            self.logger.warning(f"Dataset directory {dataset_path} not found. Cannot check hash.")
            return

        current_hash = xxhash_dir(dataset_path)

        if self.config.restore_path is None:
            # New run. Save the hash to the config and persist it.
            self.config.dataset_hash = current_hash
            self.logger.info(f"Dataset hash: {current_hash}")
            self._save_config()  # Overwrite config with the new hash
        else:
            # Restored run. Compare hashes.
            if self.config.dataset_hash is None:
                self.logger.warning(
                    "No dataset hash found in restored config. "
                    f"Current dataset hash is {current_hash}."
                )
            elif self.config.dataset_hash != current_hash:
                self.logger.warning(
                    "Dataset hash mismatch! "
                    f"Expected {self.config.dataset_hash}, got {current_hash}."
                )
            else:
                self.logger.info("Dataset hash matches the one from the restored session.")

    def save_json(self, data: dict[str, Any], filename: str) -> None:
        """Saves a dictionary to a JSON file in the session directory."""
        file_path = self.session_path / filename
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        self.logger.info(f"Saved results to {file_path}")

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optimizer,
        global_step: int,
        best_score: float,
        tag: str,
    ) -> None:
        """Saves the model and optimizer state."""
        checkpoint_path = self.session_path / f"checkpoint_{tag}.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "global_step": global_step,
                "best_score": best_score,
            },
            checkpoint_path,
        )
        self.logger.info(f"Saved {tag} checkpoint to {checkpoint_path}")

    def load_checkpoint(self, tag: str) -> Union[dict[str, Any], None]:
        """Loads the model and optimizer state."""
        checkpoint_path = self.session_path / f"checkpoint_{tag}.pt"
        if not checkpoint_path.exists():
            self.logger.info(f"No checkpoint found with tag '{tag}'.")
            return None

        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint: dict[str, Any] = torch.load(checkpoint_path)
        return checkpoint
