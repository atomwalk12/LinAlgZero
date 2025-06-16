from unittest.mock import patch

import pytest
import wandb

from linalgzero.experiments.config import ZeroConfig
from linalgzero.utils.wandb_logger import WandbLogger


@pytest.fixture
def config_with_tag() -> ZeroConfig:
    """Test fixture for config with tag."""
    return ZeroConfig(
        # Core training arguments
        batch_size=32,
        train_iterations=100,
        n_workers=4,
        gpu=False,
        # Logging and validation frequencies
        print_iterations=10,
        log_loss_iterations=10,
        log_media_iterations=50,
        val_iterations=50,
        # Metric arguments
        main_val_metric="accuracy",
        # Optimizer arguments
        learning_rate=1e-4,
        weight_decay=0.01,
        # W&B arguments
        wandb_project="test-project",
        wandb_entity=None,
        wandb_run_name=None,
        # Path arguments
        output_path="test_output",
        tag="test-experiment",
        restore_path=None,
    )


@pytest.fixture
def config_without_tag() -> ZeroConfig:
    """Test fixture for config without tag."""
    return ZeroConfig(
        # Core training arguments
        batch_size=32,
        train_iterations=100,
        n_workers=4,
        gpu=False,
        # Logging and validation frequencies
        print_iterations=10,
        log_loss_iterations=10,
        log_media_iterations=50,
        val_iterations=50,
        # Metric arguments
        main_val_metric="accuracy",
        # Optimizer arguments
        learning_rate=1e-4,
        weight_decay=0.01,
        # W&B arguments
        wandb_project="test-project",
        wandb_entity=None,
        wandb_run_name=None,
        # Path arguments
        output_path="test_output",
        tag=None,
        restore_path=None,
    )


class TestWandbLogger:
    """Test suite for WandbLogger class."""

    @patch("linalgzero.utils.wandb_logger.wandb")
    def test_init_success_with_tag(self, mock_wandb, config_with_tag):
        """Test successful initialization with tag."""
        mock_wandb.init.return_value = None
        mock_wandb.errors = wandb.errors

        logger = WandbLogger(config_with_tag, "test-project", "test-run")

        # Verify wandb.init was called with correct parameters
        mock_wandb.init.assert_called_once()
        call_args = mock_wandb.init.call_args
        assert call_args[1]["project"] == "test-project"
        assert call_args[1]["name"] == "test-run"
        assert call_args[1]["tags"] == ["test-experiment"]
        assert call_args[1]["config"]["tag"] == "test-experiment"
        assert logger._wandb_available is True

    @patch("linalgzero.utils.wandb_logger.wandb")
    def test_init_success_without_tag(self, mock_wandb, config_without_tag):
        """Test successful initialization without tag."""
        mock_wandb.init.return_value = None
        mock_wandb.errors = wandb.errors

        logger = WandbLogger(config_without_tag, "test-project", "test-run")

        # Verify wandb.init was called with correct parameters
        mock_wandb.init.assert_called_once()
        call_args = mock_wandb.init.call_args
        assert call_args[1]["project"] == "test-project"
        assert call_args[1]["name"] == "test-run"
        assert call_args[1]["tags"] == []
        assert call_args[1]["config"]["tag"] is None
        assert logger._wandb_available is True

    @patch("linalgzero.utils.wandb_logger.wandb")
    def test_init_wandb_error(self, mock_wandb, config_with_tag):
        """Test initialization when wandb fails."""
        mock_wandb.errors = wandb.errors
        mock_wandb.init.side_effect = wandb.errors.UsageError("Not logged in")

        logger = WandbLogger(config_with_tag, "test-project", "test-run")

        assert logger._wandb_available is False

    @patch("linalgzero.utils.wandb_logger.wandb")
    def test_log_when_available(self, mock_wandb, config_with_tag):
        """Test logging when wandb is available."""
        mock_wandb.init.return_value = None
        mock_wandb.errors = wandb.errors

        logger = WandbLogger(config_with_tag, "test-project", "test-run")
        test_data = {"loss": 0.5, "accuracy": 0.8}

        logger.log(test_data, step=100)

        mock_wandb.log.assert_called_once_with(test_data, step=100)

    @patch("linalgzero.utils.wandb_logger.wandb")
    def test_log_when_not_available(self, mock_wandb, config_with_tag):
        """Test logging when wandb is not available."""
        mock_wandb.errors = wandb.errors
        mock_wandb.init.side_effect = wandb.errors.UsageError("Not logged in")

        logger = WandbLogger(config_with_tag, "test-project", "test-run")
        test_data = {"loss": 0.5}

        # Should not raise an error
        logger.log(test_data, step=100)

        mock_wandb.log.assert_not_called()

    @patch("linalgzero.utils.wandb_logger.wandb")
    def test_finish(self, mock_wandb, config_with_tag):
        """Test finish method."""
        # This mocks the wandb.init method to return None
        mock_wandb.init.return_value = None
        mock_wandb.errors = wandb.errors

        logger = WandbLogger(config_with_tag, "test-project", "test-run")
        logger.finish()

        mock_wandb.finish.assert_called_once()
