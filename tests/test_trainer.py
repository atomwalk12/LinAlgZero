from collections.abc import Iterator
from typing import Any, Union
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset

from linalgzero.experiments.config import ZeroConfig
from linalgzero.metrics.metrics import Metric
from linalgzero.trainers.trainer import ZeroTrainer
from linalgzero.utils.helpers import InvalidTypeError, UninitializedError


class MockMetric(Metric):
    """Mock metric for testing."""

    def __init__(self) -> None:
        self.score = 0.8
        self.updated = False

    @property
    def name(self) -> str:
        return "mock_accuracy"

    def update(self, model_output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> None:
        self.updated = True

    def compute(self) -> float:
        return self.score

    def reset(self) -> None:
        self.updated = False


class MockTrainer(ZeroTrainer):
    """Concrete implementation of ZeroTrainer for testing."""

    def create_model(self) -> nn.Module:
        return nn.Linear(10, 2)

    def create_optimizer(self) -> Optimizer:
        if self.model is None:
            raise UninitializedError("Model")
        return torch.optim.Adam(self.model.parameters(), lr=0.001)

    def create_loss(self) -> nn.Module:
        return nn.CrossEntropyLoss()

    def create_metrics(self) -> tuple[dict[str, Metric], dict[str, Metric]]:
        train_metrics: dict[str, Metric] = {"accuracy": MockMetric()}
        val_metrics: dict[str, Metric] = {"accuracy": MockMetric()}
        return train_metrics, val_metrics

    def create_dataloaders(self) -> tuple[DataLoader[Any], DataLoader[Any]]:
        # Create simple dummy data
        x = torch.randn(100, 10)
        y = torch.randint(0, 2, (100,))
        dataset = TensorDataset(x, y)

        train_loader: DataLoader[Any] = DataLoader(dataset, batch_size=8, shuffle=True)
        val_loader: DataLoader[Any] = DataLoader(dataset, batch_size=8, shuffle=False)
        return train_loader, val_loader

    def forward_pass(self, batch: dict) -> dict:
        inputs = batch["input"]
        if self.model is None:
            raise UninitializedError("Model")
        logits = self.model(inputs)
        return {"logits": logits}

    def compute_loss(self, model_output: dict, batch: dict) -> torch.Tensor:
        if self.criterion is None:
            raise UninitializedError("Criterion")
        loss: torch.Tensor = self.criterion(model_output["logits"], batch["target"])
        return loss

    def visualize_batch(self, model_output: dict, batch: dict, mode: str) -> None:
        # Mock visualization - just log something simple
        self.wandb_logger.log({f"{mode}/mock_viz": 1.0}, step=self.global_step)

    def _prepare_batch(self, batch: Union[dict, tuple[torch.Tensor, torch.Tensor], list]) -> dict:
        """Override to handle tuple/list from TensorDataset or dict from parent."""
        if isinstance(batch, (tuple, list)):
            x, y = batch
            return {"input": x.to(self.device), "target": y.to(self.device)}
        else:
            raise InvalidTypeError(expected_type=list, actual_type=type(batch))


class TestTrainer:
    @pytest.fixture
    def config(self) -> ZeroConfig:
        """Test fixture for the config."""
        # Create a minimal config directly without file loading
        config = ZeroConfig(
            train_iterations=5,
            val_iterations=3,
            print_iterations=2,
            log_loss_iterations=2,
            log_media_iterations=10,
            main_val_metric="accuracy",
            batch_size=8,
            learning_rate=0.001,
            weight_decay=0.01,
            n_workers=0,
            gpu=False,
            output_path="test_output",
            restore_path=None,
            tag=None,
            wandb_project="test_project",
            wandb_entity=None,
            wandb_run_name=None,
        )
        return config

    @pytest.fixture
    def mocked_dependencies(self) -> Iterator[None]:
        """Fixture that mocks trainer dependencies to avoid file I/O."""
        with (
            patch("linalgzero.trainers.trainer.SessionManager") as mock_session,
            patch("linalgzero.trainers.trainer.WandbLogger") as mock_wandb,
            patch("linalgzero.utils.session.Path") as mock_path,
            patch("builtins.open", create=True) as mock_open,
            patch("linalgzero.utils.session.subprocess.check_output") as mock_subprocess,
        ):
            # Mock Path operations to prevent directory creation
            mock_path_instance = Mock()
            mock_path_instance.exists.return_value = True
            mock_path_instance.mkdir = Mock()
            mock_path_instance.name = "test_session"
            mock_path.return_value = mock_path_instance

            # Mock SessionManager
            mock_session_instance = MagicMock()
            mock_session_instance.session_path = mock_path_instance
            mock_session_instance.save_checkpoint = MagicMock()
            mock_session_instance.load_checkpoint = MagicMock(return_value=None)
            mock_session_instance.save_json = MagicMock()
            mock_session.return_value = mock_session_instance

            # Mock WandbLogger
            mock_wandb_instance = MagicMock()
            mock_wandb_instance.log = MagicMock()
            mock_wandb_instance.finish = MagicMock()
            mock_wandb.return_value = mock_wandb_instance

            # Mock file operations
            # __enter__ basically mocks the object returned by open within the with statement
            # So, when we do something like:
            # with open("file.txt", "w") as f:
            #     f.write("Hello, world!") # does nothing
            #     f.read()                 # returns "mock_git_hash"
            # The __enter__ method is called, and we can mock the return value of the file object.
            # In this case, we mock the write and read methods.
            mock_open.return_value.__enter__.return_value.write = Mock()
            mock_open.return_value.__enter__.return_value.read = Mock(return_value="mock_git_hash")

            # Mock subprocess for git operations
            mock_subprocess.return_value = "mock_git_hash"

            yield

    @pytest.fixture
    def trainer(self, config: ZeroConfig, mocked_dependencies: None) -> MockTrainer:
        """Create a mock trainer for testing."""
        return MockTrainer(config)

    def test_trainer_initialization(self, trainer: MockTrainer) -> None:
        """Test that trainer initializes properly."""
        assert trainer.config is not None
        assert trainer.global_step == 0
        assert trainer.best_score == -float("inf")
        assert trainer.device is not None
        assert trainer.session_manager is not None
        assert trainer.wandb_logger is not None

        # Components should be None before setup
        assert trainer.model is None
        assert trainer.optimizer is None
        assert trainer.train_dataloader is None
        assert trainer.val_dataloader is None

    def test_trainer_setup(self, trainer: MockTrainer) -> None:
        """Test that trainer setup creates all necessary components."""
        trainer._setup()

        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.criterion is not None
        assert trainer.train_dataloader is not None
        assert trainer.val_dataloader is not None
        assert len(trainer.train_metrics) > 0
        assert len(trainer.val_metrics) > 0
        assert hasattr(trainer, "_train_dataloader_iter")

    def test_device_setup_cpu(self, config: ZeroConfig, mocked_dependencies: None) -> None:
        """Test device setup when GPU is disabled."""
        config.gpu = False
        trainer = MockTrainer(config)
        assert trainer.device == torch.device("cpu")

    @patch("torch.cuda.is_available", return_value=True)
    def test_device_setup_gpu(
        self, mock_cuda: Mock, config: ZeroConfig, mocked_dependencies: None
    ) -> None:
        """Test device setup when GPU is available and enabled."""
        config.gpu = True
        trainer = MockTrainer(config)
        assert trainer.device == torch.device("cuda")

    def test_train_step(self, trainer: MockTrainer) -> None:
        """Test single training step execution."""
        trainer._setup()
        trainer.start_time = 0.0  # Mock start time

        step_results = trainer._train_step()

        assert "loss" in step_results
        assert "data_fetch_time" in step_results
        assert "model_update_time" in step_results
        assert isinstance(step_results["loss"], float)

        # Check that metrics were updated
        mock_metric = trainer.train_metrics["accuracy"]
        assert isinstance(mock_metric, MockMetric)
        assert mock_metric.updated

    def test_validation_step(self, trainer: MockTrainer) -> None:
        """Test validation step execution."""
        trainer._setup()

        # Get a batch from validation dataloader
        assert trainer.val_dataloader is not None
        batch = next(iter(trainer.val_dataloader))
        loss = trainer._validate_step(batch, 0)

        assert isinstance(loss, float)
        mock_metric = trainer.val_metrics["accuracy"]
        assert isinstance(mock_metric, MockMetric)
        assert mock_metric.updated

    def test_full_validation(self, trainer: MockTrainer) -> None:
        """Test complete validation loop."""
        trainer._setup()
        trainer.global_step = 1

        score = trainer._validate()

        assert isinstance(score, float)
        assert score == 0.8  # Mock metric score

        # Check that metrics were reset after validation
        val_metric = trainer.val_metrics["accuracy"]
        train_metric = trainer.train_metrics["accuracy"]
        assert isinstance(val_metric, MockMetric)
        assert isinstance(train_metric, MockMetric)
        assert not val_metric.updated
        assert not train_metric.updated

    def test_checkpoint_saving_and_loading(self, trainer: MockTrainer) -> None:
        """Test checkpoint save and load functionality."""
        trainer._setup()
        trainer.global_step = 10
        trainer.best_score = 0.9

        # Test saving
        trainer._save_checkpoint()
        trainer.session_manager.save_checkpoint.assert_called_once()  # type: ignore[attr-defined]

        # Test loading (mock returns None, so no state should change)
        original_step = trainer.global_step
        original_score = trainer.best_score
        trainer._load_checkpoint()

        assert trainer.global_step == original_step
        assert trainer.best_score == original_score

    def test_batch_preparation(self, trainer: MockTrainer) -> None:
        """Test batch preparation moves tensors to correct device."""
        trainer._setup()

        # Create a mock batch
        assert trainer.train_dataloader is not None
        raw_batch = next(iter(trainer.train_dataloader))
        prepared_batch = trainer._prepare_batch(raw_batch)

        assert "input" in prepared_batch
        assert "target" in prepared_batch
        assert prepared_batch["input"].device == trainer.device
        assert prepared_batch["target"].device == trainer.device

    def test_on_step_end_logging(self, trainer: MockTrainer) -> None:
        """Test step end handling with different intervals."""
        trainer._setup()
        trainer.start_time = 0.0

        # Test at print interval
        trainer.global_step = trainer.config.print_iterations
        step_results = {"loss": 0.5, "data_fetch_time": 0.01, "model_update_time": 0.02}

        # Should not raise any errors
        trainer._on_step_end(step_results)

        # Verify wandb logging was called
        trainer.wandb_logger.log.assert_called()  # type: ignore[attr-defined]

    def test_on_step_end_validation(self, trainer: MockTrainer) -> None:
        """Test validation trigger during step end."""
        trainer._setup()
        trainer.start_time = 0.0

        # Set step to validation interval
        trainer.global_step = trainer.config.val_iterations
        step_results = {"loss": 0.5, "data_fetch_time": 0.01, "model_update_time": 0.02}

        initial_best_score = trainer.best_score
        trainer._on_step_end(step_results)

        # Score should have improved (mock returns 0.8)
        assert trainer.best_score > initial_best_score
        trainer.session_manager.save_checkpoint.assert_called_once()  # type: ignore[attr-defined]

    def test_get_next_batch_iteration(self, trainer: MockTrainer) -> None:
        """Test batch iteration and dataloader restart."""
        trainer._setup()

        # Get multiple batches to test iteration
        batch1 = trainer._get_next_batch()
        batch2 = trainer._get_next_batch()

        assert batch1 is not None
        assert batch2 is not None

    def test_train_integration(self, trainer: MockTrainer) -> None:
        """Test complete training integration."""
        # This is a full integration test
        trainer.train()

        # Verify training completed
        assert trainer.global_step == trainer.config.train_iterations

        # Verify components were created
        assert trainer.model is not None
        assert trainer.optimizer is not None

        # Verify wandb finish was called
        trainer.wandb_logger.finish.assert_called_once()  # type: ignore[attr-defined]

    def test_model_train_eval_modes(self, trainer: MockTrainer) -> None:
        """Test that model switches between train and eval modes appropriately."""
        trainer._setup()

        # Model should be in train mode initially
        assert trainer.model is not None
        trainer.model.train()
        assert trainer.model.training

        # Validation should switch to eval mode and back
        trainer._validate()
        assert trainer.model.training  # Should be back in train mode after validation
