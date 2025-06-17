from collections.abc import Iterator
from unittest.mock import Mock, patch

import pytest
from transformers import HfArgumentParser

from linalgzero.experiments.config import ZeroConfig, get_default_config_path
from linalgzero.trainers.zero_trainer import LinAlgTrainer


class TestTrainer:
    @pytest.fixture
    def config(self) -> ZeroConfig:
        """Test fixture for the config."""
        config_path = get_default_config_path()
        parser = HfArgumentParser(ZeroConfig)  # type: ignore[arg-type]

        (args,) = parser.parse_yaml_file(config_path)
        assert args is not None
        return args

    @pytest.fixture
    def mocked_trainer(self) -> Iterator[None]:
        """Fixture that mocks all trainer dependencies to avoid file I/O."""
        with (
            patch("linalgzero.trainers.trainer.SessionManager") as mock_session,
            patch("linalgzero.trainers.zero_trainer.CifarDataset") as mock_dataset,
        ):
            # Mock SessionManager to avoid creating a session directory
            mock_session_instance = Mock()
            mock_session_instance.session_path.name = "test_session"
            mock_session.return_value = mock_session_instance

            # Mock the dataset to avoid downloading CIFAR10
            mock_dataset.return_value = Mock()

            yield

    def test_trainer(self, config: ZeroConfig, mocked_trainer: None) -> None:
        """Test the trainer."""
        # Fixture purpose is the side-effect of mocking the dependencies
        trainer = LinAlgTrainer(config)
        assert trainer is not None

    def test_trainer_train(self, config: ZeroConfig, mocked_trainer: None) -> None:
        """Test the trainer."""
        # Fixture purpose is the side-effect of mocking the dependencies
        trainer = LinAlgTrainer(config)
        assert trainer is not None
