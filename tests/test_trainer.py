import pytest
from transformers import HfArgumentParser

from linalgzero.experiments.config import ZeroConfig, get_default_config_path
from linalgzero.trainers.trainer import ZeroTrainer


@pytest.fixture
def config() -> ZeroConfig:
    """Test fixture for the config."""
    config_path = get_default_config_path()
    parser = HfArgumentParser(ZeroConfig)

    args: ZeroConfig = parser.parse_yaml_file(config_path)
    assert args is not None


def test_trainer(config: ZeroConfig):
    """Test the trainer."""
    trainer = ZeroTrainer(config)
    assert trainer is not None
