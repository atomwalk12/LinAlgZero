import pytest
from transformers import HfArgumentParser

from linalgzero.experiments.config import ZeroConfig, get_default_config_path
from linalgzero.trainers.zero_trainer import LinAlgTrainer


@pytest.fixture
def config() -> ZeroConfig:
    """Test fixture for the config."""
    config_path = get_default_config_path()
    parser = HfArgumentParser(ZeroConfig)

    (args,) = parser.parse_yaml_file(config_path)
    assert args is not None
    return args


def test_trainer(config: ZeroConfig):
    """Test the trainer."""
    trainer = LinAlgTrainer(config)
    assert trainer is not None
