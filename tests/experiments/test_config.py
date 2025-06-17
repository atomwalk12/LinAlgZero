from transformers import HfArgumentParser

from linalgzero.experiments.config import ZeroConfig, get_default_config_path


def test_default_config() -> None:
    """Test that the default config is valid."""
    config_path = get_default_config_path()
    parser = HfArgumentParser(ZeroConfig)  # type: ignore[arg-type]

    # This follows the "test the behaviour, not the implementation" principle.
    # It will fail if the config has missing entries or the data class is defined incorrectly.
    (args,) = parser.parse_yaml_file(config_path)
    assert args is not None
