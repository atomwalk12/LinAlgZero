import argparse
from argparse import Namespace

from transformers import HfArgumentParser

from linalgzero.experiments.config import ZeroConfig, get_default_config_path


def parse_args() -> Namespace:
    """Parse command line arguments."""
    default_config = get_default_config_path()

    parser = argparse.ArgumentParser(description="Run LinAlgZero training")
    parser.add_argument(
        "--config",
        type=str,
        default=default_config,
        help=f"Path to config YAML file (default: {default_config})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    config_path = args.config
    print(f"Loading config from: {config_path}")

    # Type ignore to work around mypy issue with HfArgumentParser
    parser = HfArgumentParser((ZeroConfig,))  # type: ignore[arg-type]
    (config,) = parser.parse_yaml_file(config_path)
    print(config)
