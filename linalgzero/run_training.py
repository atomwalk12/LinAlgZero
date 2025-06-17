import argparse
from pathlib import Path

from transformers import HfArgumentParser

from linalgzero.experiments.config import ZeroConfig
from linalgzero.trainers.zero_trainer import LinAlgTrainer
from linalgzero.utils.helpers import FileNotFoundException


def main() -> None:
    """Main entry point for training."""
    arg_parser = argparse.ArgumentParser(
        description="Run LinAlgZero training.",
        epilog="Use --config for new runs. Use --restore to resume a run.",
    )
    arg_parser.add_argument("--config", type=str, help="Path to a config YAML for a new run.")
    arg_parser.add_argument("--restore", type=str, help="Path to a session directory to restore.")
    args = arg_parser.parse_args()

    config: ZeroConfig

    if args.restore:
        # --- RESTORE MODE ---
        print(f"Attempting to restore session from: {args.restore}")
        restore_path = Path(args.restore)
        config_path = restore_path / "config.yml"

        if not config_path.is_file():
            raise FileNotFoundException(config_path)

        # Load the config *from the session directory*
        hf_parser = HfArgumentParser(ZeroConfig)  # type: ignore[arg-type]
        (config,) = hf_parser.parse_yaml_file(str(config_path))

        # Set the restore path on the object so the SessionManager knows what to do
        config.restore_path = str(restore_path)

    elif args.config:
        # --- NEW RUN MODE ---
        print(f"Starting new run with config: {args.config}")
        hf_parser = HfArgumentParser(ZeroConfig)  # type: ignore[arg-type]
        (config,) = hf_parser.parse_yaml_file(args.config)

    else:
        # --- INVALID ---
        arg_parser.error("You must specify either --config for a new run or --restore to resume.")

    trainer = LinAlgTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
