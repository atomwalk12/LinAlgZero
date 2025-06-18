import json
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest
import torch
import yaml
from torch.optim import SGD

from linalgzero.experiments.config import ZeroConfig
from linalgzero.utils.session import SessionManager


@pytest.fixture
def config() -> ZeroConfig:
    """Test fixture for a basic config."""
    return ZeroConfig(
        # Core training arguments
        batch_size=32,
        train_iterations=100,
        n_workers=4,
        gpu=False,
        seed=42,
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
        tags=["test"],
        restore_path=None,
    )


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


def test_session_manager_init(config: ZeroConfig, temp_dir: Path) -> None:
    """Test SessionManager initialization creates session directory."""
    config.output_path = str(temp_dir)

    session_manager = SessionManager(config)

    assert session_manager.session_path.exists()
    assert session_manager.session_path.parent == temp_dir

    # Verify log file was created
    assert (session_manager.session_path / "logs.txt").exists()

    # Verify git hash file was created (if in a git repo)
    git_hash_path = session_manager.session_path / "git_hash.txt"
    if git_hash_path.exists():
        # If git hash file exists, verify it contains a hash
        with open(git_hash_path) as f:
            git_hash = f.read().strip()
            assert len(git_hash) > 0

    # Verify config file was saved
    config_path = session_manager.session_path / "config.yml"
    assert config_path.exists()
    with open(config_path) as f:
        saved_config = yaml.safe_load(f)
        assert saved_config["batch_size"] == config.batch_size
        assert saved_config["tags"] == config.tags


def test_save_json(config: ZeroConfig, temp_dir: Path) -> None:
    """Test saving JSON data to session directory."""
    config.output_path = str(temp_dir)

    session_manager = SessionManager(config)

    test_data = {"key": "value", "number": 42}
    filename = "test_data.json"

    session_manager.save_json(test_data, filename)

    json_path = session_manager.session_path / filename
    assert json_path.exists()

    with open(json_path) as f:
        loaded_data = json.load(f)

    assert loaded_data == test_data


def test_save_and_load_checkpoint(config: ZeroConfig, temp_dir: Path) -> None:
    """Test saving and loading model checkpoint."""
    config.output_path = str(temp_dir)

    session_manager = SessionManager(config)

    # Create simple model and optimizer
    model = torch.nn.Linear(10, 1)
    optimizer = SGD(model.parameters(), lr=0.01)
    global_step = 100
    best_score = 0.95

    # Save checkpoint
    session_manager.save_checkpoint(model, optimizer, global_step, best_score, tag="last")

    # Load checkpoint
    checkpoint = session_manager.load_checkpoint("last")

    assert checkpoint is not None
    assert "model" in checkpoint
    assert "optimizer" in checkpoint
    assert checkpoint["global_step"] == global_step
    assert checkpoint["best_score"] == best_score


def test_load_checkpoint_no_file(config: ZeroConfig, temp_dir: Path) -> None:
    """Test loading checkpoint when no checkpoint file exists."""
    config.output_path = str(temp_dir)

    session_manager = SessionManager(config)

    checkpoint = session_manager.load_checkpoint("last")

    assert checkpoint is None


def test_session_manager_with_restore_path(config: ZeroConfig, temp_dir: Path) -> None:
    """Test SessionManager initialization with restore path."""
    # Create a session directory first
    session_dir = temp_dir / "existing_session"
    session_dir.mkdir()

    config.restore_path = str(session_dir)

    session_manager = SessionManager(config)

    assert session_manager.session_path == session_dir


def test_session_manager_git_hash_comparison(config: ZeroConfig, temp_dir: Path) -> None:
    """Test git hash comparison when restoring from a session with different git hash."""
    # Create a session directory with a fake git hash
    session_dir = temp_dir / "existing_session"
    session_dir.mkdir()

    # Create a fake git hash file
    git_hash_path = session_dir / "git_hash.txt"
    fake_git_hash = "abcd1234567890fake_hash"
    with open(git_hash_path, "w") as f:
        f.write(fake_git_hash)

    config.restore_path = str(session_dir)

    # This should work without raising an exception
    # The git hash comparison will log warnings if hashes differ, but won't fail
    session_manager = SessionManager(config)

    assert session_manager.session_path == session_dir

    # Verify the original fake git hash is still there
    with open(git_hash_path) as f:
        stored_hash = f.read().strip()
        assert stored_hash == fake_git_hash


def test_session_manager_restore_path_not_exists(config: ZeroConfig) -> None:
    """Test SessionManager raises error when restore path doesn't exist."""
    config.restore_path = "/nonexistent/path"

    with pytest.raises(FileNotFoundError):
        SessionManager(config)


def test_manage_dataset_hash_nonexistent_path(config: ZeroConfig, temp_dir: Path) -> None:
    """Test manage_dataset_hash when dataset path doesn't exist."""
    config.output_path = str(temp_dir)
    session_manager = SessionManager(config)

    nonexistent_path = temp_dir / "nonexistent_dataset"

    # Should not raise an exception, just log warning and return
    session_manager.manage_dataset_hash(nonexistent_path)

    # Config should remain unchanged
    assert session_manager.config.dataset_hash is None


def test_manage_dataset_hash_new_run(config: ZeroConfig, temp_dir: Path) -> None:
    """Test manage_dataset_hash for a new run (no restore_path)."""
    config.output_path = str(temp_dir)
    config.restore_path = None  # Ensure it's a new run

    # Create a fake dataset directory with some files
    dataset_dir = temp_dir / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "file1.txt").write_text("test data 1")
    (dataset_dir / "file2.txt").write_text("test data 2")

    session_manager = SessionManager(config)

    # Before calling manage_dataset_hash, dataset_hash should be None
    assert session_manager.config.dataset_hash is None

    session_manager.manage_dataset_hash(dataset_dir)

    # After calling, dataset_hash should be set
    assert session_manager.config.dataset_hash is not None
    assert len(session_manager.config.dataset_hash) > 0

    # Config should have been saved with the hash
    config_path = session_manager.session_path / "config.yml"
    assert config_path.exists()
    with open(config_path) as f:
        saved_config = yaml.safe_load(f)
        assert saved_config["dataset_hash"] == session_manager.config.dataset_hash


def test_manage_dataset_hash_restored_run_no_hash_in_config(
    config: ZeroConfig, temp_dir: Path
) -> None:
    """Test manage_dataset_hash for restored run with no dataset_hash in config."""
    # Create a session directory to restore from
    session_dir = temp_dir / "existing_session"
    session_dir.mkdir()

    # Create a fake dataset directory
    dataset_dir = temp_dir / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "file1.txt").write_text("test data")

    config.restore_path = str(session_dir)
    config.dataset_hash = None  # No hash in the restored config

    session_manager = SessionManager(config)

    # Should not raise an exception, just log warning
    session_manager.manage_dataset_hash(dataset_dir)

    # Config dataset_hash should still be None
    assert session_manager.config.dataset_hash is None


def test_manage_dataset_hash_restored_run_matching_hash(config: ZeroConfig, temp_dir: Path) -> None:
    """Test manage_dataset_hash for restored run with matching hash."""
    # Create a session directory to restore from
    session_dir = temp_dir / "existing_session"
    session_dir.mkdir()

    # Create a fake dataset directory
    dataset_dir = temp_dir / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "file1.txt").write_text("test data")

    # Calculate the expected hash
    from linalgzero.utils.helpers import xxhash_dir

    expected_hash = xxhash_dir(dataset_dir)

    config.restore_path = str(session_dir)
    config.dataset_hash = expected_hash  # Set matching hash

    session_manager = SessionManager(config)

    # Should not raise an exception, just log success
    session_manager.manage_dataset_hash(dataset_dir)

    # Hash should remain the same
    assert session_manager.config.dataset_hash == expected_hash


def test_manage_dataset_hash_restored_run_mismatched_hash(
    config: ZeroConfig, temp_dir: Path
) -> None:
    """Test manage_dataset_hash for restored run with mismatched hash."""
    # Create a session directory to restore from
    session_dir = temp_dir / "existing_session"
    session_dir.mkdir()

    # Create a fake dataset directory
    dataset_dir = temp_dir / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "file1.txt").write_text("test data")

    config.restore_path = str(session_dir)
    config.dataset_hash = "fake_hash_that_wont_match"  # Set mismatched hash

    session_manager = SessionManager(config)

    # Should not raise an exception, just log warning
    session_manager.manage_dataset_hash(dataset_dir)

    # Hash should remain the original (mismatched) value
    assert session_manager.config.dataset_hash == "fake_hash_that_wont_match"
