# CHANGELOG


## v0.2.0 (2025-06-17)

### Bug Fixes

- Update pytorch dependencies
  ([`ddb64a4`](https://github.com/atomwalk12/LinAlgZero/commit/ddb64a4482aa9d65f6e141da1e45ab9129df0aa0))

- Added support for different CUDA versions in `pyproject.toml` with specific indices for CPU and
  CUDA builds. - Updated the `README.md` to include installation instructions for PyTorch
  configurations. - Added `wandb`, `torch`, `torchvision`, and `torchaudio` as dependencies in
  `pyproject.toml`. - Updated `uv.lock` to reflect new package versions and dependencies.

### Documentation

- Improve documentation with module references
  ([`994998f`](https://github.com/atomwalk12/LinAlgZero/commit/994998f5480b42d2de6da53bac28302c76c0b491))

- Updated mkdocs.yml to include additional options.

- Update mkdocs.yml to hide the root full path in the docs
  ([`47a7da3`](https://github.com/atomwalk12/LinAlgZero/commit/47a7da3b7fb3dc2c8c3a3e90c443c00250c043c1))

- Update README for CUDA installation instructions
  ([`13c1bad`](https://github.com/atomwalk12/LinAlgZero/commit/13c1badf50a735a63ef8847fa38b2d0bdd0d7deb))

- Removed outdated installation options.

### Features

- Add CIFAR10 dataset wrapper and mock metrics for evaluation
  ([`9534367`](https://github.com/atomwalk12/LinAlgZero/commit/9534367899d4aa756757dd614820748063f6d547))

- Introduced `CifarDataset` class to wrap the CIFAR10 dataset, returning data as dictionaries. -
  Added `Metric` abstract base class and `AccuracyMetric` implementation for metric calculations. -
  Implemented `SimpleCNN` model architecture. A mock architecture intended for testing. It will be
  replaced. - Updated trainer to utilize the new dataset and metrics, including data loading and
  loss computation. - Improved session management with error handling for uninitialized components.

- Add gh-deploy target to Makefile for GitHub Pages deployment
  ([`2d398b6`](https://github.com/atomwalk12/LinAlgZero/commit/2d398b625d2ba9c756ac5cff5319e96a87b5be7c))

- Add Trainer API and update the documentation page
  ([`65d7445`](https://github.com/atomwalk12/LinAlgZero/commit/65d74458f330a579f122a33f68df3c3046ea8789))

- Added a new Makefile target `coverage-html` to generate coverage reports in HTML format. - Updated
  copyright link in `mkdocs.yml` to point to the GitHub repository. - Modified `modules.md` to
  reflect changes in module structure. - Introduced `ZeroTrainer` class in `trainer.py` for managing
  training processes. - Added unit tests for `ZeroTrainer` in `test_trainer.py` to ensure
  functionality.

- Add WandbLogger for logging metrics to Weights & Biases
  ([`18c6134`](https://github.com/atomwalk12/LinAlgZero/commit/18c613418c2681eee80b87ab8b5d143eb5ebf42a))

- Introduced `WandbLogger` class - Implemented methods for initialization, logging metrics, and
  finishing runs. - Added unit tests in `test_wandb_logger.py` to verify functionality, including
  handling of tags and errors.

- Implement configuration file and command line parsers
  ([`0788feb`](https://github.com/atomwalk12/LinAlgZero/commit/0788feb215fcb46db16df4eae4085fe39b062aac))

- Added `run_training.py` for executing training with configurable parameters. - Introduced
  `ZeroConfig` dataclass to manage training configurations. - Created default configuration
  retrieval function and example YAML files.

- Implement session management and logging utilities
  ([`cce4d8d`](https://github.com/atomwalk12/LinAlgZero/commit/cce4d8d42ac05b2d1514bd35b28a389fed73410e))

- Added `SessionManager` class for managing training sessions, including logging, checkpointing, and
  configuration management. - Introduced `setup_logging` function to configure logging to both file
  and console. - Created unit tests for `SessionManager` to verify initialization, JSON saving,
  checkpointing, and restore path handling.

- Improve training script and trainer functionality
  ([`51f25cd`](https://github.com/atomwalk12/LinAlgZero/commit/51f25cd5bf0dd9cdaa0fc74d31b2ffb4a681b15b))

- Can load previous section from the startup script. - Expanded `ZeroTrainer` class with
  comprehensive training and validation loops, including logging and checkpointing. - Updated
  `LinAlgTrainer` to implement model creation, optimizer setup, and metrics calculation.

### Refactoring

- Add additional tests for the Trainer class
  ([`2f0ee6f`](https://github.com/atomwalk12/LinAlgZero/commit/2f0ee6fe822960a9835135b968c59b7780c41c7e))

- Refactored tests in `test_trainer.py` to use fixtures for mocking dependencies. - Updated type
  hints for test functions across various test files

- Add base abstract methods for the Trainer class
  ([`349df81`](https://github.com/atomwalk12/LinAlgZero/commit/349df81b7b89068d1245d25ea91925166691f591))

- Refactored `ZeroTrainer` class in `trainer.py` to implement abstract methods for model, optimizer,
  and dataloaders creation. - Added logging and device setup to the `ZeroTrainer` class.

- Reorganise the code structure into separate files for the accuracy metric and Neural Network
  creation
  ([`efafc77`](https://github.com/atomwalk12/LinAlgZero/commit/efafc77566585546e3ef6c2b127cfa2e438cba75))

- Update trainer class and test implementation
  ([`69b2c26`](https://github.com/atomwalk12/LinAlgZero/commit/69b2c269ef3418e3ddcd58958e7e7b2d177cdf35))

- Updated test to use `LinAlgTrainer` instead of `ZeroTrainer` in `test_trainer.py`.

### Testing

- Improve testing framework for trainer functionality
  ([`d32c464`](https://github.com/atomwalk12/LinAlgZero/commit/d32c464820bfa3746fa51a4175be4ecddf3d22ad))

- Introduced `MockMetric` and `MockTrainer` classes to facilitate unit testing. - Updated test cases
  to validate trainer initialization, setup, training, and validation steps.


## v0.1.0 (2025-05-20)

### Bug Fixes

- Formatting issues
  ([`370d103`](https://github.com/atomwalk12/LinAlgZero/commit/370d1034aa62f5ff2e051e9f8f7ff9a95764c4d2))

- Remove podman flag for UID/GUI mapping and use docker equivalent
  ([`6c06aa0`](https://github.com/atomwalk12/LinAlgZero/commit/6c06aa0d52a7d14353a6fdb4082f6afb76278f0d))

- Update project versioning and add semantic release configuration
  ([`e88c217`](https://github.com/atomwalk12/LinAlgZero/commit/e88c217c5f1f58b70f600cabe2f9650672fcbc28))

### Features

- Add GitHub Actions workflow for conventional pull requests
  ([`cf1b77d`](https://github.com/atomwalk12/LinAlgZero/commit/cf1b77d71db2c2f405d0cb2ea260804527d10586))

- Add semantic release target to Makefile
  ([`94015d4`](https://github.com/atomwalk12/LinAlgZero/commit/94015d401f4856625900cc529f902135b3ec599f))

- Improve the theme layout
  ([`b3d4092`](https://github.com/atomwalk12/LinAlgZero/commit/b3d409245ee14b696d76cfc299d7670aa2335d76))

- Initial commit
  ([`63b1418`](https://github.com/atomwalk12/LinAlgZero/commit/63b1418ed5c6afe2c368919866b9cd6c3f4524cc))
