# LinAlgZero

[![Release](https://img.shields.io/github/v/release/atomwalk12/LinAlgZero)](https://img.shields.io/github/v/release/atomwalk12/LinAlgZero)
[![Build status](https://img.shields.io/github/actions/workflow/status/atomwalk12/LinAlgZero/main.yml?branch=main)](https://github.com/atomwalk12/LinAlgZero/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/atomwalk12/LinAlgZero/branch/main/graph/badge.svg)](https://codecov.io/gh/atomwalk12/LinAlgZero)
[![Commit activity](https://img.shields.io/github/commit-activity/m/atomwalk12/LinAlgZero)](https://img.shields.io/github/commit-activity/m/atomwalk12/LinAlgZero)
[![License](https://img.shields.io/github/license/atomwalk12/LinAlgZero)](https://img.shields.io/github/license/atomwalk12/LinAlgZero)

This repository offers tools for generating a linear algebra problem dataset and training an open-source base model, aiming to explore its potential for emergent reasoning as inspired by the Deepseek-R1 paper.

- **Github repository**: <https://github.com/atomwalk12/LinAlgZero/>
- **Documentation** <https://atomwalk12.github.io/LinAlgZero/>

## Installation

### PyTorch Configuration

This project is configured with PyTorch defaults that work for most users:
- **Linux**: CUDA 12.8 builds (for GPU acceleration)
- **macOS/Windows**: CPU builds

#### For Different CUDA Versions

If you need a different CUDA version, you have two options:

**Option 1: Use uv's automatic backend selection (Recommended)**
```bash
# Automatically detect your CUDA version
UV_TORCH_BACKEND=auto uv sync

# Or specify a specific CUDA version
UV_TORCH_BACKEND=cu121 uv sync  # for CUDA 12.1
UV_TORCH_BACKEND=cu124 uv sync  # for CUDA 12.4
UV_TORCH_BACKEND=cpu uv sync    # for CPU-only
```

**Option 2: Override with pip-style installation**
```bash
# For different CUDA versions
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU-only
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

For the available CUDA versions see the [official documentation](https://pytorch.org/get-started/locally/).
