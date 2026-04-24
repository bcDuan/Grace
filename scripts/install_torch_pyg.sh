#!/usr/bin/env bash
# Install PyTorch + torch-geometric (CUDA 12.1 wheels). For other CUDA, edit *_URL.
set -euo pipefail
PYTORCH_INDEX="${PYTORCH_INDEX:-https://download.pytorch.org/whl/cu121}"
PYG_URL="${PYG_URL:-https://data.pyg.org/whl/torch-2.1.0+cu121.html}"
TORCH_VER="2.1.0"

echo "Installing torch ${TORCH_VER} from ${PYTORCH_INDEX}"
pip install "torch==${TORCH_VER}" torchvision torchaudio --index-url "${PYTORCH_INDEX}"

echo "Installing torch_geometric, torch_scatter, torch_sparse from ${PYG_URL}"
pip install torch_geometric torch_scatter torch_sparse -f "${PYG_URL}"

# torch_geometric may pull NumPy 2.x; torch 2.1+cu121 expects NumPy 1.x ABI.
echo "Pinning numpy<2 for torch 2.1 compatibility"
pip install "numpy>=1.26,<2"

echo "OK: run python -c 'import torch; import torch_geometric; import numpy as np; print(np.__version__)'"
