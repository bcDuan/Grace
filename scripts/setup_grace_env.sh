#!/usr/bin/env bash
# Step-by-step env setup with visible progress (avoids long silent "conda env create").
# Prereq: conda initialized (e.g. source ~/miniconda3/etc/profile.d/conda.sh)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export CONDA_VERBOSITY="${CONDA_VERBOSITY:-1}"
export PIP_PROGRESS_BAR="${PIP_PROGRESS_BAR:-on}"

if conda env list 2>/dev/null | awk '{print $1}' | grep -qx grace; then
  echo "[info] env 'grace' already exists."
  echo "  To refresh: conda activate grace && pip install -U -r requirements.txt && bash scripts/install_torch_pyg.sh && pip install -e ."
  exit 0
fi

echo "[1/4] conda create -n grace python=3.10 (conda-forge only, fast) ..."
conda create -n grace python=3.10 -y -c conda-forge

echo "[2/4] pip install -r requirements.txt (verbose progress) ..."
conda run -n grace pip install -v -r "${ROOT}/requirements.txt"

echo "[3/4] pip install -e . (editable package) ..."
conda run -n grace pip install -v -e "${ROOT}"

echo "[4/4] PyTorch + PyG (see scripts/install_torch_pyg.sh; set PYTORCH_INDEX/PYG_URL if needed) ..."
conda run -n grace bash "${ROOT}/scripts/install_torch_pyg.sh"

echo "[done] conda activate grace"
echo "Verify: python -c \"import torch, torch_geometric, rank_bm25; print('ok')\""
