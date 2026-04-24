#!/usr/bin/env bash
# Run from repository root: bash scripts/run_smoke.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
python -c "import torch; import torch_geometric; print('torch', torch.__version__); print('pyg', torch_geometric.__version__)"
echo "smoke: imports OK (run test_bm25.py if data is present)"
if [[ -f data/raw/longmemeval/longmemeval_s.json ]]; then
  python scripts/test_bm25.py
else
  echo "Note: data/raw/longmemeval/longmemeval_s.json not found; skip test_bm25."
fi
