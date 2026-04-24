#!/usr/bin/env bash
# From grace/: verify data, smoke tests, optional eval (requires raw JSON).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
echo "== verify data =="
python scripts/verify_data.py || true
if [[ -f data/raw/longmemeval/longmemeval_s.json ]]; then
  echo "== eval retrievers (first 20) =="
  python scripts/eval_retrievers.py --limit 20 || true
else
  echo "Skip eval: no LongMemEval JSON in data/raw/"
fi
echo "== done =="
