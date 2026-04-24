"""Print LongMemEval / LoCoMo file stats (paths from configs/default or CLI)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from grace.datasets.longmemeval import load_longmemeval_s  # noqa: E402
from grace.datasets.locomo import load_locomo10  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--longmemeval",
        default=str(_ROOT / "data/raw/longmemeval/longmemeval_s.json"),
    )
    p.add_argument(
        "--locomo",
        default=str(_ROOT / "data/raw/locomo10.json"),
    )
    args = p.parse_args()
    lp = Path(args.longmemeval)
    if lp.is_file():
        samples = load_longmemeval_s(lp)
        types = Counter(s.question_type for s in samples)
        print("LongMemEval:", lp)
        print("  n_samples:", len(samples))
        print("  question_types:", dict(types))
        if samples:
            print("  first_q:", samples[0].question[:100])
    else:
        print("LongMemEval not found:", lp)
    lop = Path(args.locomo)
    if lop.is_file():
        with lop.open() as f:
            j = json.load(f)
        n = len(j) if isinstance(j, list) else 1
        print("LoCoMo:", lop, "items~", n)
    else:
        print("LoCoMo not found:", lop)


if __name__ == "__main__":
    main()
