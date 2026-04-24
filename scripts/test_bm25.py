"""Smoke: BM25 on one LongMemEval-S sample (requires data/raw/.../longmemeval_s.json)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from grace.retrievers.bm25 import BM25Retriever  # noqa: E402


def main() -> None:
    path = _ROOT / "data" / "raw" / "longmemeval" / "longmemeval_s.json"
    if not path.is_file():
        print(f"Missing {path} — download dataset first (see README).", file=sys.stderr)
        sys.exit(1)
    with path.open() as f:
        data = json.load(f)
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    sample = data[0]
    corpus: list[str] = []
    for session in sample.get("haystack_sessions", sample.get("sessions", [])):
        for turn in session if isinstance(session, (list, tuple)) else []:
            if isinstance(turn, dict):
                t = str(turn.get("content", turn.get("text", "")))
            else:
                t = str(turn)
            if t.strip():
                corpus.append(t)
    if not corpus:
        print("Empty corpus in sample 0.", file=sys.stderr)
        sys.exit(1)
    retriever = BM25Retriever(corpus)
    q = str(sample.get("question", ""))
    results = retriever.retrieve(q, k=5)
    print("Question:", q[:200])
    print("Question type:", sample.get("question_type"))
    print("Top-5 BM25:")
    for i, s, t in results:
        print(f"  [{i}] {s:.3f} {t[:100]}...")


if __name__ == "__main__":
    main()
