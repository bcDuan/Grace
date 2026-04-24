"""Load LoCoMo-10 JSON (e.g. locomo10.json)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from grace.datasets.longmemeval import LongMemSample, parse_sample


@dataclass
class LoCoMoItem:
    raw: dict[str, Any]
    sample: LongMemSample
    story_id: str | None = None


def _to_longmemeval_like(doc: dict[str, Any]) -> dict[str, Any]:
    """Map common LoCoMo fields to the shape expected by parse_sample."""
    out: dict[str, Any] = dict(doc)
    if "haystack_sessions" not in out and "session" in out and isinstance(out["session"], (list, tuple)):
        out["haystack_sessions"] = out["session"]
    if "question" not in out and "query" in out:
        out["question"] = out["query"]
    if "question_type" not in out:
        out["question_type"] = str(out.get("category", "locomo"))
    return out


def load_locomo10(path: str | Path) -> list[LoCoMoItem]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    if not isinstance(data, list):
        data = [data]  # type: ignore[list-item]
    out: list[LoCoMoItem] = []
    for i, doc in enumerate(data):
        if not isinstance(doc, dict):
            continue
        shaped = _to_longmemeval_like(doc)
        sid = str(doc.get("session_id", doc.get("id", f"item_{i}")))
        s = parse_sample(shaped)
        out.append(LoCoMoItem(raw=doc, sample=s, story_id=sid))
    return out
