"""Load LongMemEval JSON (e.g. longmemeval_s.json) and index sessions/turns for evidence mapping."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from grace.schema import Turn


@dataclass
class LongMemSample:
    raw: dict[str, Any]
    question: str
    question_id: str | int | None
    question_type: str
    answer: str | None
    turns: list[Turn] = field(default_factory=list)
    # Global node ids in sentence graph that are "positive" (evidence), if available
    evidence_global_ids: list[int] = field(default_factory=list)
    # Mapping from flattened turn global_id -> source haystack session id
    turn_to_session: dict[int, str] = field(default_factory=dict)
    # Gold answer session ids (if available)
    gold_sessions: set[str] = field(default_factory=set)


def _iter_session_messages(session: Any) -> list[dict[str, Any]]:
    if isinstance(session, list):
        return [x for x in session if isinstance(x, dict)]
    if isinstance(session, dict):
        for key in ("turns", "messages", "dialogue"):
            if key in session and isinstance(session[key], list):
                return [x for x in session[key] if isinstance(x, dict)]
    return []


def _turn_text(turn: dict[str, Any]) -> str:
    for key in ("content", "text", "value", "message", "body"):
        if key in turn and turn[key] is not None:
            s = str(turn[key]).strip()
            if s:
                return s
    if "role" in turn:
        return f"{turn.get('role', '')}: " + str(turn.get("content", ""))
    return json.dumps(turn, ensure_ascii=False)[:2000]


def _flatten_haystack_clean(sessions_src: Sequence[Any]) -> list[Turn]:
    out: list[Turn] = []
    gid = 0
    for si, session in enumerate(sessions_src):
        if isinstance(session, (list, tuple)):
            for ti, item in enumerate(session):
                if not isinstance(item, dict):
                    continue
                t = _turn_text(item)
                if not t.strip():
                    continue
                out.append(Turn(text=t, session_index=si, turn_index=ti, global_id=gid))
                gid += 1
        else:
            msgs = _iter_session_messages(session) if not isinstance(session, str) else []
            for ti, item in enumerate(msgs):
                t = _turn_text(item)
                if not t.strip():
                    continue
                out.append(Turn(text=t, session_index=si, turn_index=ti, global_id=gid))
                gid += 1
    return out


def _map_evidence_to_gids(
    sample: dict[str, Any], turns: list[Turn]
) -> list[int]:
    """Map evidence field(s) to global sentence ids. Best-effort for common formats."""
    if not turns:
        return []
    # HF xiaowu0162/longmemeval-cleaned: string session ids + haystack_session_ids index
    answer_ids = sample.get("answer_session_ids")
    haystack_ids = sample.get("haystack_session_ids")
    if (
        answer_ids
        and haystack_ids
        and isinstance(answer_ids, list)
        and isinstance(haystack_ids, list)
    ):
        want = {str(x) for x in answer_ids if x is not None}
        gids_clean: set[int] = set()
        for si, hsid in enumerate(haystack_ids):
            if str(hsid) in want:
                for t in turns:
                    if t.session_index == si:
                        gids_clean.add(t.global_id)
        if gids_clean:
            return sorted(gids_clean)
    for ev_key in (
        "evidence_sessions",
        "evidence_session_ids",
        "relevant_session_ids",
        "evidence_idx",
        "evidence",
    ):
        if ev_key in sample and sample[ev_key] is not None:
            ev = sample[ev_key]
            gids: set[int] = set()
            if isinstance(ev, (list, tuple, set)):
                for sid in ev:
                    try:
                        sidx = int(sid)
                    except (TypeError, ValueError):
                        continue
                    for t in turns:
                        if t.session_index == sidx:
                            gids.add(t.global_id)
            if gids:
                return sorted(gids)
    # String session ids
    for ev_key in ("evidence_sessions",):
        if ev_key in sample and isinstance(sample[ev_key], list) and len(sample[ev_key]) > 0:
            gids2: set[int] = set()
            for block in sample[ev_key]:
                if isinstance(block, (list, tuple)) and block:
                    # sometimes evidence is [session_idx, [turns]]
                    if isinstance(block[0], int):
                        sidx = int(block[0])
                        for t in turns:
                            if t.session_index == sidx:
                                gids2.add(t.global_id)
            if gids2:
                return sorted(gids2)
    return []


def parse_sample(sample: dict[str, Any]) -> LongMemSample:
    q = str(sample.get("question", "")).strip()
    qid = sample.get("question_id", sample.get("id", sample.get("qid")))
    qtype = str(sample.get("question_type", "unknown"))
    answer = None
    if "answer" in sample:
        answer = str(sample["answer"])
    elif "answers" in sample and sample["answers"]:
        answer = str(sample["answers"][0]) if isinstance(sample["answers"], list) else str(sample["answers"])
    turns: list[Turn] = []
    for k in ("haystack_sessions", "sessions", "context_sessions"):
        if k not in sample or sample[k] is None:
            continue
        sessions = sample[k] if isinstance(sample[k], (list, tuple)) else []
        if not sessions:
            continue
        ts = _flatten_haystack_clean(sessions)  # type: ignore[arg-type]
        if ts:
            turns = ts
            break
    ev = _map_evidence_to_gids(sample, turns)
    haystack_session_ids = sample.get("haystack_session_ids")
    answer_session_ids = sample.get("answer_session_ids")
    turn_to_session: dict[int, str] = {}
    for t in turns:
        sid = None
        if (
            isinstance(haystack_session_ids, list)
            and 0 <= t.session_index < len(haystack_session_ids)
            and haystack_session_ids[t.session_index] is not None
        ):
            sid = str(haystack_session_ids[t.session_index])
        else:
            sid = str(t.session_index)
        turn_to_session[t.global_id] = sid
    gold_sessions: set[str] = set()
    if isinstance(answer_session_ids, list):
        gold_sessions = {str(x) for x in answer_session_ids if x is not None}
    if not gold_sessions and ev:
        gold_sessions = {turn_to_session[i] for i in ev if i in turn_to_session}
    return LongMemSample(
        raw=sample,
        question=q,
        question_id=qid,
        question_type=qtype,
        answer=answer,
        turns=turns,
        evidence_global_ids=ev,
        turn_to_session=turn_to_session,
        gold_sessions=gold_sessions,
    )


def load_longmemeval_s(path: str | Path) -> list[LongMemSample]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {p}, got {type(data)}")
    return [parse_sample(x) for x in data if isinstance(x, dict)]


def split_indices(n: int, train_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n_train = int(n * train_ratio)
    return idx[:n_train], idx[n_train:]
