from __future__ import annotations

from typing import Dict, Sequence, Set

import numpy as np


def recall_at_k(retrieved: Sequence[int] | np.ndarray, positives: Set[int], k: int) -> float:
    """Turn-level recall: fraction of positives found in top-k."""
    if not positives:
        return 0.0
    r = retrieved[:k] if isinstance(retrieved, list) else retrieved[:k]
    hit = len(positives.intersection(int(x) for x in r))
    return float(hit) / float(len(positives))


def hit_at_k(retrieved: Sequence[int] | np.ndarray, positives: Set[int], k: int) -> float:
    """Binary: 1.0 if any positive is in top-k, else 0.0."""
    if not positives:
        return 0.0
    r = retrieved[:k] if isinstance(retrieved, list) else retrieved[:k]
    return 1.0 if positives.intersection(int(x) for x in r) else 0.0


def session_recall_at_k(
    retrieved: Sequence[int] | np.ndarray,
    turn_to_session: Dict[int, str],
    gold_sessions: Set[str],
    k: int,
) -> float:
    """
    Session-level recall: of the gold sessions, what fraction is hit by top-k turns.
    """
    if not gold_sessions:
        return 0.0
    r = retrieved[:k] if isinstance(retrieved, list) else retrieved[:k]
    hit_sessions = {turn_to_session[int(i)] for i in r if int(i) in turn_to_session}
    return float(len(hit_sessions & gold_sessions)) / float(len(gold_sessions))


def mrr(
    ranked_ids: Sequence[int] | np.ndarray,
    positives: Set[int],
) -> float:
    if not positives:
        return 0.0
    r = ranked_ids if isinstance(ranked_ids, list) else list(ranked_ids)
    for rank, idx in enumerate(r, start=1):
        if int(idx) in positives:
            return 1.0 / rank
    return 0.0
