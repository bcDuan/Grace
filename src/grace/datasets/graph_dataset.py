"""PyTorch Dataset: one query + memory graph + answer-aware evidence labels."""

from __future__ import annotations

from dataclasses import dataclass
import re
import string
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from sentence_transformers import SentenceTransformer

from grace.datasets.longmemeval import LongMemSample
from grace.graphs.build import MemoryGraph, build_sentence_graph


def graph_sample_to_tensors(
    mg: MemoryGraph, query_emb: np.ndarray, device: str = "cpu"
) -> dict[str, Any]:
    d = mg.to_torch(device=device)
    out: dict[str, Any] = {
        "x": d["x"],
        "edge_index": d["edge_index"],
        "q": torch.tensor(query_emb, device=device, dtype=torch.float32),
        "node_texts": d["node_texts"],
    }
    return out


@dataclass
class GraphRow:
    question: str
    question_type: str
    graph: MemoryGraph
    pos_mask: np.ndarray  # bool (N,)
    label_weights: np.ndarray | None = None  # float (N,), answer-aware BCE weights
    strong_pos_mask: np.ndarray | None = None  # bool (N,), answer-bearing turns


class GraphMatchDataset(Dataset):
    """
    Precomputed rows: (SBERT question embedding, graph, positive mask, meta).
    """

    def __init__(self, rows: list[GraphRow], sbert: SentenceTransformer | str | None = None):
        from sentence_transformers import SentenceTransformer as ST

        self._rows = rows
        if isinstance(sbert, str) or sbert is None:
            self._sbert: SentenceTransformer | None = ST(
                sbert or "sentence-transformers/all-MiniLM-L6-v2"
            )
        else:
            self._sbert = sbert
        self._q_cache: list[np.ndarray | None] = [None] * len(rows)

    def __len__(self) -> int:
        return len(self._rows)

    def _query_emb(self, i: int) -> np.ndarray:
        if self._q_cache[i] is not None:
            return self._q_cache[i]  # type: ignore[return-value]
        assert self._sbert is not None
        qv = self._sbert.encode(
            [self._rows[i].question],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        self._q_cache[i] = np.asarray(qv, dtype=np.float32)
        return self._q_cache[i]  # type: ignore[return-value]

    def __getitem__(self, idx: int) -> tuple:
        r = self._rows[idx]
        g = r.graph
        d = g.to_torch(device="cpu")
        x, ei = d["x"], d["edge_index"]
        if x is None:
            raise ValueError("Graph has no x; use build_sentence_graph (SBERT) first")
        pos = torch.tensor(r.pos_mask, dtype=torch.bool)
        if r.label_weights is None:
            label_weights = pos.float()
        else:
            label_weights = torch.tensor(r.label_weights, dtype=torch.float32)
        if r.strong_pos_mask is None:
            strong_pos = torch.zeros_like(pos)
        else:
            strong_pos = torch.tensor(r.strong_pos_mask, dtype=torch.bool)
        qe = self._query_emb(idx)
        q = torch.tensor(qe, dtype=torch.float32)
        return x, ei, q, pos, label_weights, strong_pos, g.node_texts


_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def _normalize_for_match(text: str) -> str:
    text = text.lower()
    text = text.translate(_PUNCT_TABLE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _answer_in_text(answer: str | None, text: str, *, max_answer_chars: int = 120) -> bool:
    if not answer:
        return False
    ans = _normalize_for_match(str(answer))
    if not ans or len(ans) > max_answer_chars:
        return False
    return ans in _normalize_for_match(text)


def longmem_samples_to_graph_rows(
    samples: list[LongMemSample],
    *,
    topk: int = 5,
    sbert: str = "sentence-transformers/all-MiniLM-L6-v2",
    weak_positive_weight: float = 0.2,
) -> list[GraphRow]:
    out: list[GraphRow] = []
    for s in samples:
        if not s.turns:
            continue
        g = build_sentence_graph(s.turns, topk=topk, sbert_model=sbert)
        n = len(g.node_texts)
        mask = np.zeros(n, dtype=bool)
        for gid in s.evidence_global_ids:
            if 0 <= gid < n:
                mask[gid] = True
        if not mask.any() and s.evidence_global_ids:
            continue
        if not mask.any():
            # skip or keep as no supervision — skip for training
            continue
        strong = np.zeros(n, dtype=bool)
        for gid in s.evidence_global_ids:
            if 0 <= gid < n and _answer_in_text(s.answer, g.node_texts[gid]):
                strong[gid] = True
        weights = np.zeros(n, dtype=np.float32)
        if strong.any():
            weights[mask] = float(weak_positive_weight)
            weights[strong] = 1.0
        else:
            weights[mask] = 1.0
        out.append(
            GraphRow(
                question=s.question,
                question_type=s.question_type,
                graph=g,
                pos_mask=mask,
                label_weights=weights,
                strong_pos_mask=strong,
            )
        )
    return out


def collate_graph_batch(batch: list):
    return batch
