"""Build memory graphs: sentence graph (SBERT top-k + session chain) and optional entity placeholder."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import networkx as nx
import numpy as np
import torch

from grace.schema import Turn


@dataclass
class MemoryGraph:
    """Unified graph for retrieval + GNN + PPR."""

    node_texts: list[str]
    # COO edge list for PyG [2, E] int64
    edge_index: np.ndarray
    # Optional sentence embeddings (N, d) for GNN node features
    x: np.ndarray | None = None
    # Undirected networkx for Personalized PageRank
    nx_graph: nx.Graph = field(default_factory=nx.Graph)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_torch(
        self, device: str | torch.device = "cpu"
    ) -> dict[str, Any]:
        ei = torch.from_numpy(self.edge_index).long()
        if self.x is not None:
            xt = torch.from_numpy(self.x).float()
        else:
            xt = None
        return {
            "edge_index": ei.to(device),
            "x": xt.to(device) if xt is not None else None,
            "node_texts": self.node_texts,
        }


def _dedup_edges(
    i: int, j: int, seen: set[tuple[int, int]], edges: list[tuple[int, int]]
) -> None:
    if i == j:
        return
    a, b = (i, j) if i < j else (j, i)
    if (a, b) in seen:
        return
    seen.add((a, b))
    edges.append((i, j))
    edges.append((j, i))


def build_sentence_graph(
    turns: Sequence[Turn | str],
    *,
    topk: int = 5,
    sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    embed_dim: int | None = None,
) -> MemoryGraph:
    """Sentence nodes: SBERT sim top-k (undirected) + same-session chain edges."""
    if not turns:
        return MemoryGraph(node_texts=[], edge_index=np.zeros((2, 0), dtype=np.int64), x=None)
    texts: list[str] = []
    sess: list[int] = []
    if isinstance(turns[0], Turn):
        for t in turns:  # type: ignore[assignment]
            assert isinstance(t, Turn)
            texts.append(t.text)
            sess.append(t.session_index)
    else:
        texts = [str(t) for t in turns]  # type: ignore[assignment]
        sess = [0] * len(texts)

    st = _encode_sbert(texts, sbert_model)
    n = len(texts)
    sim = st @ st.T
    np.fill_diagonal(sim, -1.0)
    k = min(topk, max(0, n - 1))
    edges: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for i in range(n):
        if k <= 0:
            break
        top = np.argsort(-sim[i])[:k]
        for j in top:
            if j < 0 or j == i:
                continue
            _dedup_edges(int(i), int(j), seen, edges)
    for i in range(n - 1):
        if sess[i] == sess[i + 1]:
            _dedup_edges(i, i + 1, seen, edges)
    if not edges and n > 1:
        for i in range(n - 1):
            _dedup_edges(i, i + 1, seen, edges)
    if not edges and n == 1:
        edges = [(0, 0)]
    if not edges:
        eix = np.zeros((2, 0), dtype=np.int64)
    else:
        arr = np.array(edges, dtype=np.int64).T
        eix = arr
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from((int(a), int(b)) for a, b in zip(eix[0], eix[1]))
    return MemoryGraph(
        node_texts=texts,
        edge_index=eix,
        x=np.asarray(st, dtype=np.float32),
        nx_graph=g,
        meta={"kind": "sentence", "sbert": sbert_model, "topk": topk},
    )


def _encode_sbert(texts: list[str], model_name: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    m = SentenceTransformer(model_name)
    emb = m.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(emb, dtype=np.float32)


def build_graph_from_corpus(
    texts: list[str],
    *,
    kind: str = "sentence",
    topk: int = 5,
    sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> MemoryGraph:
    if kind == "sentence":
        turns = [Turn(t, 0, i, i) for i, t in enumerate(texts)]
        return build_sentence_graph(turns, topk=topk, sbert_model=sbert_model)
    raise ValueError(f"Unknown graph kind: {kind}")


def build_entity_kg_stub(
    texts: list[str],
    *,
    sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> MemoryGraph:
    """Fallback: no triples available — use one node per text (same as flat nodes) with chain edges."""
    tlist = [Turn(t, 0, i, i) for i, t in enumerate(texts)]
    return build_sentence_graph(tlist, topk=0, sbert_model=sbert_model)
