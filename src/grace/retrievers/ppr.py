"""Personalized PageRank on a memory graph (NetworkX)."""

from __future__ import annotations

from typing import List, Tuple

import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer

from grace.graphs.build import MemoryGraph


class PPRRetriever:
    def __init__(
        self,
        graph: MemoryGraph,
        sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        alpha: float = 0.15,
        device: str | None = None,
    ):
        self.graph = graph
        self.alpha = alpha
        self.model = SentenceTransformer(sbert_model, device=device)
        if not isinstance(graph.nx_graph, nx.Graph):
            raise TypeError("MemoryGraph.nx_graph must be networkx.Graph")
        self._G = graph.nx_graph

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[int, float, str]]:
        if not self.graph.node_texts:
            return []
        qv = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        node_embs = self.model.encode(
            self.graph.node_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(self.graph.node_texts) > 128,
        )
        sims = np.asarray(node_embs, dtype=np.float32) @ qv.astype(np.float32)
        sims = np.maximum(sims, 0.0)
        s = sims.sum()
        if s <= 0:
            p = {n: 1.0 / len(self._G) for n in self._G.nodes}
        else:
            p = {int(n): 0.0 for n in self._G.nodes}
            for n in self._G.nodes:
                p[int(n)] = float(sims[int(n)] / s)
        try:
            pr = nx.pagerank(self._G, alpha=self.alpha, personalization=p)
        except Exception:
            pr = nx.pagerank(self._G, alpha=self.alpha)
        nodes = sorted(pr.keys(), key=lambda n: pr[n], reverse=True)[:k]
        return [(int(n), float(pr[n]), self.graph.node_texts[int(n)]) for n in nodes]
