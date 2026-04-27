"""Query-conditioned GNN retriever inference."""

from __future__ import annotations

from typing import Any, List, Tuple

import torch
from sentence_transformers import SentenceTransformer

from grace.models.query_gnn import QueryGNN


class GNNRetriever:
    def __init__(
        self,
        model_path: str,
        sbert_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
        num_layers: int = 2,
        hidden_dim: int = 256,
        arch: str = "sage",
        gat_heads: int = 4,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sbert = SentenceTransformer(sbert_name, device=self.device)
        self.model = QueryGNN(
            in_dim=384,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            query_dim=384,
            arch=arch,
            gat_heads=gat_heads,
        ).to(self.device)
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

    @torch.no_grad()
    def retrieve(
        self, query: str, graph_data: dict[str, Any], k: int = 5
    ) -> List[Tuple[int, float, str]]:
        q = self.sbert.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]
        q_t = torch.tensor(q, device=self.device, dtype=torch.float32)
        x = graph_data["x"]
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
        else:
            x = x.to(self.device)
        ei = graph_data["edge_index"]
        if not isinstance(ei, torch.Tensor):
            ei = torch.tensor(ei, device=self.device, dtype=torch.long)
        else:
            ei = ei.to(self.device)
        logits = self.model(x, ei, q_t)
        kk = min(k, int(logits.size(0)))
        topk = torch.topk(logits, k=kk)
        texts: list[str] = graph_data["node_texts"]
        return [
            (int(i), float(s), texts[int(i)])
            for i, s in zip(topk.indices.tolist(), topk.values.tolist())
        ]
