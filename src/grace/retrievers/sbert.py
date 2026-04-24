from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer as STModel


class SBERTRetriever:
    def __init__(
        self,
        corpus: list[str],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
        model: "STModel | None" = None,
    ):
        from sentence_transformers import SentenceTransformer

        self.corpus = corpus
        self.model = model if model is not None else SentenceTransformer(model_name, device=device)
        self._emb = self.model.encode(
            corpus,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(corpus) > 64,
        )
        self._emb = np.asarray(self._emb, dtype=np.float32)

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[int, float, str]]:
        q = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        scores = self._emb @ q.astype(np.float32)
        top_idx = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i]), self.corpus[i]) for i in top_idx]
