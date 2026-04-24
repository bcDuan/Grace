from __future__ import annotations

from typing import List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi


class BM25Retriever:
    def __init__(self, corpus: List[str]):
        tokenized = [doc.lower().split() for doc in corpus]
        self.bm25 = BM25Okapi(tokenized)
        self.corpus = corpus

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[int, float, str]]:
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[-k:][::-1]
        return [(int(i), float(scores[i]), self.corpus[i]) for i in top_idx]
