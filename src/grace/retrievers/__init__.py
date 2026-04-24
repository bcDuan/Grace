from importlib import import_module
from typing import Any

from grace.retrievers.bm25 import BM25Retriever
from grace.retrievers.ppr import PPRRetriever
from grace.retrievers.sbert import SBERTRetriever

__all__ = [
    "BM25Retriever",
    "SBERTRetriever",
    "PPRRetriever",
    "GNNRetriever",
]


def __getattr__(name: str) -> Any:
    if name == "GNNRetriever":
        gnn = import_module("grace.retrievers.gnn")
        return getattr(gnn, "GNNRetriever")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(__all__))
