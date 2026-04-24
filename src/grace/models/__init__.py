from importlib import import_module
from typing import Any

__all__ = ["QueryGNN", "combined_loss"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        mod = import_module("grace.models.query_gnn")
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(__all__))
