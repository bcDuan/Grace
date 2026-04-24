from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def group_metrics_by_type(
    types: list[str], values: list[T], reduce: Callable[[list[T]], T] | None = None
) -> dict[str, list[T]]:
    by: dict[str, list[T]] = defaultdict(list)
    for t, v in zip(types, values):
        by[str(t)].append(v)
    return dict(by)
