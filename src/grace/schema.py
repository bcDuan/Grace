"""Small shared types (live at package root so graphs.build does not import grace.datasets)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Turn:
    text: str
    session_index: int
    turn_index: int
    global_id: int
