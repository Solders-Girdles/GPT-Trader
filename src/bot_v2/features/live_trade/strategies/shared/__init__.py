"""Shared utilities for live trade strategies."""

from __future__ import annotations

from .decisions import create_close_decision, create_entry_decision
from .mark_window import update_mark_window
from .signals import MASnapshot, calculate_ma_snapshot
from .trailing_stop import clear_trailing_stop_state, update_trailing_stop

__all__ = [
    "update_mark_window",
    "MASnapshot",
    "calculate_ma_snapshot",
    "update_trailing_stop",
    "clear_trailing_stop_state",
    "create_entry_decision",
    "create_close_decision",
]
