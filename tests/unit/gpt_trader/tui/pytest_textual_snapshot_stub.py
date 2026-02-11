"""Minimal stub for pytest_textual_snapshot to support offline testing."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class PseudoConsole:
    """Lightweight replacement for the pytest-textual-snapshot console."""

    def __init__(self, legacy_windows: bool, size: tuple[int, int]) -> None:
        self.legacy_windows = legacy_windows
        self.size = size


class PseudoApp:
    """Stub application object used only for reporting metadata."""

    def __init__(self, console: PseudoConsole) -> None:
        self.console = console


def node_to_report_path(node: Any) -> Path:
    """Return a stable path where snapshot report data can be written."""

    node_id = getattr(node, "nodeid", getattr(node, "name", "unidentified"))
    safe_id = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in node_id)
    report_dir = Path("var") / "textual_snapshot"
    report_dir.mkdir(parents=True, exist_ok=True)
    return report_dir / f"{safe_id}.data"
