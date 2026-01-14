"""
Reusable UI primitives for the High-Fidelity TUI.
"""

from __future__ import annotations

from textual.reactive import reactive
from textual.widgets import Static

from gpt_trader.tui.theme import THEME


class SparklineWidget(Static):
    """
    Renders a crypto-style sparkline graph using Unicode block characters.
    """

    data = reactive([])  # type: ignore[var-annotated]

    def __init__(
        self,
        data: list[float] | None = None,
        color_trend: bool = True,
        id: str | None = None,
        classes: str | None = None,
    ):
        super().__init__(id=id, classes=classes)
        self.data: list[float] = data or []
        self._color_trend = color_trend

    def watch_data(self, data: list[float]) -> None:
        self.update(self._generate_sparkline(data))

    def _generate_sparkline(self, data: list[float]) -> str:
        if not data or len(data) < 2:
            return "─" * 10  # Fallback

        # Unicode block characters (0/8 to 8/8 height)
        blocks = [" ", "▂", "▃", "▄", "▅", "▆", "▇", "█"]

        min_val = min(data)
        max_val = max(data)
        rng = max_val - min_val

        if rng == 0:
            return "▄" * len(data)

        chars = []
        for val in data:
            normalized = (val - min_val) / rng
            # Map 0..1 to 0..7
            idx = int(normalized * (len(blocks) - 1))
            chars.append(blocks[idx])

        line = "".join(chars)

        if self._color_trend and len(data) >= 2:
            start, end = data[0], data[-1]
            color = THEME.colors.success if end >= start else THEME.colors.error
            return f"[{color}]{line}[/]"

        return line


class ProgressBarWidget(Static):
    """
    Renders a text-based progress bar with gradient coloring.
    Valid percentage: 0.0 to 1.0 (or 0 to 100)
    """

    percentage = reactive(0.0)

    def __init__(
        self,
        percentage: float = 0.0,
        label: str = "",
        id: str | None = None,
        classes: str | None = None,
    ):
        super().__init__(id=id, classes=classes)
        self.label = label
        self.percentage = percentage

    def watch_percentage(self, value: float) -> None:
        self.update(self._render_bar(value))

    def _render_bar(self, value: float) -> str:
        # Normalize to 0-1
        if value > 1.0:
            value /= 100.0
        value = max(0.0, min(1.0, value))

        width = self.content_size.width or 20
        # Fixed label width for alignment (pad to 4 chars)
        label_padded = f"{self.label:<4}" if self.label else ""
        # Subtract label length and percentage display
        bar_width = width - len(label_padded) - 6  # 6 chars for " 100%"

        if bar_width < 5:
            bar_width = 10

        filled_len = int(bar_width * value)
        empty_len = bar_width - filled_len

        # Determine color based on thresholds
        color = THEME.colors.success
        if value > 0.6:
            color = THEME.colors.warning
        if value > 0.85:
            color = THEME.colors.error

        bar_str = f"[{color}]{'█' * filled_len}[/][{THEME.colors.surface}]{'░' * empty_len}[/]"
        pct_str = f"{int(value * 100):>3}%"

        return f"{label_padded}{bar_str} {pct_str}"


class StatusBadgeWidget(Static):
    """
    Renders a pill-shaped status badge.
    """

    status = reactive("UNKNOWN")

    def __init__(self, status: str = "UNKNOWN", id: str | None = None, classes: str | None = None):
        super().__init__(id=id, classes=classes)
        self.status = status

    def watch_status(self, status: str) -> None:
        self._update_badge(status)

    def _update_badge(self, status: str) -> None:
        self.classes = ""  # Reset
        icon = "•"

        s_upper = status.upper()
        if s_upper in ("LIVE", "RUNNING", "CONNECTED"):
            self.add_class("badge-live")
            icon = "●"
        elif s_upper in ("STOPPED", "DISCONNECTED", "ERROR"):
            self.add_class("badge-stopped")
            icon = "■"
        elif s_upper in ("SYNCING", "CONNECTING", "WARNING"):
            self.add_class("badge-syncing")
            icon = "○"

        self.update(f"{icon} {s_upper}")
