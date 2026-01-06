"""Signal contribution bar widget.

Visualizes indicator contributions as horizontal bars showing the strength
and direction of each signal's influence on a trading decision.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Label, ProgressBar, Static

from gpt_trader.tui.types import IndicatorContribution


class SignalBar(Static):
    """Displays a single signal's contribution as a horizontal bar.

    Layout:
    ```
    Name       ████████░░░░░░░  +0.42
    ```

    The bar fills from center:
    - Positive values fill right (green)
    - Negative values fill left (red)
    """

    DEFAULT_CSS = """
    SignalBar {
        height: 1;
        layout: horizontal;
    }

    SignalBar .signal-name {
        width: 12;
        color: $text-muted;
    }

    SignalBar .signal-bar-container {
        width: 15;
    }

    SignalBar .signal-bar {
        width: 100%;
        height: 1;
    }

    SignalBar .signal-bar Bar {
        width: 100%;
    }

    SignalBar .signal-bar.-bullish Bar > .bar--bar {
        color: $success;
        background: $success 30%;
    }

    SignalBar .signal-bar.-bearish Bar > .bar--bar {
        color: $error;
        background: $error 30%;
    }

    SignalBar .signal-value {
        width: 6;
        text-align: right;
    }

    SignalBar .signal-value.positive {
        color: $success;
    }

    SignalBar .signal-value.negative {
        color: $error;
    }
    """

    def __init__(
        self,
        contribution: IndicatorContribution,
        **kwargs,
    ) -> None:
        """Initialize SignalBar.

        Args:
            contribution: The indicator contribution to display.
            **kwargs: Additional widget arguments.
        """
        super().__init__(**kwargs)
        self._contribution = contribution

    def compose(self) -> ComposeResult:
        # Truncate name to fit
        name = self._contribution.name[:10].ljust(10)
        yield Label(name, classes="signal-name")

        bar_classes = "signal-bar"
        if self._contribution.contribution > 0:
            bar_classes += " -bullish"
        else:
            bar_classes += " -bearish"

        with Horizontal(classes="signal-bar-container"):
            yield ProgressBar(total=100, show_eta=False, show_percentage=False, classes=bar_classes)

        # Value label
        sign = "+" if self._contribution.contribution >= 0 else ""
        value_str = f"{sign}{self._contribution.contribution:.2f}"
        value_classes = "signal-value"
        if self._contribution.contribution > 0:
            value_classes += " positive"
        elif self._contribution.contribution < 0:
            value_classes += " negative"

        yield Label(value_str, classes=value_classes)

    def on_mount(self) -> None:
        """Set progress bar value after mount."""
        try:
            bar = self.query_one(ProgressBar)
            bar_value = abs(self._contribution.contribution) * 100
            bar.update(progress=bar_value)
        except Exception:
            pass


class SignalBreakdown(Static):
    """Displays multiple signal bars for a decision's contributions.

    Layout:
    ```
    Trend       ████████░░░░░░░  +0.42
    Momentum    ░░░░░░░████░░░░  -0.21
    MeanRev     ██████░░░░░░░░░  +0.35
    ```
    """

    DEFAULT_CSS = """
    SignalBreakdown {
        height: auto;
        padding: 0 1;
        background: $surface;
    }

    SignalBreakdown .breakdown-header {
        color: $text-muted;
        text-style: italic;
        margin-bottom: 0;
    }

    SignalBreakdown .breakdown-content {
        height: auto;
    }

    SignalBreakdown .empty-breakdown {
        color: $text-muted;
        text-style: italic;
    }
    """

    def __init__(
        self,
        contributions: list[IndicatorContribution],
        show_header: bool = True,
        **kwargs,
    ) -> None:
        """Initialize SignalBreakdown.

        Args:
            contributions: List of indicator contributions to display.
            show_header: Whether to show the "Signal Breakdown" header.
            **kwargs: Additional widget arguments.
        """
        super().__init__(**kwargs)
        self._contributions = contributions
        self._show_header = show_header

    def compose(self) -> ComposeResult:
        if self._show_header:
            yield Label("Signal Breakdown:", classes="breakdown-header")

        if not self._contributions:
            yield Label("No signal data", classes="empty-breakdown")
        else:
            # Sort by absolute contribution (strongest first)
            sorted_contributions = sorted(
                self._contributions,
                key=lambda c: abs(c.contribution),
                reverse=True,
            )
            for contribution in sorted_contributions:
                yield SignalBar(contribution)

    def update_contributions(self, contributions: list[IndicatorContribution]) -> None:
        """Update the displayed contributions.

        Args:
            contributions: New list of contributions to display.
        """
        self._contributions = contributions
        # Remove existing bars and re-compose
        for child in self.query(SignalBar):
            child.remove()
        for child in self.query(".empty-breakdown"):
            child.remove()

        if not contributions:
            self.mount(Label("No signal data", classes="empty-breakdown"))
        else:
            sorted_contributions = sorted(
                contributions,
                key=lambda c: abs(c.contribution),
                reverse=True,
            )
            for contribution in sorted_contributions:
                self.mount(SignalBar(contribution))
