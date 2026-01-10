"""TUI adapters for graceful degradation and compatibility."""

from gpt_trader.tui.adapters.null_status_reporter import NullStatusReporter
from gpt_trader.tui.adapters.runtime_ui_adapter import TuiRuntimeUIAdapter

__all__ = ["NullStatusReporter", "TuiRuntimeUIAdapter"]
