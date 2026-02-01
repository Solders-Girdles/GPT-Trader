"""Core shell components for the High-Fidelity TUI.

Includes CommandBar (Header) and BentoGrid layouts.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Container
from textual.reactive import reactive
from textual.widgets import Label, Static

from gpt_trader.tui.services.onboarding_service import get_onboarding_service
from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from gpt_trader.tui.state import TuiState

logger = get_logger(__name__, component="tui")


class CommandBar(Static):
    """
    The main application header bar.
    Displays:
    1. Identity/Branding (Left)
    2. Connection & Bot Status (Center)
    3. System Info / Clock (Right)
    """

    time_str = reactive("00:00:00")

    # Styles moved to styles/widgets/shell.tcss

    def __init__(self, bot_mode: str = "DEMO", id: str | None = None, classes: str | None = None):
        super().__init__(id=id, classes=classes)
        self.bot_mode = bot_mode
        self._onboarding = get_onboarding_service()

    def on_mount(self) -> None:
        # Avoid time-driven updates during headless test runs to keep snapshots stable.
        if not getattr(self.app, "is_headless", False):
            self.set_interval(1.0, self.update_time)
        self.update_time()
        if hasattr(self.app, "state_registry"):
            self.app.state_registry.register(self)

    def update_time(self) -> None:
        self.time_str = datetime.now().strftime("%H:%M UTC")
        # Keep the connection badge fresh even when no new status snapshots arrive.
        # This avoids the "stale/connecting forever" feel when the bot is paused.
        try:
            state = getattr(self.app, "tui_state", None)
            if state is not None:
                self._update_connection_badge(state)
        except Exception:
            pass

    def compose(self) -> ComposeResult:
        # 1. Identity
        yield Label("GPT-TRADER", classes="identity")

        # 2. Central Status Area using containers for alignment
        with Container(classes="status-area"):
            yield Label(
                self.bot_mode, id="mode-badge", classes=f"mode-badge {self.bot_mode.lower()}"
            )
            yield Label("○ CONNECTING", id="connection-status", classes="status-text")
            yield Label("", id="ready-badge", classes="ready-badge hidden")

        # 3. System Area
        with Container(classes="system-area"):
            yield Label(self.time_str, classes="clock")

    def watch_time_str(self, time_str: str) -> None:
        try:
            self.query_one(".clock", Label).update(time_str)
        except Exception:
            pass

    def on_unmount(self) -> None:
        if hasattr(self.app, "state_registry"):
            self.app.state_registry.unregister(self)

    def on_state_updated(self, state: TuiState) -> None:
        """Update header badges from state."""
        # Update mode badge
        try:
            mode = state.data_source_mode.upper()
            mode_label = self.query_one("#mode-badge", Label)
            mode_label.update(mode)
            for cls in ("demo", "paper", "read_only", "live"):
                mode_label.remove_class(cls)
            mode_label.add_class(state.data_source_mode)
        except Exception as e:
            logger.debug("Failed updating mode badge: %s", e)

        self._update_connection_badge(state)
        self._update_ready_badge(state)

    def _update_connection_badge(self, state: TuiState) -> None:
        """Update the connection badge in the header.

        Called both from state updates and from the clock interval so the
        connection indicator remains accurate over time.
        """
        try:
            conn_label = self.query_one("#connection-status", Label)

            status_cls = "connected"
            if state.degraded_mode:
                text = "■ DEGRADED"
                status_cls = "disconnected"
            elif state.data_source_mode != "demo" and not state.running:
                # Manual-start policy: stopped is an expected, healthy state.
                text = "■ STOPPED"
                status_cls = "stopped"
            else:
                healthy = state.check_connection_health()
                if healthy:
                    latency = getattr(state.system_data, "api_latency", None)
                    latency_str = ""
                    if isinstance(latency, (int, float)) and latency > 0:
                        latency_str = f" {latency:.0f}ms"
                    text = f"● CONNECTED{latency_str}"
                    status_cls = "connected"
                else:
                    if not state.last_update_timestamp:
                        text = "○ CONNECTING"
                        status_cls = "stale"
                    else:
                        age = time.time() - state.last_update_timestamp
                        text = f"○ STALE {age:.0f}s"
                        status_cls = "stale"

            conn_label.update(text)
            conn_label.remove_class("connected", "stopped", "stale", "disconnected")
            conn_label.add_class(status_cls)
        except Exception as e:
            logger.debug("Failed updating connection badge: %s", e)

    def _update_ready_badge(self, state: TuiState) -> None:
        """Update the ready badge showing onboarding status.

        Shows "Ready" when setup is complete, or progress like "2/3" when incomplete.
        Hidden in demo mode since demo is always ready.
        """
        try:
            ready_label = self.query_one("#ready-badge", Label)
            status = self._onboarding.get_status(state)

            # Hide in demo mode - it's always ready
            if status.mode == "demo":
                ready_label.add_class("hidden")
                return

            # Show badge with status
            ready_label.remove_class("hidden")
            ready_label.remove_class("ready", "not-ready")

            if status.is_ready:
                ready_label.update("✓ READY")
                ready_label.add_class("ready")
            else:
                ready_label.update(f"○ {status.ready_label}")
                ready_label.add_class("not-ready")
        except Exception as e:
            logger.debug("Failed updating ready badge: %s", e)
