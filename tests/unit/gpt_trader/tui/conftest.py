"""
TUI Test Configuration.

Provides fixtures for TUI testing including mock bots, pilot apps, and factories.

Note: pickle is required for pytest-textual-snapshot report compatibility.
"""

import asyncio
import inspect
import pickle  # noqa: S403 - Required for pytest-textual-snapshot report format
import re
from collections.abc import Awaitable, Callable, Iterable
from importlib.util import module_from_spec, spec_from_file_location  # naming: allow
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_textual_snapshot as pts
from _pytest.fixtures import FixtureRequest
from rich.console import Console
from syrupy import SnapshotAssertion
from syrupy.extensions.single_file import SingleFileSnapshotExtension, WriteMode

# Re-export factories for convenient importing
from tests.unit.gpt_trader.tui.factories import (
    BotStatusFactory,
    MarketDataFactory,
    OrderFactory,
    PositionFactory,
    TradeFactory,
    TuiStateFactory,
)
from textual.app import App
from textual.pilot import Pilot

from gpt_trader.monitoring.status_reporter import StatusReporter
from gpt_trader.tui.app import TraderApp

# Patterns to normalize for stable snapshots
_TERMINAL_HASH_RE = re.compile(r"terminal-\d+-")
# Normalize timing values like "(157ms)" to "(XXms)" for API connectivity checks
_TIMING_RE = re.compile(r"\((\d+)ms\)")
# Normalize UTC time display like "19:22&#160;UTC" to "XX:XX&#160;UTC"
_UTC_TIME_RE = re.compile(r"\d{2}:\d{2}(&#160;| )UTC")


def _normalize_svg(text: str) -> str:
    """Apply all normalizations to SVG snapshot content."""
    text = _TERMINAL_HASH_RE.sub("terminal-", text)
    text = _TIMING_RE.sub("(XXms)", text)
    text = _UTC_TIME_RE.sub("XX:XX UTC", text)
    return text


class NormalizedSVGImageExtension(SingleFileSnapshotExtension):
    """SVG extension that normalizes Textual's hashed terminal CSS class names."""

    _file_extension = "svg"
    _write_mode = WriteMode.TEXT

    def serialize(self, data, *, exclude=None, include=None, matcher=None):
        """Normalize volatile content before serializing."""
        text = super().serialize(data, exclude=exclude, include=include, matcher=matcher)
        return _normalize_svg(text)


def _import_app_from_path(app_path: str) -> App[Any]:
    """Import an App class from a file path using public stdlib APIs.

    This replaces textual._import_app.import_app with standard library equivalents.
    """
    path = Path(app_path)
    spec = spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {app_path}")

    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    # Find the App subclass in the module
    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, type) and issubclass(obj, App) and obj is not App:
            return obj()

    raise ImportError(f"No App subclass found in {app_path}")


async def _capture_screenshot(
    app: App[Any],
    press: Iterable[str],
    terminal_size: tuple[int, int],
    run_before: Callable[[Pilot], Awaitable[None] | None] | None,
) -> str:
    """Capture an SVG screenshot using public Textual APIs.

    This replaces textual._doc.take_svg_screenshot with run_test() + export_screenshot().
    """
    async with app.run_test(size=terminal_size) as pilot:
        # Run any setup code
        if run_before is not None:
            result = run_before(pilot)
            if asyncio.iscoroutine(result):
                await result

        # Apply key presses
        for key in press:
            if key == "_":
                await pilot.pause()
            else:
                await pilot.press(key)

        # Small pause to let UI settle
        await pilot.pause()

        # Capture screenshot using public API
        return app.export_screenshot()


@pytest.fixture
def snap_compare(
    snapshot: SnapshotAssertion, request: FixtureRequest
) -> Callable[[str | Path | App[Any]], bool]:
    """
    Snapshot comparison fixture with normalization for Textual's hashed CSS classes.

    This overrides pytest-textual-snapshot's snap_compare to strip the session-specific
    terminal-<digits>- prefix from SVG output, ensuring snapshots remain stable across runs.

    Uses only public Textual APIs (no private _doc or _import_app imports).
    """
    # Use our normalized extension
    snapshot = snapshot.use_extension(NormalizedSVGImageExtension)

    def compare(
        app: str | Path | App[Any],
        press: Iterable[str] = (),
        terminal_size: tuple[int, int] = (80, 24),
        run_before: Callable[[Pilot], Awaitable[None] | None] | None = None,
    ) -> bool:
        """Compare a screenshot with normalized terminal class names."""
        node = request.node

        if isinstance(app, App):
            app_instance = app
            app_path = ""
        else:
            path = Path(app)
            if path.is_absolute():
                app_path = str(path.resolve())
            else:
                node_path = node.path.parent
                resolved = (node_path / app).resolve()
                app_path = str(resolved)
            app_instance = _import_app_from_path(app_path)

        # Capture screenshot using public APIs
        actual_screenshot = asyncio.get_event_loop().run_until_complete(
            _capture_screenshot(app_instance, press, terminal_size, run_before)
        )

        # Normalize the actual screenshot
        actual_screenshot = _normalize_svg(actual_screenshot)

        console = Console(legacy_windows=False, force_terminal=True)
        p_app = pts.PseudoApp(pts.PseudoConsole(console.legacy_windows, console.size))

        result = snapshot == actual_screenshot

        # Store data for report generation (matches plugin behavior)
        execution_index = (
            snapshot._custom_index and snapshot._execution_name_index.get(snapshot._custom_index)
        ) or snapshot.num_executions - 1
        assertion_result = snapshot.executions.get(execution_index)

        snapshot_exists = (
            execution_index in snapshot.executions
            and assertion_result
            and assertion_result.final_data is not None
        )

        expected_svg_text = str(snapshot)
        full_path, line_number, name = node.reportinfo()

        data = (
            result,
            expected_svg_text,
            actual_screenshot,
            p_app,
            full_path,
            line_number,
            name,
            inspect.getdoc(node.function) or "",
            app_path,
            snapshot_exists,
        )
        data_path = pts.node_to_report_path(request.node)
        data_path.write_bytes(pickle.dumps(data))  # noqa: S301

        return result

    return compare


@pytest.fixture
def mock_bot():
    """
    Creates a mock TradingBot with all necessary components for TUI testing.
    """
    bot = MagicMock()
    bot.running = False
    bot.run = AsyncMock()
    bot.stop = AsyncMock()
    bot.config = MagicMock()

    # Mock engine and status reporter
    bot.engine = MagicMock()
    bot.engine.status_reporter = StatusReporter()
    bot.engine.context = MagicMock()
    bot.engine.context.runtime_state = None

    return bot


@pytest.fixture
def mock_app(mock_bot):
    """
    Creates a TraderApp instance with a mock bot.
    """
    return TraderApp(bot=mock_bot)


@pytest.fixture
async def pilot_app(mock_bot):
    """
    Creates a TraderApp with Pilot for interactive testing.

    Use this fixture to test keyboard interactions, widget updates,
    and screen navigation flows.

    Usage:
        async def test_start_stop(pilot_app):
            pilot, app = pilot_app
            await pilot.press("s")  # Press 's' to start bot
            await pilot.pause()     # Wait for UI updates
            assert app.tui_state.running is True

    Yields:
        tuple[Pilot, TraderApp]: The pilot instance and app for testing
    """
    app = TraderApp(bot=mock_bot)
    async with app.run_test() as pilot:
        yield pilot, app


@pytest.fixture
def bot_status_factory():
    """Provides access to BotStatusFactory for creating test BotStatus objects."""
    return BotStatusFactory


@pytest.fixture
def tui_state_factory():
    """Provides access to TuiStateFactory for creating test TuiState objects."""
    return TuiStateFactory


@pytest.fixture
def market_data_factory():
    """Provides access to MarketDataFactory for creating test market data."""
    return MarketDataFactory


@pytest.fixture
def position_factory():
    """Provides access to PositionFactory for creating test positions."""
    return PositionFactory


@pytest.fixture
def order_factory():
    """Provides access to OrderFactory for creating test orders."""
    return OrderFactory


@pytest.fixture
def trade_factory():
    """Provides access to TradeFactory for creating test trades."""
    return TradeFactory
