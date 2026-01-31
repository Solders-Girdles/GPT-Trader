from __future__ import annotations

# Autouse fixtures here isolate preferences, patch Pilot.pause timing, and clear TUI singletons.
import pytest

# Small pause to let Textual process events in tests without slowing runs too much.
_DEFAULT_PILOT_PAUSE_SECONDS = 0.01
_TUI_TEST_PATH_FRAGMENT = "tests/unit/gpt_trader/tui"


def _is_tui_test(request) -> bool:
    node_id = getattr(request.node, "nodeid", "")
    if _TUI_TEST_PATH_FRAGMENT in str(node_id).replace("\\", "/"):
        return True
    node_path = getattr(request.node, "path", None) or getattr(request.node, "fspath", "")
    return _TUI_TEST_PATH_FRAGMENT in str(node_path).replace("\\", "/")


# ============================================================
# Theme isolation for deterministic tests
# ============================================================


@pytest.fixture(autouse=True)
def isolate_tui_preferences(request):
    """Isolate TUI preferences for all TUI tests.

    This prevents tests from reading/writing to the real preferences file,
    ensuring consistent theme and mode settings across test runs.

    Tests that need to verify default preferences path behavior should use
    the 'uses_real_preferences' marker to skip this fixture.
    """
    if not _is_tui_test(request):
        return

    monkeypatch = request.getfixturevalue("monkeypatch")
    tmp_path = request.getfixturevalue("tmp_path")

    # Ensure Textual color output isn't suppressed by environment defaults.
    monkeypatch.delenv("NO_COLOR", raising=False)

    if request.node.get_closest_marker("uses_real_preferences"):
        # Allow test to use real preferences path
        return

    prefs_file = tmp_path / "test_preferences.json"
    prefs_file.write_text('{"theme": "dark"}')
    monkeypatch.setenv("GPT_TRADER_TUI_PREFERENCES_PATH", str(prefs_file))


# ============================================================
# Pilot pause stabilization
# ============================================================


@pytest.fixture(autouse=True)
def stabilize_pilot_pause(request):
    """Ensure pilot.pause always yields at least a short delay."""
    if not _is_tui_test(request):
        return

    monkeypatch = request.getfixturevalue("monkeypatch")
    from textual.pilot import Pilot

    original_pause = Pilot.pause

    async def _pause(self, delay: float | None = None) -> None:
        await original_pause(self, _DEFAULT_PILOT_PAUSE_SECONDS if delay is None else delay)

    monkeypatch.setattr(Pilot, "pause", _pause)


# ============================================================
# Singleton cleanup for test isolation
# ============================================================


@pytest.fixture(autouse=True)
def clear_tui_singletons(request):
    """Clear TUI singleton services before each test for isolation.

    This prevents state leakage between tests which can cause flaky
    snapshot comparisons due to accumulated state from prior tests.
    """
    if not _is_tui_test(request):
        yield
        return

    # Clear before test
    _clear_all_tui_singletons()
    yield
    # Clear after test
    _clear_all_tui_singletons()


def _clear_all_tui_singletons():
    """Clear all TUI singleton services."""
    from gpt_trader.tui.services.onboarding_service import clear_onboarding_service
    from gpt_trader.tui.services.performance_service import clear_tui_performance_service
    from gpt_trader.tui.services.trading_stats_service import clear_trading_stats_service

    clear_tui_performance_service()
    clear_trading_stats_service()
    clear_onboarding_service()

    # Clear theme manager
    import gpt_trader.tui.theme as theme_module

    theme_module._theme_manager = None

    # Clear log handler
    import gpt_trader.tui.log_manager as log_module

    if log_module._tui_log_handler is not None:
        log_module.detach_tui_log_handler()
        log_module._tui_log_handler = None

    # Clear execution telemetry
    from gpt_trader.features.live_trade.telemetry import clear_execution_telemetry

    clear_execution_telemetry()

    # Clear preferences service
    import gpt_trader.tui.services.preferences_service as prefs_module

    prefs_module._preferences_service = None

    # Clear global metrics collector to avoid cross-suite leakage
    from gpt_trader.monitoring.metrics_collector import reset_all as reset_metrics

    reset_metrics()
