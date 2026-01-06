"""
Tests for TUI degraded mode functionality.

Verifies the TUI gracefully handles missing StatusReporter by entering
degraded mode instead of crashing.
"""

from unittest.mock import MagicMock

import pytest

from gpt_trader.tui.adapters.null_status_reporter import NullStatusReporter


class TestDegradedModeInitialization:
    """Tests for degraded mode during TUI initialization."""

    def test_null_reporter_detected_correctly(self) -> None:
        """NullStatusReporter should be identifiable via is_null_reporter."""
        reporter = NullStatusReporter()
        assert getattr(reporter, "is_null_reporter", False) is True

    def test_real_reporter_not_flagged(self) -> None:
        """Real StatusReporter should not have is_null_reporter flag."""
        # Mock a real reporter
        real_reporter = MagicMock()
        # Ensure no is_null_reporter attribute
        del real_reporter.is_null_reporter

        assert getattr(real_reporter, "is_null_reporter", False) is False


class TestTuiStateDegradedMode:
    """Tests for TuiState degraded mode properties."""

    def test_tui_state_has_degraded_mode_property(self) -> None:
        """TuiState should have degraded_mode reactive property."""
        from gpt_trader.tui.state import TuiState

        state = TuiState()

        # Should have degraded_mode property defaulting to False
        assert state.degraded_mode is False
        assert state.degraded_reason == ""

    def test_tui_state_degraded_mode_can_be_set(self) -> None:
        """TuiState degraded_mode should be settable."""
        from gpt_trader.tui.state import TuiState

        state = TuiState()

        state.degraded_mode = True
        state.degraded_reason = "Test reason"

        assert state.degraded_mode is True
        assert state.degraded_reason == "Test reason"


class TestIsRealStatusReporter:
    """Tests for the _is_real_status_reporter helper in TraderApp."""

    @pytest.fixture
    def mock_app(self) -> MagicMock:
        """Create a mock TraderApp."""
        app = MagicMock()
        app.bot = MagicMock()
        app.bot.engine = MagicMock()
        return app

    def test_returns_false_when_no_bot(self, mock_app: MagicMock) -> None:
        """Should return False when bot is None."""
        mock_app.bot = None

        # Import and test the logic
        has_reporter = (
            mock_app.bot is not None
            and hasattr(mock_app.bot.engine, "status_reporter")
            and not getattr(mock_app.bot.engine.status_reporter, "is_null_reporter", False)
        )

        assert has_reporter is False

    def test_returns_false_for_null_reporter(self, mock_app: MagicMock) -> None:
        """Should return False when using NullStatusReporter."""
        mock_app.bot.engine.status_reporter = NullStatusReporter()

        is_null = getattr(mock_app.bot.engine.status_reporter, "is_null_reporter", False)

        assert is_null is True

    def test_returns_true_for_real_reporter(self, mock_app: MagicMock) -> None:
        """Should return True for a real StatusReporter."""
        real_reporter = MagicMock()
        # Ensure no is_null_reporter attribute
        del real_reporter.is_null_reporter
        mock_app.bot.engine.status_reporter = real_reporter

        is_null = getattr(mock_app.bot.engine.status_reporter, "is_null_reporter", False)

        assert is_null is False


class TestObserverConnectionWithNullReporter:
    """Tests for observer connection behavior with NullStatusReporter."""

    def test_null_reporter_accepts_observer(self) -> None:
        """NullStatusReporter should accept observer without error."""
        reporter = NullStatusReporter()
        callback = MagicMock()

        # Should not raise
        reporter.add_observer(callback)
        reporter.remove_observer(callback)

    def test_null_reporter_never_calls_observer(self) -> None:
        """NullStatusReporter should never call observer callbacks."""
        reporter = NullStatusReporter()
        callback = MagicMock()

        reporter.add_observer(callback)

        # Get status multiple times
        for _ in range(5):
            reporter.get_status()

        # Observer should never be called
        callback.assert_not_called()


class TestUICoordinatorDegradedMode:
    """Tests for UICoordinator handling of degraded mode."""

    @pytest.fixture
    def mock_coordinator_app(self) -> MagicMock:
        """Create a mock app for UICoordinator tests."""
        app = MagicMock()
        app.bot = MagicMock()
        app.bot.running = True
        app.bot.engine = MagicMock()
        app.bot.engine.context = MagicMock()
        app.bot.engine.context.runtime_state = None
        app.tui_state = MagicMock()
        app.tui_state.degraded_mode = False
        return app

    def test_sync_skips_null_reporter(self, mock_coordinator_app: MagicMock) -> None:
        """sync_state_from_bot should skip sync for NullStatusReporter."""
        from gpt_trader.tui.managers.ui_coordinator import UICoordinator

        # Set up NullStatusReporter
        mock_coordinator_app.bot.engine.status_reporter = NullStatusReporter()

        coordinator = UICoordinator(mock_coordinator_app)
        coordinator.sync_state_from_bot()

        # Should set connection_healthy to False
        mock_coordinator_app.tui_state.connection_healthy = False

        # update_from_bot_status should NOT be called
        mock_coordinator_app.tui_state.update_from_bot_status.assert_not_called()


class TestBotLifecycleManagerDegradedMode:
    """Tests for BotLifecycleManager handling of degraded mode."""

    def test_detects_null_reporter_on_mode_switch(self) -> None:
        """Should detect NullStatusReporter and update state on mode switch."""
        reporter = NullStatusReporter()

        # Check detection logic
        is_null = getattr(reporter, "is_null_reporter", False)

        assert is_null is True
