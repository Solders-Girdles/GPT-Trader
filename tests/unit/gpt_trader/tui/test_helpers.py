from unittest.mock import Mock

import pytest

from gpt_trader.tui import helpers


class _FakeApp:
    """Minimal stand-in for a Textual app."""

    def __init__(self, *, raise_error: bool = False) -> None:
        self.raise_error = raise_error
        self.run_calls = 0

    def run(self) -> None:
        self.run_calls += 1
        if self.raise_error:
            raise RuntimeError("boom")


def test_reset_terminal_modes_emits_mouse_disable_sequences(
    capsys: pytest.CaptureFixture[str],
) -> None:
    helpers.reset_terminal_modes()
    out = capsys.readouterr().out
    assert "\x1b[?1003l" in out
    assert "\x1b[?1006l" in out


def test_run_tui_app_with_cleanup_resets_on_success(capsys: pytest.CaptureFixture[str]) -> None:
    capsys.readouterr()  # Clear any buffered output
    app = _FakeApp()
    helpers.run_tui_app_with_cleanup(app)

    out = capsys.readouterr().out
    assert "\x1b[?1000l" in out
    assert app.run_calls == 1


def test_run_tui_app_with_cleanup_resets_on_error(capsys: pytest.CaptureFixture[str]) -> None:
    capsys.readouterr()
    app = _FakeApp(raise_error=True)

    with pytest.raises(RuntimeError):
        helpers.run_tui_app_with_cleanup(app)

    out = capsys.readouterr().out
    assert "\x1b[?1003l" in out
    assert app.run_calls == 1


# Tests for @safe_update decorator


def test_safe_update_backward_compatibility() -> None:
    """Test that existing @safe_update usage (no parameters) still works."""

    class TestWidget:
        @helpers.safe_update
        def failing_update(self) -> int:
            raise ValueError("Test error")

    widget = TestWidget()
    result = widget.failing_update()

    # Should catch error and return None
    assert result is None


def test_safe_update_success_case() -> None:
    """Test that @safe_update passes through successful calls."""

    class TestWidget:
        @helpers.safe_update
        def successful_update(self) -> int:
            return 42

    widget = TestWidget()
    result = widget.successful_update()

    assert result == 42


def test_safe_update_with_notify_user() -> None:
    """Test @safe_update with user notification enabled."""
    mock_app = Mock()
    mock_app.notify = Mock()

    class TestWidget:
        app = mock_app

        @helpers.safe_update(notify_user=True, severity="warning")
        def failing_update(self) -> None:
            raise ValueError("Test error")

    widget = TestWidget()
    result = widget.failing_update()

    # Should return None
    assert result is None

    # Should have called app.notify
    assert mock_app.notify.called
    call_args = mock_app.notify.call_args
    assert "TestWidget.failing_update" in str(call_args)
    assert call_args.kwargs["severity"] == "warning"
    assert call_args.kwargs["timeout"] == 5


def test_safe_update_with_error_tracker() -> None:
    """Test @safe_update with error tracker enabled."""
    mock_app = Mock()
    mock_tracker = Mock()
    mock_app.error_tracker = mock_tracker

    class TestWidget:
        app = mock_app

        @helpers.safe_update(error_tracker=True)
        def failing_update(self) -> None:
            raise ValueError("Test error")

    widget = TestWidget()
    result = widget.failing_update()

    # Should return None
    assert result is None

    # Should have called error_tracker.add_error
    assert mock_tracker.add_error.called
    call_args = mock_tracker.add_error.call_args
    assert call_args.kwargs["widget"] == "TestWidget"
    assert call_args.kwargs["method"] == "failing_update"
    assert "Test error" in call_args.kwargs["error"]


def test_safe_update_with_both_notify_and_tracker() -> None:
    """Test @safe_update with both notification and tracking enabled."""
    mock_app = Mock()
    mock_app.notify = Mock()
    mock_tracker = Mock()
    mock_app.error_tracker = mock_tracker

    class TestWidget:
        app = mock_app

        @helpers.safe_update(notify_user=True, error_tracker=True, severity="error")
        def failing_update(self) -> None:
            raise ValueError("Test error")

    widget = TestWidget()
    result = widget.failing_update()

    # Should return None
    assert result is None

    # Both notification and tracking should be called
    assert mock_app.notify.called
    assert mock_tracker.add_error.called


def test_safe_update_no_app_attribute() -> None:
    """Test @safe_update when widget has no app attribute."""

    class TestWidget:
        @helpers.safe_update(notify_user=True, error_tracker=True)
        def failing_update(self) -> None:
            raise ValueError("Test error")

    widget = TestWidget()
    result = widget.failing_update()

    # Should not crash even without app attribute
    assert result is None


def test_safe_update_app_is_none() -> None:
    """Test @safe_update when app attribute is None."""

    class TestWidget:
        app = None

        @helpers.safe_update(notify_user=True)
        def failing_update(self) -> None:
            raise ValueError("Test error")

    widget = TestWidget()
    result = widget.failing_update()

    # Should not crash when app is None
    assert result is None


def test_safe_update_notification_failure_doesnt_crash() -> None:
    """Test @safe_update handles notification failures gracefully."""
    mock_app = Mock()
    mock_app.notify = Mock(side_effect=RuntimeError("Notification failed"))

    class TestWidget:
        app = mock_app

        @helpers.safe_update(notify_user=True)
        def failing_update(self) -> None:
            raise ValueError("Test error")

    widget = TestWidget()
    result = widget.failing_update()

    # Should not crash even if notification fails
    assert result is None
    assert mock_app.notify.called


def test_safe_update_tracker_failure_doesnt_crash() -> None:
    """Test @safe_update handles tracker failures gracefully."""
    mock_app = Mock()
    mock_tracker = Mock()
    mock_tracker.add_error = Mock(side_effect=RuntimeError("Tracker failed"))
    mock_app.error_tracker = mock_tracker

    class TestWidget:
        app = mock_app

        @helpers.safe_update(error_tracker=True)
        def failing_update(self) -> None:
            raise ValueError("Test error")

    widget = TestWidget()
    result = widget.failing_update()

    # Should not crash even if tracker fails
    assert result is None
    assert mock_tracker.add_error.called
