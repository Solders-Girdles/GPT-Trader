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
