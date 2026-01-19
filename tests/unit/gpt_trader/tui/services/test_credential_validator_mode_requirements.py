from __future__ import annotations

from gpt_trader.tui.services.credential_validator import MODE_REQUIREMENTS


class TestModeRequirements:
    """Tests for MODE_REQUIREMENTS configuration."""

    def test_demo_requires_nothing(self) -> None:
        """Demo mode should not require any credentials."""
        req = MODE_REQUIREMENTS["demo"]
        assert req["requires_credentials"] is False
        assert req["requires_view"] is False
        assert req["requires_trade"] is False

    def test_read_only_requires_view(self) -> None:
        """Read-only mode requires view but not trade."""
        req = MODE_REQUIREMENTS["read_only"]
        assert req["requires_credentials"] is True
        assert req["requires_view"] is True
        assert req["requires_trade"] is False

    def test_paper_requires_view(self) -> None:
        """Paper mode requires view but not trade."""
        req = MODE_REQUIREMENTS["paper"]
        assert req["requires_credentials"] is True
        assert req["requires_view"] is True
        assert req["requires_trade"] is False

    def test_live_requires_trade(self) -> None:
        """Live mode requires both view and trade."""
        req = MODE_REQUIREMENTS["live"]
        assert req["requires_credentials"] is True
        assert req["requires_view"] is True
        assert req["requires_trade"] is True
