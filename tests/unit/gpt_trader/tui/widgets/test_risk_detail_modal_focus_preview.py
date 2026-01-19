"""Tests for RiskDetailModal focus preview parameter."""

from gpt_trader.tui.types import RiskState
from gpt_trader.tui.widgets.risk_detail_modal import RiskDetailModal


class TestFocusPreviewParameter:
    """Tests for focus_preview parameter functionality."""

    def test_default_focus_preview_is_false(self):
        """Modal defaults to focus_preview=False."""
        data = RiskState()
        modal = RiskDetailModal(data)
        assert modal._focus_preview is False

    def test_focus_preview_parameter_stored(self):
        """Modal stores focus_preview parameter when True."""
        data = RiskState()
        modal = RiskDetailModal(data, focus_preview=True)
        assert modal._focus_preview is True

    def test_focus_preview_explicit_false(self):
        """Modal accepts explicit focus_preview=False."""
        data = RiskState()
        modal = RiskDetailModal(data, focus_preview=False)
        assert modal._focus_preview is False
