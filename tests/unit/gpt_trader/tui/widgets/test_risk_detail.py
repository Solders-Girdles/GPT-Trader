"""Tests for RiskDetailModal guard visibility and focus preview helpers."""

import time

from gpt_trader.tui.types import RiskGuard, RiskState
from gpt_trader.tui.widgets.risk_detail_modal import RiskDetailModal


class TestEnhancedGuardVisibility:
    """Tests for enhanced guard visibility features."""

    def test_get_sorted_guards_by_severity(self):
        """Guards are sorted by severity (highest first)."""
        guards = [
            RiskGuard(name="LowGuard", severity="LOW"),
            RiskGuard(name="CriticalGuard", severity="CRITICAL"),
            RiskGuard(name="MediumGuard", severity="MEDIUM"),
            RiskGuard(name="HighGuard", severity="HIGH"),
        ]
        data = RiskState(guards=guards)
        modal = RiskDetailModal(data)

        sorted_guards = modal._get_sorted_guards(data)
        assert len(sorted_guards) == 4
        assert sorted_guards[0].name == "CriticalGuard"
        assert sorted_guards[1].name == "HighGuard"
        assert sorted_guards[2].name == "MediumGuard"
        assert sorted_guards[3].name == "LowGuard"

    def test_format_guard_row_with_recent_trigger(self):
        """Guard row shows recent trigger in red."""
        now = time.time()
        guard = RiskGuard(
            name="TestGuard",
            severity="HIGH",
            last_triggered=now - 30,  # 30 seconds ago
            triggered_count=5,
        )
        data = RiskState()
        modal = RiskDetailModal(data)

        formatted = modal._format_guard_row(guard)
        text_str = str(formatted)
        assert "TestGuard" in text_str
        assert "30s ago" in text_str or "29s ago" in text_str or "31s ago" in text_str
        assert "×5" in text_str  # Trigger count

    def test_format_guard_row_never_triggered(self):
        """Guard row shows 'never' for untriggered guards."""
        guard = RiskGuard(
            name="TestGuard",
            severity="MEDIUM",
            last_triggered=0.0,  # Never triggered
            triggered_count=0,
        )
        data = RiskState()
        modal = RiskDetailModal(data)

        formatted = modal._format_guard_row(guard)
        text_str = str(formatted)
        assert "TestGuard" in text_str
        assert "never" in text_str

    def test_format_age_seconds(self):
        """Age formatting for seconds."""
        data = RiskState()
        modal = RiskDetailModal(data)

        now = time.time()
        assert modal._format_age(now - 45) == "45s ago"

    def test_format_age_minutes(self):
        """Age formatting for minutes."""
        data = RiskState()
        modal = RiskDetailModal(data)

        now = time.time()
        assert modal._format_age(now - 180) == "3m ago"

    def test_format_age_hours(self):
        """Age formatting for hours."""
        data = RiskState()
        modal = RiskDetailModal(data)

        now = time.time()
        assert modal._format_age(now - 7200) == "2h ago"

    def test_format_age_days(self):
        """Age formatting for days."""
        data = RiskState()
        modal = RiskDetailModal(data)

        now = time.time()
        assert modal._format_age(now - 172800) == "2d ago"

    def test_severity_order_property(self):
        """RiskGuard.severity_order returns correct numeric values."""
        assert RiskGuard(name="test", severity="LOW").severity_order == 1
        assert RiskGuard(name="test", severity="MEDIUM").severity_order == 2
        assert RiskGuard(name="test", severity="HIGH").severity_order == 3
        assert RiskGuard(name="test", severity="CRITICAL").severity_order == 4

    def test_severity_order_case_insensitive(self):
        """Severity order works regardless of case."""
        assert RiskGuard(name="test", severity="low").severity_order == 1
        assert RiskGuard(name="test", severity="HIGH").severity_order == 3
        assert RiskGuard(name="test", severity="Critical").severity_order == 4


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
