"""Tests for CFM-specific risk model dataclasses."""

from decimal import Decimal

from gpt_trader.features.live_trade.risk.manager import (
    ExposureState,
    RiskWarning,
    RiskWarningLevel,
)


class TestExposureState:
    """Tests for the ExposureState dataclass."""

    def test_default_values(self):
        """Default values are all zero."""
        state = ExposureState()
        assert state.spot_exposure == Decimal("0")
        assert state.cfm_exposure == Decimal("0")
        assert state.cfm_margin_used == Decimal("0")
        assert state.cfm_available_margin == Decimal("0")
        assert state.cfm_buying_power == Decimal("0")

    def test_total_exposure(self):
        """Total exposure sums spot and CFM."""
        state = ExposureState(
            spot_exposure=Decimal("10000"),
            cfm_exposure=Decimal("5000"),
        )
        assert state.total_exposure == Decimal("15000")

    def test_cfm_margin_utilization_empty(self):
        """Zero margin returns zero utilization."""
        state = ExposureState()
        assert state.cfm_margin_utilization == Decimal("0")

    def test_cfm_margin_utilization(self):
        """Margin utilization calculated correctly."""
        state = ExposureState(
            cfm_margin_used=Decimal("1000"),
            cfm_available_margin=Decimal("4000"),
        )
        assert state.cfm_margin_utilization == Decimal("0.2")

    def test_to_payload(self):
        """Exposure state serializes to dict."""
        state = ExposureState(
            spot_exposure=Decimal("10000"),
            cfm_exposure=Decimal("5000"),
        )
        payload = state.to_payload()
        assert payload["spot_exposure"] == "10000"
        assert payload["cfm_exposure"] == "5000"
        assert payload["total_exposure"] == "15000"


class TestRiskWarning:
    """Tests for the RiskWarning dataclass."""

    def test_warning_creation(self):
        """Can create risk warning."""
        warning = RiskWarning(
            level=RiskWarningLevel.WARNING,
            message="Test warning",
            action="REDUCE_POSITION",
            symbol="BTC-USD",
        )
        assert warning.level == RiskWarningLevel.WARNING
        assert warning.message == "Test warning"
        assert warning.action == "REDUCE_POSITION"
        assert warning.symbol == "BTC-USD"

    def test_critical_warning(self):
        """Can create critical warning."""
        warning = RiskWarning(
            level=RiskWarningLevel.CRITICAL,
            message="Critical test",
        )
        assert warning.level == RiskWarningLevel.CRITICAL

    def test_to_payload(self):
        """Warning serializes to dict."""
        warning = RiskWarning(
            level=RiskWarningLevel.WARNING,
            message="Test",
            action="TEST_ACTION",
            symbol="ETH-USD",
            details={"key": "value"},
        )
        payload = warning.to_payload()
        assert payload["level"] == "WARNING"
        assert payload["message"] == "Test"
        assert payload["action"] == "TEST_ACTION"
        assert payload["symbol"] == "ETH-USD"
        assert payload["details"] == {"key": "value"}
