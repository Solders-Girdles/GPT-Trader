"""Tests for CFM-specific risk management in LiveRiskManager."""

from decimal import Decimal

import pytest

from gpt_trader.features.live_trade.risk.config import RiskConfig
from gpt_trader.features.live_trade.risk.manager import (
    ExposureState,
    LiveRiskManager,
    RiskWarning,
    RiskWarningLevel,
    ValidationError,
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
        # Used 1000 out of total 5000 = 20%
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


class TestLiveRiskManagerCFMExposure:
    """Tests for CFM exposure tracking in LiveRiskManager."""

    def test_update_exposure_spot_only(self):
        """Updates spot exposure from positions."""
        manager = LiveRiskManager(state_file=None)

        positions = [
            {
                "quantity": Decimal("1"),
                "mark_price": Decimal("50000"),
                "product_type": "SPOT",
            },
            {
                "quantity": Decimal("10"),
                "mark_price": Decimal("3000"),
                "product_type": "SPOT",
            },
        ]

        state = manager.update_exposure(positions)

        assert state.spot_exposure == Decimal("80000")  # 50000 + 30000
        assert state.cfm_exposure == Decimal("0")

    def test_update_exposure_cfm_only(self):
        """Updates CFM exposure from positions with leverage."""
        manager = LiveRiskManager(state_file=None)

        positions = [
            {
                "quantity": Decimal("1"),
                "mark_price": Decimal("50000"),
                "product_type": "FUTURE",
                "leverage": 5,
            },
        ]

        state = manager.update_exposure(positions)

        assert state.spot_exposure == Decimal("0")
        assert state.cfm_exposure == Decimal("250000")  # 50000 * 5

    def test_update_exposure_hybrid(self):
        """Updates both spot and CFM exposure."""
        manager = LiveRiskManager(state_file=None)

        positions = [
            {
                "quantity": Decimal("1"),
                "mark_price": Decimal("50000"),
                "product_type": "SPOT",
            },
            {
                "quantity": Decimal("0.5"),
                "mark_price": Decimal("50000"),
                "product_type": "FUTURE",
                "leverage": 3,
            },
        ]

        state = manager.update_exposure(positions)

        assert state.spot_exposure == Decimal("50000")
        assert state.cfm_exposure == Decimal("75000")  # 25000 * 3
        assert state.total_exposure == Decimal("125000")

    def test_update_exposure_with_cfm_balance(self):
        """Updates margin info from CFM balance."""
        manager = LiveRiskManager(state_file=None)

        cfm_balance = {
            "margin_used": "1000",
            "available_margin": "4000",
            "futures_buying_power": "20000",
        }

        state = manager.update_exposure([], cfm_balance)

        assert state.cfm_margin_used == Decimal("1000")
        assert state.cfm_available_margin == Decimal("4000")
        assert state.cfm_buying_power == Decimal("20000")

    def test_get_exposure_state(self):
        """Can retrieve current exposure state."""
        manager = LiveRiskManager(state_file=None)

        positions = [
            {"quantity": "1", "mark_price": "50000", "product_type": "SPOT"},
        ]
        manager.update_exposure(positions)

        state = manager.get_exposure_state()
        assert state.spot_exposure == Decimal("50000")

    def test_get_total_exposure(self):
        """Can get total exposure directly."""
        manager = LiveRiskManager(state_file=None)

        positions = [
            {"quantity": "1", "mark_price": "50000", "product_type": "SPOT"},
            {"quantity": "1", "mark_price": "50000", "product_type": "FUTURE", "leverage": 2},
        ]
        manager.update_exposure(positions)

        assert manager.get_total_exposure() == Decimal("150000")


class TestLiveRiskManagerCFMLiquidationBuffer:
    """Tests for CFM liquidation buffer checks."""

    def test_check_cfm_liquidation_buffer_ok(self):
        """No warnings when buffer is sufficient."""
        config = RiskConfig(cfm_min_liquidation_buffer_pct=0.15)
        manager = LiveRiskManager(config=config, state_file=None)

        cfm_balance = {
            "total_usd_balance": "10000",
            "maintenance_margin": "5000",
            # Buffer = (10000 - 5000) / 10000 = 50%
        }

        warnings = manager.check_cfm_liquidation_buffer(cfm_balance)

        assert len(warnings) == 0
        assert not manager.is_cfm_reduce_only_mode()

    def test_check_cfm_liquidation_buffer_warning(self):
        """Warning when buffer is below threshold but not critical."""
        config = RiskConfig(cfm_min_liquidation_buffer_pct=0.15)
        manager = LiveRiskManager(config=config, state_file=None)

        cfm_balance = {
            "total_usd_balance": "10000",
            "maintenance_margin": "9000",
            # Buffer = (10000 - 9000) / 10000 = 10% < 15%
        }

        warnings = manager.check_cfm_liquidation_buffer(cfm_balance)

        assert len(warnings) == 1
        assert warnings[0].level == RiskWarningLevel.WARNING
        assert "10.0%" in warnings[0].message
        assert warnings[0].action == "REDUCE_POSITION"
        # Not critical, so no reduce-only mode
        assert not manager.is_cfm_reduce_only_mode()

    def test_check_cfm_liquidation_buffer_critical(self):
        """Critical warning and reduce-only when buffer is very low."""
        config = RiskConfig(cfm_min_liquidation_buffer_pct=0.20)
        manager = LiveRiskManager(config=config, state_file=None)

        cfm_balance = {
            "total_usd_balance": "10000",
            "maintenance_margin": "9500",
            # Buffer = (10000 - 9500) / 10000 = 5% < 10% (half of 20%)
        }

        warnings = manager.check_cfm_liquidation_buffer(cfm_balance)

        assert len(warnings) == 1
        assert warnings[0].level == RiskWarningLevel.CRITICAL
        assert manager.is_cfm_reduce_only_mode()

    def test_check_cfm_liquidation_buffer_with_percentage_field(self):
        """Uses liquidation_buffer_percentage field if available."""
        config = RiskConfig(cfm_min_liquidation_buffer_pct=0.15)
        manager = LiveRiskManager(config=config, state_file=None)

        cfm_balance = {
            "liquidation_buffer_percentage": 0.10,
        }

        warnings = manager.check_cfm_liquidation_buffer(cfm_balance)

        assert len(warnings) == 1
        assert warnings[0].level == RiskWarningLevel.WARNING

    def test_check_cfm_liquidation_buffer_none_balance(self):
        """Returns empty list for None balance."""
        manager = LiveRiskManager(state_file=None)

        warnings = manager.check_cfm_liquidation_buffer(None)

        assert warnings == []


class TestLiveRiskManagerCFMExposureLimits:
    """Tests for CFM exposure limit checks."""

    def test_check_cfm_exposure_limits_ok(self):
        """No warnings when within limits."""
        config = RiskConfig(cfm_max_exposure_pct=0.8)
        manager = LiveRiskManager(config=config, state_file=None)

        # Set up 50% exposure
        positions = [
            {"quantity": "1", "mark_price": "50000", "product_type": "FUTURE", "leverage": 1},
        ]
        manager.update_exposure(positions)

        warnings = manager.check_cfm_exposure_limits(equity=Decimal("100000"))

        assert len(warnings) == 0

    def test_check_cfm_exposure_limits_exceeded(self):
        """Warning when CFM exposure exceeds limit."""
        config = RiskConfig(cfm_max_exposure_pct=0.5)
        manager = LiveRiskManager(config=config, state_file=None)

        # Set up 75% exposure
        positions = [
            {"quantity": "1.5", "mark_price": "50000", "product_type": "FUTURE", "leverage": 1},
        ]
        manager.update_exposure(positions)

        warnings = manager.check_cfm_exposure_limits(equity=Decimal("100000"))

        assert len(warnings) == 1
        assert warnings[0].level == RiskWarningLevel.WARNING
        assert "75.0%" in warnings[0].message
        assert "50%" in warnings[0].message

    def test_check_cfm_exposure_limits_zero_equity(self):
        """Returns empty list for zero equity."""
        manager = LiveRiskManager(state_file=None)

        warnings = manager.check_cfm_exposure_limits(equity=Decimal("0"))

        assert warnings == []


class TestLiveRiskManagerCFMLeverageValidation:
    """Tests for CFM leverage validation."""

    def test_validate_cfm_leverage_ok(self):
        """No error when leverage within limit."""
        config = RiskConfig(cfm_max_leverage=5)
        manager = LiveRiskManager(config=config, state_file=None)

        # Should not raise
        result = manager.validate_cfm_leverage("BTC-20DEC30-CDE", requested_leverage=3)
        assert result is None

    def test_validate_cfm_leverage_at_limit(self):
        """No error when leverage at limit."""
        config = RiskConfig(cfm_max_leverage=5)
        manager = LiveRiskManager(config=config, state_file=None)

        # Should not raise
        result = manager.validate_cfm_leverage("BTC-20DEC30-CDE", requested_leverage=5)
        assert result is None

    def test_validate_cfm_leverage_exceeded(self):
        """Error when leverage exceeds limit."""
        config = RiskConfig(cfm_max_leverage=5)
        manager = LiveRiskManager(config=config, state_file=None)

        with pytest.raises(ValidationError) as exc_info:
            manager.validate_cfm_leverage("BTC-20DEC30-CDE", requested_leverage=10)

        assert "10x exceeds CFM limit 5x" in str(exc_info.value)

    def test_validate_cfm_leverage_default(self):
        """Uses default 5x when no config."""
        manager = LiveRiskManager(state_file=None)

        with pytest.raises(ValidationError):
            manager.validate_cfm_leverage("BTC-20DEC30-CDE", requested_leverage=10)

        # 5x should be ok
        manager.validate_cfm_leverage("BTC-20DEC30-CDE", requested_leverage=5)


class TestLiveRiskManagerCFMReduceOnlyMode:
    """Tests for CFM-specific reduce-only mode."""

    def test_cfm_reduce_only_mode_default(self):
        """CFM reduce-only mode is off by default."""
        manager = LiveRiskManager(state_file=None)
        assert not manager.is_cfm_reduce_only_mode()

    def test_set_cfm_reduce_only_mode(self):
        """Can set CFM reduce-only mode."""
        manager = LiveRiskManager(state_file=None)

        manager.set_cfm_reduce_only_mode(True, reason="test_reason")

        assert manager.is_cfm_reduce_only_mode()
        assert manager._cfm_reduce_only_reason == "test_reason"

    def test_clear_cfm_reduce_only_mode(self):
        """Can clear CFM reduce-only mode."""
        manager = LiveRiskManager(state_file=None)
        manager.set_cfm_reduce_only_mode(True, reason="test")

        manager.set_cfm_reduce_only_mode(False)

        assert not manager.is_cfm_reduce_only_mode()


class TestLiveRiskManagerCFMRiskSummary:
    """Tests for CFM risk summary."""

    def test_get_cfm_risk_summary(self):
        """Can get CFM risk summary."""
        config = RiskConfig()
        manager = LiveRiskManager(config=config, state_file=None)

        # Set up some state
        positions = [
            {"quantity": "1", "mark_price": "50000", "product_type": "FUTURE", "leverage": 2},
        ]
        manager.update_exposure(positions)
        manager.set_cfm_reduce_only_mode(True, reason="test")

        summary = manager.get_cfm_risk_summary()

        assert "exposure" in summary
        assert summary["exposure"]["cfm_exposure"] == "100000"
        assert summary["reduce_only_mode"] is True
        assert summary["reduce_only_reason"] == "test"
        assert "warnings_count" in summary
        assert "warnings" in summary

    def test_get_risk_warnings(self):
        """Can get all risk warnings."""
        config = RiskConfig(cfm_min_liquidation_buffer_pct=0.50)
        manager = LiveRiskManager(config=config, state_file=None)

        # Trigger some warnings
        cfm_balance = {"liquidation_buffer_percentage": 0.20}
        manager.check_cfm_liquidation_buffer(cfm_balance)

        warnings = manager.get_risk_warnings()
        assert len(warnings) >= 1

    def test_clear_risk_warnings(self):
        """Can clear all warnings."""
        config = RiskConfig(cfm_min_liquidation_buffer_pct=0.50)
        manager = LiveRiskManager(config=config, state_file=None)

        # Trigger warnings
        cfm_balance = {"liquidation_buffer_percentage": 0.20}
        manager.check_cfm_liquidation_buffer(cfm_balance)

        manager.clear_risk_warnings()

        assert len(manager.get_risk_warnings()) == 0


class TestLiveRiskManagerResetWithCFM:
    """Tests for reset_daily_tracking with CFM state."""

    def test_reset_daily_tracking_clears_cfm_state(self):
        """Reset clears CFM-specific state."""
        manager = LiveRiskManager(state_file=None)

        # Set up CFM state
        manager.set_cfm_reduce_only_mode(True, reason="test")
        manager._risk_warnings.append(RiskWarning(level=RiskWarningLevel.WARNING, message="test"))

        manager.reset_daily_tracking()

        assert not manager.is_cfm_reduce_only_mode()
        assert manager._cfm_reduce_only_reason == ""
        assert len(manager._risk_warnings) == 0
