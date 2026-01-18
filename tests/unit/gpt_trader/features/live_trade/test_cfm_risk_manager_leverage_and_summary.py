"""Tests for CFM leverage validation and CFM risk summaries."""

import pytest

from gpt_trader.features.live_trade.risk.config import RiskConfig
from gpt_trader.features.live_trade.risk.manager import (
    LiveRiskManager,
    ValidationError,
)


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
