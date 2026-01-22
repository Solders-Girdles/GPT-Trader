"""Tests for CFM liquidation buffer and reduce-only mode."""

from gpt_trader.features.live_trade.risk.config import RiskConfig
from gpt_trader.features.live_trade.risk.manager import (
    LiveRiskManager,
    RiskWarning,
    RiskWarningLevel,
)


class TestLiveRiskManagerCFMLiquidationBuffer:
    """Tests for CFM liquidation buffer checks."""

    def test_check_cfm_liquidation_buffer_ok(self):
        """No warnings when buffer is sufficient."""
        config = RiskConfig(cfm_min_liquidation_buffer_pct=0.15)
        manager = LiveRiskManager(config=config, state_file=None)

        cfm_balance = {
            "total_usd_balance": "10000",
            "maintenance_margin": "5000",
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
        }

        warnings = manager.check_cfm_liquidation_buffer(cfm_balance)

        assert len(warnings) == 1
        assert warnings[0].level == RiskWarningLevel.WARNING
        assert "10.0%" in warnings[0].message
        assert warnings[0].action == "REDUCE_POSITION"
        assert not manager.is_cfm_reduce_only_mode()

    def test_check_cfm_liquidation_buffer_critical(self):
        """Critical warning and reduce-only when buffer is very low."""
        config = RiskConfig(cfm_min_liquidation_buffer_pct=0.20)
        manager = LiveRiskManager(config=config, state_file=None)

        cfm_balance = {
            "total_usd_balance": "10000",
            "maintenance_margin": "9500",
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


class TestLiveRiskManagerResetWithCFM:
    """Tests for reset_daily_tracking with CFM state."""

    def test_reset_daily_tracking_clears_cfm_state(self):
        """Reset clears CFM-specific state."""
        manager = LiveRiskManager(state_file=None)

        manager.set_cfm_reduce_only_mode(True, reason="test")
        manager._risk_warnings.append(RiskWarning(level=RiskWarningLevel.WARNING, message="test"))

        manager.reset_daily_tracking()

        assert not manager.is_cfm_reduce_only_mode()
        assert manager._cfm_reduce_only_reason == ""
        assert len(manager._risk_warnings) == 0


class TestLiveRiskManagerCFMRiskSummary:
    """Tests for CFM risk summary."""

    def test_get_cfm_risk_summary(self):
        """Can get CFM risk summary."""
        config = RiskConfig()
        manager = LiveRiskManager(config=config, state_file=None)

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

        cfm_balance = {"liquidation_buffer_percentage": 0.20}
        manager.check_cfm_liquidation_buffer(cfm_balance)

        warnings = manager.get_risk_warnings()
        assert len(warnings) >= 1

    def test_clear_risk_warnings(self):
        """Can clear all warnings."""
        config = RiskConfig(cfm_min_liquidation_buffer_pct=0.50)
        manager = LiveRiskManager(config=config, state_file=None)

        cfm_balance = {"liquidation_buffer_percentage": 0.20}
        manager.check_cfm_liquidation_buffer(cfm_balance)

        manager.clear_risk_warnings()

        assert len(manager.get_risk_warnings()) == 0
