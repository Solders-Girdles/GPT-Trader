"""Tests for CFM exposure tracking and limit checks in LiveRiskManager."""

from decimal import Decimal

from gpt_trader.features.live_trade.risk.config import RiskConfig
from gpt_trader.features.live_trade.risk.manager import LiveRiskManager, RiskWarningLevel


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
