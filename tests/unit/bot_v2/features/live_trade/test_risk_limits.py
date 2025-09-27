"""Risk limit enforcement tests."""

import pytest
from decimal import Decimal

from bot_v2.features.live_trade.risk import LiveRiskManager
from bot_v2.config.live_trade_config import RiskConfig
from bot_v2.features.brokerages.core.interfaces import Product, MarketType


class TestDailyLossLimits:
    """Daily P&L tracking and enforcement."""

    @pytest.fixture
    def risk_manager(self):
        config = RiskConfig(daily_loss_limit=Decimal("100"))
        return LiveRiskManager(config=config)

    def test_daily_loss_tracking(self, risk_manager):
        assert risk_manager.track_daily_pnl(Decimal("10000"), {}) is False
        risk_manager.track_daily_pnl(
            Decimal("10000"),
            {"BTC-PERP": {"realized_pnl": Decimal("-30"), "unrealized_pnl": Decimal("20")}},
        )

        assert risk_manager.daily_pnl == Decimal("-10")
        assert risk_manager.daily_pnl > -risk_manager.config.daily_loss_limit

    def test_daily_loss_limit_blocks_trading(self, risk_manager):
        risk_manager.track_daily_pnl(Decimal("10000"), {})
        triggered = risk_manager.track_daily_pnl(
            Decimal("10000"),
            {"BTC-PERP": {"realized_pnl": Decimal("-120"), "unrealized_pnl": Decimal("0")}},
        )

        assert triggered is True
        assert risk_manager.is_reduce_only_mode() is True

    def test_daily_reset(self, risk_manager):
        risk_manager.track_daily_pnl(Decimal("10000"), {})
        risk_manager.track_daily_pnl(
            Decimal("10000"),
            {"BTC-PERP": {"realized_pnl": Decimal("-50"), "unrealized_pnl": Decimal("0")}},
        )
        assert risk_manager.daily_pnl == Decimal("-50")

        risk_manager.reset_daily_tracking(current_equity=Decimal("10000"))
        assert risk_manager.daily_pnl == Decimal("0")
        assert risk_manager.start_of_day_equity == Decimal("10000")


class TestLiquidationBuffer:
    """Liquidation buffer checks."""

    @pytest.fixture
    def risk_manager(self):
        config = RiskConfig(
            min_liquidation_buffer_pct=0.15,
            enable_pre_trade_liq_projection=True,
        )
        return LiveRiskManager(config=config)

    @pytest.fixture
    def test_product(self):
        return Product(
            symbol="BTC-PERP",
            base_asset="BTC",
            quote_asset="USD",
            market_type=MarketType.PERPETUAL,
            min_size=Decimal("0.001"),
            step_size=Decimal("0.001"),
            min_notional=Decimal("1"),
            price_increment=Decimal("0.01"),
        )

    def test_liquidation_buffer_check(self, risk_manager, test_product):
        risk_manager.validate_liquidation_buffer(
            symbol="BTC-PERP",
            qty=Decimal("0.1"),
            price=Decimal("50000"),
            product=test_product,
            equity=Decimal("10000"),
        )

    def test_check_liquidation_buffer_runtime(self, risk_manager):
        safe = risk_manager.check_liquidation_buffer(
            symbol="BTC-PERP",
            position_data={"qty": Decimal("0.2"), "mark": Decimal("50000"), "liquidation_price": Decimal("40000")},
            equity=Decimal("15000"),
        )
        assert safe is False

        breach = risk_manager.check_liquidation_buffer(
            symbol="BTC-PERP",
            position_data={"qty": Decimal("5"), "mark": Decimal("50000")},
            equity=Decimal("1000"),
        )
        assert breach is True
