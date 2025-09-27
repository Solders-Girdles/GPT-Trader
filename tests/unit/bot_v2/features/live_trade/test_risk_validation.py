"""Risk validation unit tests for LiveRiskManager."""

import pytest
from decimal import Decimal

from bot_v2.features.live_trade.risk import LiveRiskManager, ValidationError
from bot_v2.config.live_trade_config import RiskConfig
from bot_v2.features.brokerages.core.interfaces import Product, MarketType


class TestBasicRiskValidation:
    """Validate leverage, exposure, and slippage guards."""

    @pytest.fixture
    def risk_manager(self):
        config = RiskConfig(
            max_leverage=5,
            daily_loss_limit=Decimal("100"),
            max_position_pct_per_symbol=0.2,
            slippage_guard_bps=50,
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

    def test_leverage_validation(self, risk_manager, test_product):
        risk_manager.validate_leverage(
            symbol="BTC-PERP",
            qty=Decimal("0.2"),
            price=Decimal("100000"),
            product=test_product,
            equity=Decimal("10000"),
        )

        with pytest.raises(ValidationError, match=r"(?i)leverage"):
            risk_manager.validate_leverage(
                symbol="BTC-PERP",
                qty=Decimal("0.6"),
                price=Decimal("100000"),
                product=test_product,
                equity=Decimal("10000"),
            )

    def test_exposure_limits(self, risk_manager, test_product):
        risk_manager.validate_exposure_limits(
            symbol="BTC-PERP",
            notional=Decimal("1500"),
            equity=Decimal("10000"),
            current_positions={},
        )

        with pytest.raises(ValidationError, match="exposure"):
            risk_manager.validate_exposure_limits(
                symbol="BTC-PERP",
                notional=Decimal("2500"),
                equity=Decimal("10000"),
                current_positions={},
            )

    def test_slippage_guard(self, risk_manager, test_product):
        risk_manager.validate_slippage_guard(
            symbol="BTC-PERP",
            side="buy",
            qty=Decimal("0.1"),
            expected_price=Decimal("50150"),
            mark_or_quote=Decimal("50000"),
        )

        with pytest.raises(ValidationError, match="slippage"):
            risk_manager.validate_slippage_guard(
                symbol="BTC-PERP",
                side="buy",
                qty=Decimal("0.1"),
                expected_price=Decimal("50300"),
                mark_or_quote=Decimal("50000"),
            )


class TestPreTradeValidation:
    """Full pre-trade validation flows."""

    @pytest.fixture
    def risk_manager(self):
        config = RiskConfig(
            max_leverage=5,
            daily_loss_limit=Decimal("100"),
            max_position_pct_per_symbol=0.2,
            kill_switch_enabled=False,
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

    def test_pre_trade_validation_passes(self, risk_manager, test_product):
        risk_manager.pre_trade_validate(
            symbol="BTC-PERP",
            side="buy",
            qty=Decimal("0.01"),
            price=Decimal("50000"),
            product=test_product,
            equity=Decimal("10000"),
            current_positions={},
        )

    def test_pre_trade_validation_with_kill_switch(self):
        config = RiskConfig(kill_switch_enabled=True)
        risk_manager = LiveRiskManager(config=config)

        product = Product(
            symbol="BTC-PERP",
            base_asset="BTC",
            quote_asset="USD",
            market_type=MarketType.PERPETUAL,
            min_size=Decimal("0.001"),
            step_size=Decimal("0.001"),
            min_notional=Decimal("1"),
            price_increment=Decimal("0.01"),
        )

        with pytest.raises(ValidationError, match=r"(?i)kill.?switch"):
            risk_manager.pre_trade_validate(
                symbol="BTC-PERP",
                side="sell",
                qty=Decimal("0.01"),
                price=Decimal("50000"),
                product=product,
                equity=Decimal("10000"),
                current_positions={},
            )
