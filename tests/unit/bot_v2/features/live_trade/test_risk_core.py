"""Risk core tests covering calculations, limits, and validation flows."""

import datetime as dt
from decimal import Decimal

import pytest

from bot_v2.config.live_trade_config import RiskConfig
from bot_v2.features.brokerages.core.interfaces import MarketType, Product
from bot_v2.features.live_trade.risk import LiveRiskManager, ValidationError
from bot_v2.features.live_trade.risk_calculations import (
    effective_mmr,
    effective_symbol_leverage_cap,
    evaluate_daytime_window,
)


class _CalcConfig:
    max_leverage = 5
    leverage_max_per_symbol = {"BTC-PERP": 10}
    day_leverage_max_per_symbol = {"BTC-PERP": 6}
    night_leverage_max_per_symbol = {"BTC-PERP": 3}
    default_maintenance_margin_rate = 0.01
    day_mmr_per_symbol = {"BTC-PERP": 0.015}
    night_mmr_per_symbol = {"BTC-PERP": 0.02}
    daytime_start_utc = "09:00"
    daytime_end_utc = "17:00"


class _Logger:
    def __init__(self):
        self.debug_calls = []

    def debug(self, msg, *args):
        self.debug_calls.append(msg % args if args else msg)


@pytest.fixture(name="calc_config")
def fixture_calc_config():
    return _CalcConfig()


def test_evaluate_daytime_window_day(calc_config):
    now = dt.datetime(2025, 1, 6, 10, 0, tzinfo=dt.timezone.utc)
    assert evaluate_daytime_window(calc_config, now) is True


def test_evaluate_daytime_window_night(calc_config):
    now = dt.datetime(2025, 1, 6, 20, 0, tzinfo=dt.timezone.utc)
    assert evaluate_daytime_window(calc_config, now) is False


def test_effective_leverage_cap_respects_day_schedule(calc_config):
    logger = _Logger()
    cap = effective_symbol_leverage_cap(
        "BTC-PERP",
        calc_config,
        now=dt.datetime(2025, 1, 6, 10, 0, tzinfo=dt.timezone.utc),
        risk_info_provider=None,
        logger=logger,
    )
    assert cap == 6


def test_effective_leverage_cap_provider_override(calc_config):
    logger = _Logger()
    cap = effective_symbol_leverage_cap(
        "BTC-PERP",
        calc_config,
        now=dt.datetime(2025, 1, 6, 22, 0, tzinfo=dt.timezone.utc),
        risk_info_provider=lambda symbol: {"max_leverage": 2},
        logger=logger,
    )
    assert cap == 2


def test_effective_mmr_prefers_provider(calc_config):
    logger = _Logger()
    mmr = effective_mmr(
        "BTC-PERP",
        calc_config,
        now=dt.datetime(2025, 1, 6, 22, 0, tzinfo=dt.timezone.utc),
        risk_info_provider=lambda symbol: {"maintenance_margin_rate": 0.03},
        logger=logger,
    )
    assert mmr == Decimal("0.03")


def test_effective_mmr_falls_back_to_schedule(calc_config):
    logger = _Logger()
    mmr = effective_mmr(
        "BTC-PERP",
        calc_config,
        now=dt.datetime(2025, 1, 6, 21, 0, tzinfo=dt.timezone.utc),
        risk_info_provider=None,
        logger=logger,
    )
    assert mmr == Decimal("0.02")


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
            quantity=Decimal("0.1"),
            price=Decimal("50000"),
            product=test_product,
            equity=Decimal("10000"),
        )

    def test_check_liquidation_buffer_runtime(self, risk_manager):
        safe = risk_manager.check_liquidation_buffer(
            symbol="BTC-PERP",
            position_data={
                "quantity": Decimal("0.2"),
                "mark": Decimal("50000"),
                "liquidation_price": Decimal("40000"),
            },
            equity=Decimal("15000"),
        )
        assert safe is False

        breach = risk_manager.check_liquidation_buffer(
            symbol="BTC-PERP",
            position_data={"quantity": Decimal("5"), "mark": Decimal("50000")},
            equity=Decimal("1000"),
        )
        assert breach is True


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
            quantity=Decimal("0.2"),
            price=Decimal("100000"),
            product=test_product,
            equity=Decimal("10000"),
        )

        with pytest.raises(ValidationError, match=r"(?i)leverage"):
            risk_manager.validate_leverage(
                symbol="BTC-PERP",
                quantity=Decimal("0.6"),
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
            quantity=Decimal("0.1"),
            expected_price=Decimal("50150"),
            mark_or_quote=Decimal("50000"),
        )

        with pytest.raises(ValidationError, match="slippage"):
            risk_manager.validate_slippage_guard(
                symbol="BTC-PERP",
                side="buy",
                quantity=Decimal("0.1"),
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
            quantity=Decimal("0.01"),
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
                quantity=Decimal("0.01"),
                price=Decimal("50000"),
                product=product,
                equity=Decimal("10000"),
                current_positions={},
            )


@pytest.mark.xfail(reason="TODO(2025-01-31): circuit breaker APIs still pending", strict=False)
def test_circuit_breakers_placeholder():
    raise NotImplementedError("Circuit breaker methods not yet implemented")


@pytest.mark.xfail(reason="TODO(2025-01-31): impact cost modelling pending", strict=False)
def test_impact_cost_placeholder():
    raise NotImplementedError("Impact cost methods not yet implemented")


@pytest.mark.xfail(reason="TODO(2025-01-31): dynamic position sizing backlog", strict=False)
def test_position_sizing_placeholder():
    raise NotImplementedError("Position sizing methods not yet implemented")


@pytest.mark.xfail(reason="TODO(2025-01-31): risk metrics aggregation backlog", strict=False)
def test_risk_metrics_placeholder():
    raise NotImplementedError("Risk metrics methods not yet implemented")
