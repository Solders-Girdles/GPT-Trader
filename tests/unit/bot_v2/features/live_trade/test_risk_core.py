"""Risk core tests covering calculations, limits, and validation flows."""

import datetime as dt
import math
from decimal import Decimal

import pytest

from bot_v2.config.live_trade_config import RiskConfig
from bot_v2.features.brokerages.core.interfaces import MarketType, Product
from bot_v2.features.live_trade.risk import (
    ImpactAssessment,
    ImpactRequest,
    LiveRiskManager,
    ValidationError,
)
from bot_v2.features.live_trade.risk_metrics import RiskMetricsAggregator
from bot_v2.features.live_trade.risk_calculations import (
    effective_mmr,
    effective_symbol_leverage_cap,
    evaluate_daytime_window,
)
from bot_v2.features.live_trade.risk_runtime import CircuitBreakerAction


class _CalculationConfig:
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


class _RecordingEventStore:
    def __init__(self) -> None:
        self.metrics: list[dict] = []

    def append_metric(self, **kwargs) -> None:  # pragma: no cover - exercised in tests
        self.metrics.append(kwargs)


@pytest.fixture(name="calculation_config")
def fixture_calculation_config():
    return _CalculationConfig()


def test_evaluate_daytime_window_day(calculation_config):
    now = dt.datetime(2025, 1, 6, 10, 0, tzinfo=dt.timezone.utc)
    assert evaluate_daytime_window(calculation_config, now) is True


def test_evaluate_daytime_window_night(calculation_config):
    now = dt.datetime(2025, 1, 6, 20, 0, tzinfo=dt.timezone.utc)
    assert evaluate_daytime_window(calculation_config, now) is False


def test_effective_leverage_cap_respects_day_schedule(calculation_config):
    logger = _Logger()
    cap = effective_symbol_leverage_cap(
        "BTC-PERP",
        calculation_config,
        now=dt.datetime(2025, 1, 6, 10, 0, tzinfo=dt.timezone.utc),
        risk_info_provider=None,
        logger=logger,
    )
    assert cap == 6


def test_effective_leverage_cap_provider_override(calculation_config):
    logger = _Logger()
    cap = effective_symbol_leverage_cap(
        "BTC-PERP",
        calculation_config,
        now=dt.datetime(2025, 1, 6, 22, 0, tzinfo=dt.timezone.utc),
        risk_info_provider=lambda symbol: {"max_leverage": 2},
        logger=logger,
    )
    assert cap == 2


def test_effective_mmr_prefers_provider(calculation_config):
    logger = _Logger()
    mmr = effective_mmr(
        "BTC-PERP",
        calculation_config,
        now=dt.datetime(2025, 1, 6, 22, 0, tzinfo=dt.timezone.utc),
        risk_info_provider=lambda symbol: {"maintenance_margin_rate": 0.03},
        logger=logger,
    )
    assert mmr == Decimal("0.03")


def test_effective_mmr_falls_back_to_schedule(calculation_config):
    logger = _Logger()
    mmr = effective_mmr(
        "BTC-PERP",
        calculation_config,
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


def _generate_marks_with_volatility(target_vol: float, n: int = 30) -> list[Decimal]:
    if n < 25:
        n = 25
    target = Decimal(str(target_vol))
    step = target / Decimal(str(math.sqrt(252.0)))
    price = Decimal("100")
    marks = [price]
    for idx in range(n - 1):
        direction = Decimal("1") if idx % 2 == 0 else Decimal("-1")
        price = max(Decimal("0.01"), price * (Decimal("1") + step * direction))
        marks.append(price)
    return marks


def test_circuit_breakers_progressive_actions():
    config = RiskConfig(
        enable_volatility_circuit_breaker=True,
        volatility_window_periods=20,
        circuit_breaker_cooldown_minutes=0,
        volatility_warning_threshold=0.05,
        volatility_reduce_only_threshold=0.08,
        volatility_kill_switch_threshold=0.11,
    )
    risk_manager = LiveRiskManager(config=config)

    symbol = "BTC-PERP"

    warning_marks = _generate_marks_with_volatility(0.055)
    outcome_warning = risk_manager.check_volatility_circuit_breaker(symbol, warning_marks)
    assert outcome_warning.triggered is True
    assert outcome_warning.action is CircuitBreakerAction.WARNING
    assert risk_manager.is_reduce_only_mode() is False
    assert risk_manager.config.kill_switch_enabled is False

    reduce_marks = _generate_marks_with_volatility(0.085)
    outcome_reduce = risk_manager.check_volatility_circuit_breaker(symbol, reduce_marks)
    assert outcome_reduce.triggered is True
    assert outcome_reduce.action is CircuitBreakerAction.REDUCE_ONLY
    assert risk_manager.is_reduce_only_mode() is True

    kill_marks = _generate_marks_with_volatility(0.12)
    outcome_kill = risk_manager.check_volatility_circuit_breaker(symbol, kill_marks)
    assert outcome_kill.triggered is True
    assert outcome_kill.action is CircuitBreakerAction.KILL_SWITCH
    assert risk_manager.config.kill_switch_enabled is True

    snapshot = risk_manager.circuit_breaker_state.get("volatility_circuit_breaker", symbol)
    assert snapshot is not None
    assert snapshot.last_action is CircuitBreakerAction.KILL_SWITCH


def test_market_impact_guard_allows_within_limit():
    config = RiskConfig(
        enable_market_impact_guard=True,
        max_market_impact_bps=Decimal("40"),
        max_position_pct_per_symbol=0.5,
    )
    store = _RecordingEventStore()

    def estimator(request: ImpactRequest) -> ImpactAssessment:
        return ImpactAssessment(
            symbol=request.symbol,
            side=request.side,
            quantity=request.quantity,
            estimated_impact_bps=Decimal("35"),
            slippage_cost=Decimal("12.5"),
            recommended_slicing=True,
            max_slice_size=Decimal("0.25"),
        )

    risk_manager = LiveRiskManager(config=config, event_store=store, impact_estimator=estimator)
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

    risk_manager.pre_trade_validate(
        symbol="BTC-PERP",
        side="buy",
        quantity=Decimal("0.5"),
        price=Decimal("50000"),
        product=product,
        equity=Decimal("100000"),
        current_positions={},
    )

    events = [
        m for m in store.metrics if m.get("metrics", {}).get("event_type") == "market_impact_guard"
    ]
    assert events
    assert events[-1]["metrics"]["status"] == "allowed"


def test_market_impact_guard_blocks_when_exceeding_limit():
    config = RiskConfig(
        enable_market_impact_guard=True,
        max_market_impact_bps=Decimal("25"),
        max_position_pct_per_symbol=0.5,
    )
    store = _RecordingEventStore()

    def estimator(request: ImpactRequest) -> ImpactAssessment:
        return ImpactAssessment(
            symbol=request.symbol,
            side=request.side,
            quantity=request.quantity,
            estimated_impact_bps=Decimal("42"),
            slippage_cost=Decimal("18.2"),
            recommended_slicing=False,
            max_slice_size=None,
        )

    risk_manager = LiveRiskManager(config=config, event_store=store, impact_estimator=estimator)

    with pytest.raises(ValidationError, match=r"impact"):
        risk_manager.pre_trade_validate(
            symbol="ETH-PERP",
            side="sell",
            quantity=Decimal("1.0"),
            price=Decimal("2500"),
            product=Product(
                symbol="ETH-PERP",
                base_asset="ETH",
                quote_asset="USD",
                market_type=MarketType.PERPETUAL,
                min_size=Decimal("0.001"),
                step_size=Decimal("0.001"),
                min_notional=Decimal("1"),
                price_increment=Decimal("0.01"),
            ),
            equity=Decimal("50000"),
            current_positions={},
        )

    events = [
        m for m in store.metrics if m.get("metrics", {}).get("event_type") == "market_impact_guard"
    ]
    assert events
    assert events[-1]["metrics"]["status"] == "blocked"


class _FakeEventStore:
    def __init__(self, events: list[dict]):
        self._events = events

    def tail(
        self, bot_id: str, limit: int = 2000, types: tuple[str, ...] | list[str] | None = None
    ):
        allowed_types = set(types or [])
        filtered = [
            event
            for event in self._events
            if event.get("bot_id") == bot_id
            and (not allowed_types or event.get("type") in allowed_types)
        ]
        return filtered[-limit:]


def test_risk_metrics_aggregator_summarizes_recent_window():
    now = dt.datetime(2025, 1, 7, 12, 0, tzinfo=dt.timezone.utc)
    events = [
        {  # Outside the 2-day lookback
            "bot_id": "risk_engine",
            "type": "metric",
            "timestamp": (now - dt.timedelta(days=3)).isoformat(),
            "equity": "9500",
            "total_notional": "4000",
            "exposure_pct": "0.42",
            "max_leverage": "1.8",
            "daily_pnl": "-25",
            "daily_pnl_pct": "-0.0026",
            "reduce_only": "false",
            "kill_switch": "false",
        },
        {
            "bot_id": "risk_engine",
            "type": "metric",
            "timestamp": (now - dt.timedelta(days=1)).isoformat(),
            "equity": "10200",
            "total_notional": "6100",
            "exposure_pct": "0.50",
            "max_leverage": "2.1",
            "daily_pnl": "180",
            "daily_pnl_pct": "0.017",
            "reduce_only": "false",
            "kill_switch": "false",
        },
        {
            "bot_id": "risk_engine",
            "type": "metric",
            "timestamp": (now - dt.timedelta(minutes=10)).isoformat(),
            "equity": "12000",
            "total_notional": "8500",
            "exposure_pct": "0.72",
            "max_leverage": "2.6",
            "daily_pnl": "-50",
            "daily_pnl_pct": "-0.0042",
            "reduce_only": "true",
            "kill_switch": "true",
        },
        {  # Irrelevant bot entry
            "bot_id": "execution",
            "type": "metric",
            "timestamp": now.isoformat(),
            "equity": "9999",
        },
    ]

    aggregator = RiskMetricsAggregator(_FakeEventStore(events), now=lambda: now)
    summary = aggregator.aggregate(window=dt.timedelta(days=2))

    assert summary.count == 2
    assert summary.first_timestamp and summary.first_timestamp.isoformat() == events[1]["timestamp"]
    assert summary.last_timestamp and summary.last_timestamp.isoformat() == events[2]["timestamp"]
    assert summary.latest and summary.latest.equity == Decimal("12000")
    assert summary.exposure_pct_max == Decimal("0.72")
    assert summary.exposure_pct_avg == Decimal("0.61")
    assert summary.total_notional_max == Decimal("8500")
    assert summary.leverage_max == Decimal("2.6")
    assert summary.equity_min == Decimal("10200")
    assert summary.equity_max == Decimal("12000")
    assert summary.daily_pnl_min == Decimal("-50")
    assert summary.daily_pnl_max == Decimal("180")
    assert summary.daily_pnl_pct_min == Decimal("-0.0042")
    assert summary.daily_pnl_pct_max == Decimal("0.017")
    assert summary.reduce_only_active is True
    assert summary.kill_switch_active is True


def test_risk_metrics_aggregator_empty_summary():
    aggregator = RiskMetricsAggregator(_FakeEventStore([]))
    summary = aggregator.aggregate(window=dt.timedelta(hours=1))
    assert summary.count == 0
    assert summary.latest is None
    assert summary.reduce_only_active is False
    assert summary.kill_switch_active is False
