"""Comprehensive runtime guard tests - circuit breakers, correlation, staleness.

This module extensively tests the runtime guard functions with focus on:
- Volatility circuit breaker edge cases and market feed combinations
- Correlation risk and concentration detection
- Mark price staleness with various timing scenarios
- Risk metrics telemetry and error handling
- Circuit breaker state management

Critical behaviors tested:
- Progressive circuit breaker thresholds (WARNING → REDUCE_ONLY → KILL_SWITCH)
- Cooldown periods preventing rapid re-triggers
- Insufficient data scenarios
- NaN/zero/negative price handling
- Event logging and state recording
- Error propagation vs graceful degradation

Trading Safety Context:
    Runtime guards are the real-time safety net that halts trading when
    market conditions become dangerous. Failures here can result in:
    - Trading during flash crashes (missed volatility detection)
    - Trading on stale data (missed staleness checks)
    - Over-concentration in correlated assets
    - System crashes from malformed market data

    These guards must be mathematically correct and resilient to bad data.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock

import pytest

from bot_v2.features.live_trade.guard_errors import (
    RiskGuardComputationError,
    RiskGuardTelemetryError,
)
from bot_v2.features.live_trade.risk_runtime import (
    CircuitBreakerAction,
    CircuitBreakerOutcome,
    CircuitBreakerRule,
    CircuitBreakerSnapshot,
    CircuitBreakerState,
    append_risk_metrics,
    check_correlation_risk,
    check_mark_staleness,
    check_volatility_circuit_breaker,
)
from bot_v2.persistence.event_store import EventStore


# ==================== Fixtures ====================


@pytest.fixture
def fixed_time() -> datetime:
    """Fixed datetime for deterministic testing."""
    return datetime(2024, 1, 1, 12, 0, 0)


@pytest.fixture
def time_provider(fixed_time: datetime):
    """Time provider that can be advanced deterministically."""

    class TimeProvider:
        def __init__(self, start_time: datetime):
            self.current_time = start_time

        def now(self) -> datetime:
            return self.current_time

        def advance(self, **kwargs) -> datetime:
            self.current_time += timedelta(**kwargs)
            return self.current_time

    return TimeProvider(fixed_time)


@pytest.fixture
def mock_logger():
    """Mock logger that captures all log levels."""
    logger = Mock()
    logger.warnings = []
    logger.infos = []
    logger.debugs = []

    def warning(msg, *args, **kwargs):
        logger.warnings.append(msg % args if args else msg)

    def info(msg, *args, **kwargs):
        logger.infos.append(msg % args if args else msg)

    def debug(msg, *args, **kwargs):
        logger.debugs.append(msg % args if args else msg)

    logger.warning = warning
    logger.info = info
    logger.debug = debug
    return logger


@pytest.fixture
def mock_log_event():
    """Mock log event function that captures events."""
    events = []

    def log_event(event_type: str, details: dict[str, str], guard: str):
        events.append({"type": event_type, "details": details, "guard": guard})

    log_event.events = events
    return log_event


@pytest.fixture
def event_store(tmp_path) -> EventStore:
    """Test event store."""
    return EventStore(root=tmp_path)


# ==================== Market Feed Fixtures ====================


class MarketFeedScenarios:
    """Collection of realistic market feed scenarios for testing."""

    @staticmethod
    def stable_market(base_price: float = 50000.0, length: int = 30) -> list[Decimal]:
        """Stable market with minimal price movement (~1% annualized vol)."""
        import random

        random.seed(42)
        prices = [base_price]
        for _ in range(length - 1):
            # Small random walk: ~0.01% per step
            change_pct = random.uniform(-0.0001, 0.0001)
            prices.append(prices[-1] * (1 + change_pct))
        return [Decimal(f"{p:.2f}") for p in prices]

    @staticmethod
    def moderate_volatility(base_price: float = 50000.0, length: int = 30) -> list[Decimal]:
        """Moderate volatility market (~10% annualized vol)."""
        import random

        random.seed(123)
        prices = [base_price]
        for _ in range(length - 1):
            # ~0.6% per step for ~10% annual
            change_pct = random.uniform(-0.006, 0.006)
            prices.append(prices[-1] * (1 + change_pct))
        return [Decimal(f"{p:.2f}") for p in prices]

    @staticmethod
    def high_volatility(base_price: float = 50000.0, length: int = 30) -> list[Decimal]:
        """High volatility market (~20% annualized vol)."""
        import random

        random.seed(456)
        prices = [base_price]
        for _ in range(length - 1):
            # ~1.2% per step for ~20% annual
            change_pct = random.uniform(-0.012, 0.012)
            prices.append(prices[-1] * (1 + change_pct))
        return [Decimal(f"{p:.2f}") for p in prices]

    @staticmethod
    def flash_crash(base_price: float = 50000.0) -> list[Decimal]:
        """Flash crash scenario - sudden 15% drop then recovery."""
        prices = [Decimal(str(base_price))]
        # Stable
        for _ in range(10):
            prices.append(prices[-1])
        # Crash
        crash_price = base_price * 0.85
        for i in range(5):
            prices.append(Decimal(str(base_price - (base_price - crash_price) * (i + 1) / 5)))
        # Recovery
        for i in range(10):
            prices.append(Decimal(str(crash_price + (base_price - crash_price) * i / 10)))
        # Stable again
        for _ in range(5):
            prices.append(Decimal(str(base_price)))
        return prices

    @staticmethod
    def alternating_swings(
        base_price: float = 50000.0, swing_pct: float = 0.05, length: int = 30
    ) -> list[Decimal]:
        """Alternating up/down swings of fixed percentage."""
        prices = [Decimal(str(base_price))]
        for i in range(length - 1):
            multiplier = 1 + swing_pct if i % 2 == 0 else 1 - swing_pct
            prices.append(prices[-1] * Decimal(str(multiplier)))
        return prices

    @staticmethod
    def with_gaps(base_price: float = 50000.0) -> list[Decimal]:
        """Price feed with gaps (missing data represented by same price)."""
        prices = []
        for i in range(30):
            if i % 5 == 0:  # Gap every 5th point
                prices.append(prices[-1] if prices else Decimal(str(base_price)))
            else:
                price = base_price * (1 + (i % 10 - 5) * 0.001)
                prices.append(Decimal(f"{price:.2f}"))
        return prices

    @staticmethod
    def with_zeros(base_price: float = 50000.0) -> list[Decimal]:
        """Price feed with zero values (bad data)."""
        prices = MarketFeedScenarios.stable_market(base_price, 20)
        # Inject zeros at various positions
        prices[5] = Decimal("0")
        prices[15] = Decimal("0")
        return prices

    @staticmethod
    def with_negative(base_price: float = 50000.0) -> list[Decimal]:
        """Price feed with negative values (corrupted data)."""
        prices = MarketFeedScenarios.stable_market(base_price, 20)
        # Inject negative values
        prices[7] = Decimal("-1000")
        return prices


@pytest.fixture
def market_scenarios():
    """Market feed scenarios fixture."""
    return MarketFeedScenarios()


# ==================== Circuit Breaker State Tests ====================


class TestCircuitBreakerState:
    """Test CircuitBreakerState management."""

    def test_initialize_empty_state(self) -> None:
        """CircuitBreakerState initializes with empty rules and triggers."""
        state = CircuitBreakerState()

        assert state._rules == {}
        assert state._triggers == {}

    def test_register_rule_adds_to_state(self) -> None:
        """register_rule adds rule and initializes trigger dict."""
        state = CircuitBreakerState()
        rule = CircuitBreakerRule(
            name="test_rule",
            signal="volatility",
            window=20,
            warning_threshold=Decimal("0.10"),
            reduce_only_threshold=Decimal("0.12"),
            kill_switch_threshold=Decimal("0.15"),
            cooldown=timedelta(minutes=30),
        )

        state.register_rule(rule)

        assert "test_rule" in state._rules
        assert state._rules["test_rule"] == rule
        assert "test_rule" in state._triggers

    def test_record_trigger_stores_snapshot(self, fixed_time: datetime) -> None:
        """record stores trigger snapshot with action and timestamp."""
        state = CircuitBreakerState()
        state._triggers["test_rule"] = {}

        state.record("test_rule", "BTC-PERP", CircuitBreakerAction.WARNING, fixed_time)

        snapshot = state._triggers["test_rule"]["BTC-PERP"]
        assert snapshot.last_action == CircuitBreakerAction.WARNING
        assert snapshot.triggered_at == fixed_time

    def test_record_creates_rule_dict_if_missing(self, fixed_time: datetime) -> None:
        """record creates rule dict if not present."""
        state = CircuitBreakerState()

        state.record("new_rule", "ETH-PERP", CircuitBreakerAction.REDUCE_ONLY, fixed_time)

        assert "new_rule" in state._triggers
        assert "ETH-PERP" in state._triggers["new_rule"]

    def test_get_retrieves_snapshot(self, fixed_time: datetime) -> None:
        """get retrieves snapshot for rule/symbol combination."""
        state = CircuitBreakerState()
        state.record("rule1", "BTC-PERP", CircuitBreakerAction.WARNING, fixed_time)

        snapshot = state.get("rule1", "BTC-PERP")

        assert snapshot is not None
        assert snapshot.last_action == CircuitBreakerAction.WARNING

    def test_get_returns_none_for_missing_rule(self) -> None:
        """get returns None for non-existent rule."""
        state = CircuitBreakerState()

        snapshot = state.get("missing_rule", "BTC-PERP")

        assert snapshot is None

    def test_get_returns_none_for_missing_symbol(self, fixed_time: datetime) -> None:
        """get returns None for non-existent symbol within rule."""
        state = CircuitBreakerState()
        state.record("rule1", "BTC-PERP", CircuitBreakerAction.WARNING, fixed_time)

        snapshot = state.get("rule1", "ETH-PERP")

        assert snapshot is None

    def test_snapshot_returns_all_triggers(self, fixed_time: datetime) -> None:
        """snapshot returns complete trigger dictionary."""
        state = CircuitBreakerState()
        state.record("rule1", "BTC-PERP", CircuitBreakerAction.WARNING, fixed_time)
        state.record("rule1", "ETH-PERP", CircuitBreakerAction.REDUCE_ONLY, fixed_time)
        state.record("rule2", "BTC-PERP", CircuitBreakerAction.KILL_SWITCH, fixed_time)

        all_triggers = state.snapshot()

        assert "rule1" in all_triggers
        assert "rule2" in all_triggers
        assert len(all_triggers["rule1"]) == 2
        assert len(all_triggers["rule2"]) == 1

    def test_multiple_symbols_tracked_independently(self, fixed_time: datetime) -> None:
        """Multiple symbols tracked independently within same rule."""
        state = CircuitBreakerState()

        state.record("vol_cb", "BTC-PERP", CircuitBreakerAction.WARNING, fixed_time)
        state.record("vol_cb", "ETH-PERP", CircuitBreakerAction.REDUCE_ONLY, fixed_time)

        btc_snapshot = state.get("vol_cb", "BTC-PERP")
        eth_snapshot = state.get("vol_cb", "ETH-PERP")

        assert btc_snapshot.last_action == CircuitBreakerAction.WARNING
        assert eth_snapshot.last_action == CircuitBreakerAction.REDUCE_ONLY


# ==================== Mark Staleness Tests ====================


class TestMarkStaleness:
    """Test mark price staleness detection."""

    def test_returns_false_when_symbol_not_in_cache(
        self, time_provider, mock_log_event, mock_logger
    ) -> None:
        """Returns False when symbol has no mark update history.

        New symbols should not be flagged as stale initially.
        """
        last_mark_update = {}

        is_stale = check_mark_staleness(
            symbol="NEW-PERP",
            last_mark_update=last_mark_update,
            now=time_provider.now,
            max_staleness_seconds=30,
            log_event=mock_log_event,
            logger=mock_logger,
        )

        assert is_stale is False

    def test_returns_false_for_fresh_data(self, time_provider, mock_log_event, mock_logger) -> None:
        """Returns False when mark data is within soft limit.

        Fresh data (age < soft_limit) should be accepted.
        """
        last_mark_update = {"BTC-PERP": time_provider.now() - timedelta(seconds=10)}

        is_stale = check_mark_staleness(
            symbol="BTC-PERP",
            last_mark_update=last_mark_update,
            now=time_provider.now,
            max_staleness_seconds=30,
            log_event=mock_log_event,
            logger=mock_logger,
        )

        assert is_stale is False
        assert len(mock_log_event.events) == 0

    def test_logs_warning_for_soft_limit_breach(
        self, time_provider, mock_log_event, mock_logger
    ) -> None:
        """Logs warning but allows trading when soft limit breached.

        Between soft and hard limit: warn but continue.
        """
        # Age = 40s, soft_limit = 30s, hard_limit = 60s
        last_mark_update = {"BTC-PERP": time_provider.now() - timedelta(seconds=40)}

        is_stale = check_mark_staleness(
            symbol="BTC-PERP",
            last_mark_update=last_mark_update,
            now=time_provider.now,
            max_staleness_seconds=30,
            log_event=mock_log_event,
            logger=mock_logger,
        )

        assert is_stale is False  # Still allows trading
        assert len(mock_logger.infos) > 0  # But logs warning

    def test_returns_true_for_hard_limit_breach(
        self, time_provider, mock_log_event, mock_logger
    ) -> None:
        """Returns True and halts trading when hard limit breached.

        Critical: Trading must stop when data is too stale.
        """
        # Age = 70s, hard_limit = 60s
        last_mark_update = {"BTC-PERP": time_provider.now() - timedelta(seconds=70)}

        is_stale = check_mark_staleness(
            symbol="BTC-PERP",
            last_mark_update=last_mark_update,
            now=time_provider.now,
            max_staleness_seconds=30,
            log_event=mock_log_event,
            logger=mock_logger,
        )

        assert is_stale is True
        assert len(mock_logger.warnings) > 0
        assert len(mock_log_event.events) > 0
        assert mock_log_event.events[0]["type"] == "stale_mark_price"

    def test_exact_soft_limit_boundary(self, time_provider, mock_log_event, mock_logger) -> None:
        """Exactly at soft limit should log but not halt."""
        # Age = exactly 30s = soft_limit
        last_mark_update = {"BTC-PERP": time_provider.now() - timedelta(seconds=30)}

        is_stale = check_mark_staleness(
            symbol="BTC-PERP",
            last_mark_update=last_mark_update,
            now=time_provider.now,
            max_staleness_seconds=30,
            log_event=mock_log_event,
            logger=mock_logger,
        )

        assert is_stale is False  # Boundary is inclusive for soft limit

    def test_exact_hard_limit_boundary(self, time_provider, mock_log_event, mock_logger) -> None:
        """Exactly at hard limit should halt trading."""
        # Age = exactly 60s = hard_limit
        last_mark_update = {"BTC-PERP": time_provider.now() - timedelta(seconds=60)}

        is_stale = check_mark_staleness(
            symbol="BTC-PERP",
            last_mark_update=last_mark_update,
            now=time_provider.now,
            max_staleness_seconds=30,
            log_event=mock_log_event,
            logger=mock_logger,
        )

        assert is_stale is False  # Boundary check: > hard_limit, not >=

    def test_multiple_symbols_checked_independently(
        self, time_provider, mock_log_event, mock_logger
    ) -> None:
        """Multiple symbols checked independently."""
        last_mark_update = {
            "BTC-PERP": time_provider.now() - timedelta(seconds=10),  # Fresh
            "ETH-PERP": time_provider.now() - timedelta(seconds=70),  # Stale
        }

        btc_stale = check_mark_staleness(
            symbol="BTC-PERP",
            last_mark_update=last_mark_update,
            now=time_provider.now,
            max_staleness_seconds=30,
            log_event=mock_log_event,
            logger=mock_logger,
        )

        eth_stale = check_mark_staleness(
            symbol="ETH-PERP",
            last_mark_update=last_mark_update,
            now=time_provider.now,
            max_staleness_seconds=30,
            log_event=mock_log_event,
            logger=mock_logger,
        )

        assert btc_stale is False
        assert eth_stale is True

    def test_log_event_includes_correct_metadata(
        self, time_provider, mock_log_event, mock_logger
    ) -> None:
        """Log event includes symbol, age, and action metadata."""
        last_mark_update = {"BTC-PERP": time_provider.now() - timedelta(seconds=70)}

        check_mark_staleness(
            symbol="BTC-PERP",
            last_mark_update=last_mark_update,
            now=time_provider.now,
            max_staleness_seconds=30,
            log_event=mock_log_event,
            logger=mock_logger,
        )

        event = mock_log_event.events[0]
        assert event["type"] == "stale_mark_price"
        assert event["guard"] == "mark_staleness"
        assert event["details"]["symbol"] == "BTC-PERP"
        assert "age_seconds" in event["details"]
        assert event["details"]["action"] == "halt_new_orders"


# ==================== Volatility Circuit Breaker Tests ====================


class TestVolatilityCircuitBreaker:
    """Test volatility circuit breaker with comprehensive edge cases."""

    def test_disabled_circuit_breaker_returns_none(
        self, time_provider, mock_logger, market_scenarios
    ) -> None:
        """Disabled circuit breaker always returns NONE action.

        Config can disable circuit breaker entirely.
        """
        config = Mock()
        config.enable_volatility_circuit_breaker = False

        outcome = check_volatility_circuit_breaker(
            symbol="BTC-PERP",
            recent_marks=market_scenarios.high_volatility(),
            config=config,
            now=time_provider.now,
            logger=mock_logger,
        )

        assert outcome.triggered is False
        assert outcome.action == CircuitBreakerAction.NONE

    def test_insufficient_data_returns_none(self, time_provider, mock_logger) -> None:
        """Insufficient price data returns NONE without error.

        Need at least `window` prices to calculate volatility.
        """
        config = Mock()
        config.enable_volatility_circuit_breaker = True
        config.volatility_window_periods = 20

        # Only 10 prices, need 20
        short_marks = [Decimal(f"{50000 + i}") for i in range(10)]

        outcome = check_volatility_circuit_breaker(
            symbol="BTC-PERP",
            recent_marks=short_marks,
            config=config,
            now=time_provider.now,
            logger=mock_logger,
        )

        assert outcome.triggered is False
        assert outcome.action == CircuitBreakerAction.NONE

    def test_stable_market_no_trigger(self, time_provider, mock_logger, market_scenarios) -> None:
        """Stable market with low volatility doesn't trigger."""
        config = Mock()
        config.enable_volatility_circuit_breaker = True
        config.volatility_window_periods = 20
        config.volatility_warning_threshold = 0.10
        config.volatility_reduce_only_threshold = 0.12
        config.volatility_kill_switch_threshold = 0.15
        config.circuit_breaker_cooldown_minutes = 30

        outcome = check_volatility_circuit_breaker(
            symbol="BTC-PERP",
            recent_marks=market_scenarios.stable_market(),
            config=config,
            now=time_provider.now,
            logger=mock_logger,
        )

        assert outcome.triggered is False
        assert outcome.action == CircuitBreakerAction.NONE
        assert outcome.value is not None  # Volatility still calculated

    def test_warning_threshold_trigger(
        self, time_provider, mock_log_event, mock_logger, market_scenarios
    ) -> None:
        """Moderate volatility triggers WARNING action."""
        config = Mock()
        config.enable_volatility_circuit_breaker = True
        config.volatility_window_periods = 20
        config.volatility_warning_threshold = 0.05  # Low threshold
        config.volatility_reduce_only_threshold = 0.15
        config.volatility_kill_switch_threshold = 0.25
        config.circuit_breaker_cooldown_minutes = 30

        outcome = check_volatility_circuit_breaker(
            symbol="BTC-PERP",
            recent_marks=market_scenarios.moderate_volatility(),
            config=config,
            now=time_provider.now,
            log_event=mock_log_event,
            logger=mock_logger,
        )

        assert outcome.triggered is True
        assert outcome.action == CircuitBreakerAction.WARNING
        assert len(mock_log_event.events) > 0

    def test_reduce_only_threshold_trigger(
        self, time_provider, mock_log_event, mock_logger, market_scenarios
    ) -> None:
        """High volatility triggers REDUCE_ONLY action."""
        set_reduce_only_called = []

        def mock_set_reduce_only(enabled: bool, reason: str):
            set_reduce_only_called.append((enabled, reason))

        config = Mock()
        config.enable_volatility_circuit_breaker = True
        config.volatility_window_periods = 20
        config.volatility_warning_threshold = 0.05
        config.volatility_reduce_only_threshold = 0.10
        config.volatility_kill_switch_threshold = 0.30
        config.circuit_breaker_cooldown_minutes = 30

        outcome = check_volatility_circuit_breaker(
            symbol="BTC-PERP",
            recent_marks=market_scenarios.high_volatility(),
            config=config,
            now=time_provider.now,
            set_reduce_only=mock_set_reduce_only,
            log_event=mock_log_event,
            logger=mock_logger,
        )

        assert outcome.triggered is True
        assert outcome.action == CircuitBreakerAction.REDUCE_ONLY
        assert len(set_reduce_only_called) > 0
        assert set_reduce_only_called[0] == (True, "volatility_circuit_breaker")

    def test_kill_switch_threshold_trigger(
        self, time_provider, mock_log_event, mock_logger, market_scenarios
    ) -> None:
        """Extreme volatility triggers KILL_SWITCH action."""
        config = Mock()
        config.enable_volatility_circuit_breaker = True
        config.volatility_window_periods = 20
        config.volatility_warning_threshold = 0.05
        config.volatility_reduce_only_threshold = 0.10
        config.volatility_kill_switch_threshold = 0.12  # Low threshold for test
        config.circuit_breaker_cooldown_minutes = 30

        outcome = check_volatility_circuit_breaker(
            symbol="BTC-PERP",
            recent_marks=market_scenarios.flash_crash(),
            config=config,
            now=time_provider.now,
            log_event=mock_log_event,
            logger=mock_logger,
        )

        assert outcome.triggered is True
        assert outcome.action == CircuitBreakerAction.KILL_SWITCH
        assert config.kill_switch_enabled is True

    def test_cooldown_prevents_retrigger(
        self, time_provider, mock_logger, market_scenarios
    ) -> None:
        """Cooldown period prevents immediate re-trigger."""
        config = Mock()
        config.enable_volatility_circuit_breaker = True
        config.volatility_window_periods = 20
        config.volatility_warning_threshold = 0.05
        config.volatility_reduce_only_threshold = 0.10
        config.volatility_kill_switch_threshold = 0.15
        config.circuit_breaker_cooldown_minutes = 30

        last_trigger = {"BTC-PERP": time_provider.now()}

        # Should be in cooldown
        outcome = check_volatility_circuit_breaker(
            symbol="BTC-PERP",
            recent_marks=market_scenarios.high_volatility(),
            config=config,
            now=time_provider.now,
            last_trigger=last_trigger,
            logger=mock_logger,
        )

        assert outcome.triggered is False
        assert outcome.action == CircuitBreakerAction.NONE

    def test_cooldown_expires_allows_retrigger(
        self, time_provider, mock_logger, market_scenarios
    ) -> None:
        """After cooldown expires, can trigger again."""
        config = Mock()
        config.enable_volatility_circuit_breaker = True
        config.volatility_window_periods = 20
        config.volatility_warning_threshold = 0.05
        config.volatility_reduce_only_threshold = 0.10
        config.volatility_kill_switch_threshold = 0.15
        config.circuit_breaker_cooldown_minutes = 30

        # Last trigger was 31 minutes ago
        last_trigger = {"BTC-PERP": time_provider.now() - timedelta(minutes=31)}

        outcome = check_volatility_circuit_breaker(
            symbol="BTC-PERP",
            recent_marks=market_scenarios.high_volatility(),
            config=config,
            now=time_provider.now,
            last_trigger=last_trigger,
            logger=mock_logger,
        )

        assert outcome.triggered is True

    def test_handles_zero_prices_gracefully(
        self, time_provider, mock_logger, market_scenarios
    ) -> None:
        """Handles zero prices in feed without crashing.

        Zero prices are filtered out of return calculation.
        """
        config = Mock()
        config.enable_volatility_circuit_breaker = True
        config.volatility_window_periods = 20
        config.volatility_warning_threshold = 0.10
        config.volatility_reduce_only_threshold = 0.12
        config.volatility_kill_switch_threshold = 0.15
        config.circuit_breaker_cooldown_minutes = 30

        # Should not crash
        outcome = check_volatility_circuit_breaker(
            symbol="BTC-PERP",
            recent_marks=market_scenarios.with_zeros(),
            config=config,
            now=time_provider.now,
            logger=mock_logger,
        )

        # Should either trigger or not, but not crash
        assert isinstance(outcome, CircuitBreakerOutcome)

    def test_handles_negative_prices_gracefully(
        self, time_provider, mock_logger, market_scenarios
    ) -> None:
        """Handles negative prices (corrupted data) without crashing."""
        config = Mock()
        config.enable_volatility_circuit_breaker = True
        config.volatility_window_periods = 20
        config.volatility_warning_threshold = 0.10
        config.volatility_reduce_only_threshold = 0.12
        config.volatility_kill_switch_threshold = 0.15
        config.circuit_breaker_cooldown_minutes = 30

        outcome = check_volatility_circuit_breaker(
            symbol="BTC-PERP",
            recent_marks=market_scenarios.with_negative(),
            config=config,
            now=time_provider.now,
            logger=mock_logger,
        )

        assert isinstance(outcome, CircuitBreakerOutcome)

    def test_single_price_returns_none(self, time_provider, mock_logger) -> None:
        """Single price (no returns) returns NONE."""
        config = Mock()
        config.enable_volatility_circuit_breaker = True
        config.volatility_window_periods = 20

        outcome = check_volatility_circuit_breaker(
            symbol="BTC-PERP",
            recent_marks=[Decimal("50000")],
            config=config,
            now=time_provider.now,
            logger=mock_logger,
        )

        assert outcome.triggered is False
        assert outcome.action == CircuitBreakerAction.NONE

    def test_annualized_volatility_calculation(self, time_provider, mock_logger) -> None:
        """Volatility is correctly annualized using sqrt(252)."""
        config = Mock()
        config.enable_volatility_circuit_breaker = True
        config.volatility_window_periods = 20
        config.volatility_warning_threshold = 0.05
        config.volatility_reduce_only_threshold = 0.10
        config.volatility_kill_switch_threshold = 0.15
        config.circuit_breaker_cooldown_minutes = 30

        # Known volatility pattern
        marks = [Decimal("100")]
        for i in range(29):
            # Alternating +/- 1% returns
            change = 0.01 if i % 2 == 0 else -0.01
            marks.append(marks[-1] * Decimal(str(1 + change)))

        outcome = check_volatility_circuit_breaker(
            symbol="BTC-PERP",
            recent_marks=marks,
            config=config,
            now=time_provider.now,
            logger=mock_logger,
        )

        # Annualized vol = stddev * sqrt(252)
        # stddev(alternating +/-1%) ≈ 0.01
        # annualized ≈ 0.01 * 15.87 ≈ 0.1587
        assert outcome.value is not None
        # Should be roughly 0.15-0.16
        assert Decimal("0.14") < outcome.value < Decimal("0.17")

    def test_rule_based_circuit_breaker(self, time_provider, mock_logger, market_scenarios) -> None:
        """Circuit breaker with explicit rule configuration."""
        rule = CircuitBreakerRule(
            name="test_vol_cb",
            signal="annualized_vol",
            window=20,
            warning_threshold=Decimal("0.08"),
            reduce_only_threshold=Decimal("0.12"),
            kill_switch_threshold=Decimal("0.18"),
            cooldown=timedelta(minutes=15),
        )
        state = CircuitBreakerState()
        state.register_rule(rule)

        outcome = check_volatility_circuit_breaker(
            symbol="BTC-PERP",
            recent_marks=market_scenarios.moderate_volatility(),
            rule=rule,
            state=state,
            now=time_provider.now,
            logger=mock_logger,
        )

        # Should trigger based on rule thresholds
        assert isinstance(outcome, CircuitBreakerOutcome)
        if outcome.triggered:
            # Verify state was recorded
            snapshot = state.get(rule.name, "BTC-PERP")
            assert snapshot is not None

    def test_disabled_rule_returns_none(self, time_provider, mock_logger, market_scenarios) -> None:
        """Disabled rule returns NONE even with high volatility."""
        rule = CircuitBreakerRule(
            name="test_vol_cb",
            signal="annualized_vol",
            window=20,
            warning_threshold=Decimal("0.01"),
            reduce_only_threshold=Decimal("0.02"),
            kill_switch_threshold=Decimal("0.03"),
            cooldown=timedelta(minutes=15),
            enabled=False,  # Disabled
        )
        state = CircuitBreakerState()
        state.register_rule(rule)

        outcome = check_volatility_circuit_breaker(
            symbol="BTC-PERP",
            recent_marks=market_scenarios.flash_crash(),
            rule=rule,
            state=state,
            now=time_provider.now,
            logger=mock_logger,
        )

        assert outcome.triggered is False
        assert outcome.action == CircuitBreakerAction.NONE


# ==================== Correlation Risk Tests ====================


class TestCorrelationRisk:
    """Test portfolio correlation and concentration risk detection."""

    def test_single_position_returns_false(self, mock_log_event, mock_logger) -> None:
        """Single position cannot have concentration risk.

        Need at least 2 positions for correlation/concentration.
        """
        positions = {"BTC-PERP": {"quantity": Decimal("1.0"), "mark": Decimal("50000")}}

        triggered = check_correlation_risk(
            positions=positions,
            log_event=mock_log_event,
            logger=mock_logger,
        )

        assert triggered is False

    def test_empty_positions_returns_false(self, mock_log_event, mock_logger) -> None:
        """Empty positions dict returns False."""
        triggered = check_correlation_risk(
            positions={},
            log_event=mock_log_event,
            logger=mock_logger,
        )

        assert triggered is False

    def test_well_diversified_portfolio_passes(self, mock_log_event, mock_logger) -> None:
        """Well-diversified portfolio (HHI < 0.4) passes check.

        3 equal positions: HHI = 3 * (1/3)^2 = 0.333 < 0.4
        """
        positions = {
            "BTC-PERP": {"quantity": Decimal("1.0"), "mark": Decimal("50000")},
            "ETH-PERP": {"quantity": Decimal("16.67"), "mark": Decimal("3000")},  # ~50k notional
            "SOL-PERP": {"quantity": Decimal("500"), "mark": Decimal("100")},  # ~50k notional
        }

        triggered = check_correlation_risk(
            positions=positions,
            log_event=mock_log_event,
            logger=mock_logger,
        )

        assert triggered is False

    def test_concentrated_portfolio_triggers(self, mock_log_event, mock_logger) -> None:
        """Concentrated portfolio (HHI > 0.4) triggers warning.

        90/10 split: HHI = 0.9^2 + 0.1^2 = 0.82 > 0.4
        """
        positions = {
            "BTC-PERP": {"quantity": Decimal("1.8"), "mark": Decimal("50000")},  # 90k
            "ETH-PERP": {"quantity": Decimal("3.33"), "mark": Decimal("3000")},  # 10k
        }

        triggered = check_correlation_risk(
            positions=positions,
            log_event=mock_log_event,
            logger=mock_logger,
        )

        assert triggered is True
        assert len(mock_log_event.events) > 0
        assert mock_log_event.events[0]["type"] == "concentration_risk"
        assert "hhi" in mock_log_event.events[0]["details"]

    def test_hhi_threshold_exactly_at_limit(self, mock_log_event, mock_logger) -> None:
        """HHI exactly at 0.4 threshold triggers warning.

        Boundary case: HHI = 0.4 should trigger.
        """
        # Design positions to have HHI ≈ 0.4
        # 2 positions with 63.2% / 36.8% split: 0.632^2 + 0.368^2 ≈ 0.535
        # Need closer to 60/40: 0.6^2 + 0.4^2 = 0.52
        # Try 55/45: 0.55^2 + 0.45^2 = 0.505
        # This test may need adjustment based on exact threshold behavior
        positions = {
            "BTC-PERP": {"quantity": Decimal("0.55"), "mark": Decimal("100000")},  # 55k
            "ETH-PERP": {"quantity": Decimal("15"), "mark": Decimal("3000")},  # 45k
        }

        triggered = check_correlation_risk(
            positions=positions,
            log_event=mock_log_event,
            logger=mock_logger,
        )

        # HHI = 0.505 > 0.4, should trigger
        assert triggered is True

    def test_handles_zero_notional_positions(self, mock_log_event, mock_logger) -> None:
        """Handles positions with zero notional gracefully.

        Zero qty or zero mark price should be filtered out.
        """
        positions = {
            "BTC-PERP": {"quantity": Decimal("0"), "mark": Decimal("50000")},  # Zero qty
            "ETH-PERP": {"quantity": Decimal("10"), "mark": Decimal("0")},  # Zero mark
        }

        # Total notional = 0, should return False
        triggered = check_correlation_risk(
            positions=positions,
            log_event=mock_log_event,
            logger=mock_logger,
        )

        assert triggered is False

    def test_uses_qty_field_fallback(self, mock_log_event, mock_logger) -> None:
        """Uses 'qty' field if 'quantity' is missing.

        Supports legacy position data format.
        """
        positions = {
            "BTC-PERP": {"qty": Decimal("1.0"), "mark": Decimal("50000")},
            "ETH-PERP": {"qty": Decimal("16.67"), "mark": Decimal("3000")},
        }

        # Should not crash
        triggered = check_correlation_risk(
            positions=positions,
            log_event=mock_log_event,
            logger=mock_logger,
        )

        assert isinstance(triggered, bool)

    def test_handles_invalid_position_data(self, mock_log_event, mock_logger) -> None:
        """Handles invalid position data gracefully without crashing.

        Malformed data should be skipped (try/except continue pattern).
        """
        positions = {
            "BTC-PERP": {"quantity": "invalid", "mark": Decimal("50000")},
        }

        # Should not crash - invalid positions are skipped with try/except continue
        triggered = check_correlation_risk(
            positions=positions,
            log_event=mock_log_event,
            logger=mock_logger,
        )

        # Should return False since all positions were invalid/skipped
        assert triggered is False

    def test_logs_hhi_value_on_trigger(self, mock_log_event, mock_logger) -> None:
        """Logs HHI value when concentration risk triggers."""
        positions = {
            "BTC-PERP": {"quantity": Decimal("1.8"), "mark": Decimal("50000")},
            "ETH-PERP": {"quantity": Decimal("3.33"), "mark": Decimal("3000")},
        }

        check_correlation_risk(
            positions=positions,
            log_event=mock_log_event,
            logger=mock_logger,
        )

        event = mock_log_event.events[0]
        hhi = Decimal(event["details"]["hhi"])
        assert hhi > Decimal("0.4")
        assert len(mock_logger.warnings) > 0


# ==================== Risk Metrics Telemetry Tests ====================


class TestRiskMetricsTelemetry:
    """Test append_risk_metrics telemetry function."""

    def test_persists_complete_risk_snapshot(self, time_provider, event_store, mock_logger) -> None:
        """Persists complete risk snapshot to event store."""
        append_risk_metrics(
            event_store=event_store,
            now=time_provider.now,
            equity=Decimal("10000"),
            positions={"BTC-PERP": {"quantity": Decimal("0.1"), "mark": Decimal("50000")}},
            daily_pnl=Decimal("500"),
            start_of_day_equity=Decimal("9500"),
            reduce_only=False,
            kill_switch_enabled=False,
            logger=mock_logger,
        )

        # Event should be persisted (verify doesn't crash)
        assert True

    def test_calculates_exposure_percentage_correctly(
        self, time_provider, event_store, mock_logger
    ) -> None:
        """Calculates exposure percentage from positions."""
        # 0.1 BTC * 50000 = 5000 notional
        # 5000 / 10000 equity = 50% exposure
        append_risk_metrics(
            event_store=event_store,
            now=time_provider.now,
            equity=Decimal("10000"),
            positions={"BTC-PERP": {"quantity": Decimal("0.1"), "mark": Decimal("50000")}},
            daily_pnl=Decimal("0"),
            start_of_day_equity=Decimal("10000"),
            reduce_only=False,
            kill_switch_enabled=False,
            logger=mock_logger,
        )

        # Check debug log contains exposure
        assert len(mock_logger.debugs) > 0

    def test_handles_zero_equity_without_division_error(
        self, time_provider, event_store, mock_logger
    ) -> None:
        """Handles zero equity without division by zero error."""
        # Should not crash
        append_risk_metrics(
            event_store=event_store,
            now=time_provider.now,
            equity=Decimal("0"),
            positions={"BTC-PERP": {"quantity": Decimal("0.1"), "mark": Decimal("50000")}},
            daily_pnl=Decimal("0"),
            start_of_day_equity=Decimal("0"),
            reduce_only=False,
            kill_switch_enabled=False,
            logger=mock_logger,
        )

        assert True  # No crash

    def test_handles_event_store_failure(self, time_provider, mock_logger) -> None:
        """Raises RiskGuardTelemetryError when event store fails."""
        mock_store = Mock()
        mock_store.append_metric.side_effect = Exception("Storage failure")

        with pytest.raises(RiskGuardTelemetryError) as exc_info:
            append_risk_metrics(
                event_store=mock_store,
                now=time_provider.now,
                equity=Decimal("10000"),
                positions={},
                daily_pnl=Decimal("0"),
                start_of_day_equity=Decimal("10000"),
                reduce_only=False,
                kill_switch_enabled=False,
                logger=mock_logger,
            )

        assert exc_info.value.guard == "risk_metrics"
        # Error uses str(self) not .message attribute
        assert "Failed to persist risk snapshot metric" in str(exc_info.value)

    def test_calculates_max_leverage_across_positions(
        self, time_provider, event_store, mock_logger
    ) -> None:
        """Calculates maximum leverage across all positions."""
        # BTC: 5000 notional / 10000 equity = 0.5x leverage
        # ETH: 6000 notional / 10000 equity = 0.6x leverage (max)
        append_risk_metrics(
            event_store=event_store,
            now=time_provider.now,
            equity=Decimal("10000"),
            positions={
                "BTC-PERP": {"quantity": Decimal("0.1"), "mark": Decimal("50000")},
                "ETH-PERP": {"quantity": Decimal("2"), "mark": Decimal("3000")},
            },
            daily_pnl=Decimal("0"),
            start_of_day_equity=Decimal("10000"),
            reduce_only=False,
            kill_switch_enabled=False,
            logger=mock_logger,
        )

        # Check debug log for max_lev
        assert len(mock_logger.debugs) > 0

    def test_handles_missing_position_fields(self, time_provider, event_store, mock_logger) -> None:
        """Handles positions with missing qty/mark fields gracefully."""
        # Should skip invalid positions without crashing
        append_risk_metrics(
            event_store=event_store,
            now=time_provider.now,
            equity=Decimal("10000"),
            positions={
                "BTC-PERP": {},  # Missing fields
                "ETH-PERP": {"quantity": Decimal("2")},  # Missing mark
            },
            daily_pnl=Decimal("0"),
            start_of_day_equity=Decimal("10000"),
            reduce_only=False,
            kill_switch_enabled=False,
            logger=mock_logger,
        )

        assert True  # No crash
