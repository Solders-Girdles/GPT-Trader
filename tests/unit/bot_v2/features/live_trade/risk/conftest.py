"""
Comprehensive test infrastructure for risk management subsystem.

Provides reusable fixtures and utilities for testing all risk components
including pre-trade validation, runtime monitoring, position sizing, and
emergency scenarios.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any
from unittest.mock import Mock

import pytest

from bot_v2.config.live_trade_config import RiskConfig
from bot_v2.features.brokerages.core.interfaces import MarketType, Product
from bot_v2.features.live_trade.risk_calculations import effective_mmr, effective_symbol_leverage_cap
from bot_v2.persistence.event_store import EventStore
from bot_v2.orchestration.runtime_settings import RuntimeSettings
from pathlib import Path


# =============================================================================
# SYNTHETIC POSITION AND BALANCE FIXTURES
# =============================================================================

@dataclass(frozen=True)
class SyntheticPosition:
    """Synthetic position for testing."""
    symbol: str
    side: str  # "long" or "short"
    quantity: Decimal
    entry_price: Decimal
    mark_price: Decimal
    unrealized_pnl: Decimal

    @classmethod
    def create_long(cls, symbol: str, quantity: Decimal, entry_price: Decimal,
                    mark_price: Decimal) -> SyntheticPosition:
        """Create a long position."""
        unrealized_pnl = (mark_price - entry_price) * quantity
        return cls(symbol=symbol, side="long", quantity=quantity,
                   entry_price=entry_price, mark_price=mark_price,
                   unrealized_pnl=unrealized_pnl)

    @classmethod
    def create_short(cls, symbol: str, quantity: Decimal, entry_price: Decimal,
                     mark_price: Decimal) -> SyntheticPosition:
        """Create a short position."""
        unrealized_pnl = (entry_price - mark_price) * quantity
        return cls(symbol=symbol, side="short", quantity=quantity,
                   entry_price=entry_price, mark_price=mark_price,
                   unrealized_pnl=unrealized_pnl)


@dataclass(frozen=True)
class SyntheticPortfolio:
    """Synthetic portfolio state for testing."""
    positions: dict[str, SyntheticPosition]
    cash_balance: Decimal
    total_equity: Decimal
    available_margin: Decimal
    maintenance_margin: Decimal

    @classmethod
    def create_empty(cls, equity: Decimal = Decimal("10000")) -> SyntheticPortfolio:
        """Create empty portfolio."""
        return cls(
            positions={},
            cash_balance=equity,
            total_equity=equity,
            available_margin=equity * Decimal("0.9"),  # 90% available
            maintenance_margin=Decimal("0")
        )

    @classmethod
    def with_positions(cls, positions: list[SyntheticPosition],
                      equity: Decimal) -> SyntheticPortfolio:
        """Create portfolio with positions."""
        total_unrealized = sum(pos.unrealized_pnl for pos in positions)
        cash_balance = equity - total_unrealized

        # Simple margin calculation for testing
        total_notional = sum(pos.quantity * pos.mark_price for pos in positions)
        maintenance_margin = total_notional * Decimal("0.1")  # 10% MMR
        available_margin = equity - maintenance_margin

        return cls(
            positions={pos.symbol: pos for pos in positions},
            cash_balance=cash_balance,
            total_equity=equity,
            available_margin=max(available_margin, Decimal("0")),
            maintenance_margin=maintenance_margin
        )


@pytest.fixture
def synthetic_btc_position() -> SyntheticPosition:
    """Standard BTC long position for testing."""
    return SyntheticPosition.create_long(
        symbol="BTC-USD",
        quantity=Decimal("0.5"),
        entry_price=Decimal("50000"),
        mark_price=Decimal("52000")
    )


@pytest.fixture
def synthetic_eth_position() -> SyntheticPosition:
    """Standard ETH short position for testing."""
    return SyntheticPosition.create_short(
        symbol="ETH-USD",
        quantity=Decimal("10"),
        entry_price=Decimal("3000"),
        mark_price=Decimal("2900")
    )


@pytest.fixture
def synthetic_portfolio_empty() -> SyntheticPortfolio:
    """Empty portfolio for testing."""
    return SyntheticPortfolio.create_empty(Decimal("10000"))


@pytest.fixture
def synthetic_portfolio_with_positions(synthetic_btc_position, synthetic_eth_position) -> SyntheticPortfolio:
    """Portfolio with multiple positions for testing."""
    return SyntheticPortfolio.with_positions(
        positions=[synthetic_btc_position, synthetic_eth_position],
        equity=Decimal("15000")
    )


# =============================================================================
# MARKET SCENARIO FIXTURES
# =============================================================================

@dataclass(frozen=True)
class MarketScenario:
    """Market scenario for testing risk responses."""
    symbol: str
    current_price: Decimal
    price_volatility: float  # Annualized volatility
    price_change_pct: float  # Recent price change
    timestamp: datetime
    liquidity_depth: dict[str, Decimal]  # bid/ask depths

    @classmethod
    def normal_market(cls, symbol: str, price: Decimal) -> MarketScenario:
        """Normal market conditions."""
        return cls(
            symbol=symbol,
            current_price=price,
            price_volatility=0.25,  # 25% annual vol
            price_change_pct=0.01,  # 1% change
            timestamp=datetime.now(timezone.utc),
            liquidity_depth={
                "bid_5": Decimal("100"),  # $100K at 5 bps
                "ask_5": Decimal("100")
            }
        )

    @classmethod
    def volatile_market(cls, symbol: str, price: Decimal) -> MarketScenario:
        """High volatility market conditions."""
        return cls(
            symbol=symbol,
            current_price=price,
            price_volatility=0.8,   # 80% annual vol
            price_change_pct=-0.15,  # -15% drop
            timestamp=datetime.now(timezone.utc),
            liquidity_depth={
                "bid_5": Decimal("20"),   # Thin liquidity
                "ask_5": Decimal("20")
            }
        )

    @classmethod
    def illiquid_market(cls, symbol: str, price: Decimal) -> MarketScenario:
        """Illiquid market conditions."""
        return cls(
            symbol=symbol,
            current_price=price,
            price_volatility=0.3,
            price_change_pct=0.05,
            timestamp=datetime.now(timezone.utc),
            liquidity_depth={
                "bid_5": Decimal("5"),    # Very thin
                "ask_5": Decimal("5")
            }
        )


@pytest.fixture
def normal_btc_market() -> MarketScenario:
    """Normal BTC market scenario."""
    return MarketScenario.normal_market("BTC-USD", Decimal("50000"))


@pytest.fixture
def volatile_eth_market() -> MarketScenario:
    """Volatile ETH market scenario."""
    return MarketScenario.volatile_market("ETH-USD", Decimal("3000"))


@pytest.fixture
def illiquid_market() -> MarketScenario:
    """Illiquid market scenario."""
    return MarketScenario.illiquid_market("SOL-USD", Decimal("100"))


# =============================================================================
# RISK CONFIGURATION FIXTURES
# =============================================================================

@pytest.fixture
def conservative_risk_config() -> RiskConfig:
    """Conservative risk configuration for testing."""
    return RiskConfig(
        kill_switch_enabled=False,
        enable_pre_trade_liq_projection=True,
        max_leverage=3,                     # Conservative leverage
        leverage_max_per_symbol={"BTC-USD": 1, "ETH-USD": 1},
        min_liquidation_buffer_pct=0.2,    # 20% buffer
        max_market_impact_bps=2,           # 2 bps max impact
        enable_market_impact_guard=True,
        default_maintenance_margin_rate=0.01  # 1% MMR
    )


@pytest.fixture
def aggressive_risk_config() -> RiskConfig:
    """Aggressive risk configuration for testing."""
    return RiskConfig(
        kill_switch_enabled=False,
        enable_pre_trade_liq_projection=True,
        max_leverage=10,                    # Aggressive leverage
        leverage_max_per_symbol={"BTC-USD": 3, "ETH-USD": 4},
        min_liquidation_buffer_pct=0.1,    # 10% buffer
        max_market_impact_bps=10,          # 10 bps max impact
        enable_market_impact_guard=True,
        default_maintenance_margin_rate=0.005  # 0.5% MMR
    )


@pytest.fixture
def emergency_risk_config() -> RiskConfig:
    """Emergency risk configuration with kill switch enabled."""
    return RiskConfig(
        kill_switch_enabled=True,
        enable_pre_trade_liq_projection=True,
        max_leverage=2,                     # Very conservative leverage
        leverage_max_per_symbol={"BTC-USD": 0.5},  # Reduced leverage
        min_liquidation_buffer_pct=0.3,    # 30% buffer
        max_market_impact_bps=1,           # 1 bps max impact
        enable_market_impact_guard=True,
        default_maintenance_margin_rate=0.02  # 2% MMR
    )


# =============================================================================
# PRODUCT AND MARKET TYPE FIXTURES
# =============================================================================

@dataclass(frozen=True)
class TestProduct:
    """Test product implementation."""
    symbol: str
    market_type: MarketType
    min_order_size: Decimal
    max_order_size: Decimal
    tick_size: Decimal
    step_size: Decimal

    def __post_init__(self):
        object.__setattr__(self, 'id', self.symbol)

    @property
    def base_increment(self) -> Decimal:
        return self.step_size

    @property
    def quote_increment(self) -> Decimal:
        return self.tick_size


@pytest.fixture
def btc_perpetual_product() -> TestProduct:
    """BTC perpetual product for testing."""
    return TestProduct(
        symbol="BTC-USD",
        market_type=MarketType.PERPETUAL,
        min_order_size=Decimal("0.001"),
        max_order_size=Decimal("100"),
        tick_size=Decimal("0.1"),
        step_size=Decimal("0.001")
    )


@pytest.fixture
def eth_spot_product() -> TestProduct:
    """ETH spot product for testing."""
    return TestProduct(
        symbol="ETH-USD",
        market_type=MarketType.SPOT,
        min_order_size=Decimal("0.01"),
        max_order_size=Decimal("1000"),
        tick_size=Decimal("0.01"),
        step_size=Decimal("0.01")
    )


# =============================================================================
# EVENT STORE AND TELEMETRY FIXTURES
# =============================================================================

class MockEventStore:
    """Mock event store for testing."""

    def __init__(self) -> None:
        self.metrics: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []

    def append_metric(self, *args, **kwargs) -> None:
        """Record a metric event."""
        if kwargs:
            self.metrics.append(kwargs)
        else:
            self.metrics.append({"args": args})

    def append(self, event: dict[str, Any]) -> None:
        """Record an event."""
        self.events.append(event)

    def clear(self) -> None:
        """Clear all recorded events."""
        self.metrics.clear()
        self.events.clear()

    def get_last_metric(self) -> dict[str, Any] | None:
        """Get the last recorded metric."""
        return self.metrics[-1] if self.metrics else None

    def get_metrics_by_type(self, event_type: str) -> list[dict[str, Any]]:
        """Get metrics by event type."""
        return [m for m in self.metrics if m.get("event_type") == event_type]


@pytest.fixture
def mock_event_store() -> MockEventStore:
    """Mock event store for testing."""
    return MockEventStore()


# =============================================================================
# MARKET IMPACT AND LIQUIDITY FIXTURES
# =============================================================================

@dataclass(frozen=True)
class MarketImpactResult:
    """Market impact calculation result."""
    estimated_impact_bps: Decimal
    slippage_cost: Decimal
    liquidity_sufficient: bool
    recommended_slicing: int | None = None
    max_slice_size: Decimal | None = None

    @classmethod
    def minimal_impact(cls) -> MarketImpactResult:
        """Minimal market impact scenario."""
        return cls(
            estimated_impact_bps=Decimal("0.5"),
            slippage_cost=Decimal("0.25"),
            liquidity_sufficient=True,
            recommended_slicing=1,
            max_slice_size=Decimal("100")
        )

    @classmethod
    def high_impact(cls) -> MarketImpactResult:
        """High market impact scenario."""
        return cls(
            estimated_impact_bps=Decimal("15"),
            slippage_cost=Decimal("7.5"),
            liquidity_sufficient=False,
            recommended_slicing=5,
            max_slice_size=Decimal("1")
        )

    @classmethod
    def insufficient_liquidity(cls) -> MarketImpactResult:
        """Insufficient liquidity scenario."""
        return cls(
            estimated_impact_bps=Decimal("50"),
            slippage_cost=Decimal("25"),
            liquidity_sufficient=False,
            recommended_slicing=10,
            max_slice_size=Decimal("0.1")
        )


@pytest.fixture
def minimal_market_impact() -> MarketImpactResult:
    """Minimal market impact fixture."""
    return MarketImpactResult.minimal_impact()


@pytest.fixture
def high_market_impact() -> MarketImpactResult:
    """High market impact fixture."""
    return MarketImpactResult.high_impact()


@pytest.fixture
def insufficient_liquidity_impact() -> MarketImpactResult:
    """Insufficient liquidity impact fixture."""
    return MarketImpactResult.insufficient_liquidity()


# =============================================================================
# RUNTIME SETTINGS FIXTURE
# =============================================================================

@pytest.fixture
def risk_runtime_settings() -> RuntimeSettings:
    """Runtime settings configured for risk testing."""
    return RuntimeSettings(
        raw_env={
            "RISK_CHECK_INTERVAL": "1.0",
            "RISK_EMERGENCY_MODE": "false",
            "RISK_MARGIN_CALL_THRESHOLD": "0.8",
            "RISK_LIQUIDATION_THRESHOLD": "0.9",
            "CORRELATION_LOOKBACK_DAYS": "30",
            "VOLATILITY_LOOKBACK_HOURS": "24",
        },
        runtime_root=Path("/tmp/test_risk"),
        event_store_root_override=None,
        coinbase_default_quote="USD",
        coinbase_default_quote_overridden=False,
        coinbase_enable_derivatives=True,
        coinbase_enable_derivatives_overridden=False,
        perps_enable_streaming=True,
        perps_stream_level=2,
        perps_paper_trading=True,
        perps_force_mock=False,
        perps_skip_startup_reconcile=False,
        perps_position_fraction=Decimal("0.1"),
        order_preview_enabled=False,
        spot_force_live=False,
        broker_hint=None,
        coinbase_sandbox_enabled=True,
        coinbase_api_mode="sandbox",
        risk_config_path=None,
        coinbase_intx_portfolio_uuid=None,
    )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_position_dict(positions: list[SyntheticPosition]) -> dict[str, dict[str, Any]]:
    """Convert synthetic positions to dict format expected by risk functions."""
    result = {}
    for pos in positions:
        result[pos.symbol] = {
            "side": pos.side,
            "quantity": str(pos.quantity),
            "price": str(pos.entry_price),
            "mark_price": str(pos.mark_price),
            "unrealized_pnl": str(pos.unrealized_pnl)
        }
    return result


def create_trade_request(
    symbol: str,
    side: str,
    quantity: Decimal,
    price: Decimal,
    product: Product
) -> dict[str, Any]:
    """Create a trade request for testing."""
    return {
        "symbol": symbol,
        "side": side,
        "quantity": quantity,
        "price": price,
        "product": product,
        "order_type": "limit",
        "time_in_force": "gtc"
    }


# =============================================================================
# EMERGENCY SCENARIO FACTORIES
# =============================================================================

@dataclass(frozen=True)
class EmergencyScenario:
    """Emergency scenario for testing risk responses."""
    name: str
    description: str
    trigger_conditions: dict[str, Any]
    expected_response: str
    portfolio_state: SyntheticPortfolio
    market_conditions: dict[str, MarketScenario]

    @classmethod
    def flash_crash(cls) -> EmergencyScenario:
        """Flash crash emergency scenario."""
        btc_crash = MarketScenario.volatile_market("BTC-USD", Decimal("30000"))  # -40%
        eth_crash = MarketScenario.volatile_market("ETH-USD", Decimal("1500"))   # -50%

        # Create losing portfolio
        btc_pos = SyntheticPosition.create_long("BTC-USD", Decimal("1"), Decimal("50000"), Decimal("30000"))
        eth_pos = SyntheticPosition.create_long("ETH-USD", Decimal("10"), Decimal("3000"), Decimal("1500"))
        portfolio = SyntheticPortfolio.with_positions([btc_pos, eth_pos], Decimal("5000"))  # -75% equity

        return cls(
            name="flash_crash",
            description="Simultaneous 40-50% price crashes across major assets",
            trigger_conditions={
                "price_drop_pct": 0.4,
                "volatility_spike": 2.0,
                "portfolio_loss_pct": 0.75
            },
            expected_response="kill_switch_triggered",
            portfolio_state=portfolio,
            market_conditions={"BTC-USD": btc_crash, "ETH-USD": eth_crash}
        )

    @classmethod
    def liquidity_crisis(cls) -> EmergencyScenario:
        """Liquidity crisis scenario."""
        illiquid_market = MarketScenario.illiquid_market("SOL-USD", Decimal("100"))

        # Large position in illiquid asset
        sol_pos = SyntheticPosition.create_long("SOL-USD", Decimal("1000"), Decimal("120"), Decimal("100"))
        portfolio = SyntheticPortfolio.with_positions([sol_pos], Decimal("10000"))

        return cls(
            name="liquidity_crisis",
            description="Large position in extremely illiquid market",
            trigger_conditions={
                "liquidity_depth_usd": 5,
                "position_notional_usd": 100000,
                "market_impact_estimate_bps": 50
            },
            expected_response="trade_blocked_liquidation_risk",
            portfolio_state=portfolio,
            market_conditions={"SOL-USD": illiquid_market}
        )

    @classmethod
    def margin_call_cascade(cls) -> EmergencyScenario:
        """Margin call cascade scenario."""
        # Multiple positions approaching liquidation
        btc_pos = SyntheticPosition.create_short("BTC-USD", Decimal("2"), Decimal("40000"), Decimal("45000"))
        eth_pos = SyntheticPosition.create_short("ETH-USD", Decimal("20"), Decimal("2500"), Decimal("2800"))

        portfolio = SyntheticPortfolio.with_positions([btc_pos, eth_pos], Decimal("2000"))  # Low margin

        normal_btc = MarketScenario.normal_market("BTC-USD", Decimal("45000"))
        normal_eth = MarketScenario.normal_market("ETH-USD", Decimal("2800"))

        return cls(
            name="margin_call_cascade",
            description="Multiple positions approaching liquidation thresholds",
            trigger_conditions={
                "margin_utilization_pct": 0.95,
                "maintenance_margin_ratio": 0.15,
                "equity_buffer_pct": 0.1
            },
            expected_response="reduce_only_mode_activated",
            portfolio_state=portfolio,
            market_conditions={"BTC-USD": normal_btc, "ETH-USD": normal_eth}
        )


@pytest.fixture
def flash_crash_scenario() -> EmergencyScenario:
    """Flash crash emergency scenario."""
    return EmergencyScenario.flash_crash()


@pytest.fixture
def liquidity_crisis_scenario() -> EmergencyScenario:
    """Liquidity crisis emergency scenario."""
    return EmergencyScenario.liquidity_crisis()


@pytest.fixture
def margin_call_cascade_scenario() -> EmergencyScenario:
    """Margin call cascade emergency scenario."""
    return EmergencyScenario.margin_call_cascade()