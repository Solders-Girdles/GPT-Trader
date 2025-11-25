"""Backtesting type definitions."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum


class ClockSpeed(Enum):
    """Clock speed for bar replay."""

    INSTANT = "instant"  # As fast as possible
    REAL_TIME = "1x"  # 1:1 with wall clock
    FAST_10X = "10x"  # 10x faster than real-time
    FAST_100X = "100x"  # 100x faster than real-time


class FeeTier(Enum):
    """Coinbase Advanced Trade fee tiers."""

    TIER_0 = "< 10K"  # < $10K 30-day volume
    TIER_1 = "10K-50K"  # $10K-$50K
    TIER_2 = "50K-100K"  # $50K-$100K
    TIER_3 = "100K-1M"  # $100K-$1M
    TIER_4 = "1M-15M"  # $1M-$15M
    TIER_5 = "15M-75M"  # $15M-$75M
    TIER_6 = "75M-250M"  # $75M-$250M
    TIER_7 = "> 250M"  # > $250M


@dataclass(frozen=True)
class FeeRate:
    """Fee rates for maker and taker orders."""

    maker_bps: Decimal  # Basis points (1 bp = 0.01%)
    taker_bps: Decimal


# Fee tier mapping (as of 2024)
FEE_TIER_RATES: dict[FeeTier, FeeRate] = {
    FeeTier.TIER_0: FeeRate(maker_bps=Decimal("60"), taker_bps=Decimal("80")),
    FeeTier.TIER_1: FeeRate(maker_bps=Decimal("40"), taker_bps=Decimal("60")),
    FeeTier.TIER_2: FeeRate(maker_bps=Decimal("25"), taker_bps=Decimal("40")),
    FeeTier.TIER_3: FeeRate(maker_bps=Decimal("15"), taker_bps=Decimal("25")),
    FeeTier.TIER_4: FeeRate(maker_bps=Decimal("10"), taker_bps=Decimal("20")),
    FeeTier.TIER_5: FeeRate(maker_bps=Decimal("5"), taker_bps=Decimal("15")),
    FeeTier.TIER_6: FeeRate(maker_bps=Decimal("3"), taker_bps=Decimal("10")),
    FeeTier.TIER_7: FeeRate(maker_bps=Decimal("0"), taker_bps=Decimal("5")),
}


@dataclass
class SimulationConfig:
    """Configuration for backtesting simulation."""

    # Time range
    start_date: datetime
    end_date: datetime
    granularity: str  # ONE_MINUTE, FIVE_MINUTE, etc.

    # Initial capital
    initial_equity_usd: Decimal

    # Fee configuration
    fee_tier: FeeTier = FeeTier.TIER_2

    # Slippage model (basis points per symbol)
    slippage_bps: dict[str, Decimal] | None = None

    # Spread impact (fraction of spread to apply)
    spread_impact_pct: Decimal = Decimal("0.5")  # 50% of spread

    # Order fill model
    limit_fill_volume_threshold: Decimal = Decimal("2.0")  # 2x order size

    # Clock speed
    clock_speed: ClockSpeed = ClockSpeed.INSTANT

    # Funding PnL
    enable_funding_pnl: bool = True
    funding_accrual_hours: int = 1  # Accrue every hour
    funding_settlement_hours: int = 12  # Settle every 12 hours

    # Validation
    enable_golden_path_validation: bool = False
    enable_chaos_testing: bool = False

    def __post_init__(self) -> None:
        """Set default slippage if not provided."""
        if self.slippage_bps is None:
            # Default slippage: 2 bps for BTC/ETH, 5 bps for others
            object.__setattr__(self, "slippage_bps", {})


@dataclass
class BacktestResult:
    """Results from a backtest run."""

    # Time range
    start_date: datetime
    end_date: datetime
    duration_days: int

    # Performance
    initial_equity: Decimal
    final_equity: Decimal
    total_return: Decimal  # %
    total_return_usd: Decimal

    # PnL breakdown
    realized_pnl: Decimal
    unrealized_pnl: Decimal
    funding_pnl: Decimal
    fees_paid: Decimal

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: Decimal

    # Risk metrics
    max_drawdown: Decimal  # %
    max_drawdown_usd: Decimal
    sharpe_ratio: Decimal | None = None
    sortino_ratio: Decimal | None = None

    # Position statistics
    avg_position_size_usd: Decimal = Decimal("0")
    max_position_size_usd: Decimal = Decimal("0")
    avg_leverage: Decimal = Decimal("0")
    max_leverage: Decimal = Decimal("0")

    # Execution quality
    avg_slippage_bps: Decimal = Decimal("0")
    limit_fill_rate: Decimal = Decimal("0")  # % of limit orders filled

    # Circuit breaker events
    circuit_breaker_triggers: int = 0
    reduce_only_periods: int = 0


@dataclass
class ValidationDivergence:
    """Represents a divergence between live and simulated decisions."""

    cycle_id: str
    symbol: str
    timestamp: datetime

    # Live decision
    live_action: str
    live_quantity: Decimal
    live_price: Decimal | None

    # Simulated decision
    sim_action: str
    sim_quantity: Decimal
    sim_price: Decimal | None

    # Reason for divergence
    reason: str
    impact_pct: Decimal | None = None


@dataclass
class ValidationReport:
    """Report from golden-path validation."""

    cycle_id: str
    timestamp: datetime
    total_decisions: int
    matching_decisions: int
    divergences: list[ValidationDivergence]

    @property
    def match_rate(self) -> Decimal:
        """Calculate percentage of matching decisions."""
        if self.total_decisions == 0:
            return Decimal("100")
        return (Decimal(self.matching_decisions) / Decimal(self.total_decisions)) * Decimal("100")


@dataclass
class ChaosScenario:
    """Configuration for chaos testing scenario."""

    name: str
    enabled: bool = True

    # Missing candles
    missing_candles_probability: Decimal = Decimal("0")  # 0-1

    # Stale marks
    stale_marks_delay_seconds: int = 0

    # Wide spreads
    spread_multiplier: Decimal = Decimal("1.0")  # 1.0 = normal, 5.0 = 5x wider

    # Order errors
    order_error_probability: Decimal = Decimal("0")  # 0-1

    # Partial fills
    partial_fill_probability: Decimal = Decimal("0")  # 0-1
    partial_fill_pct: Decimal = Decimal("50")  # % of order filled

    # Network latency
    network_latency_ms: int = 0


@dataclass
class PortfolioType(Enum):
    """Portfolio type for transfers."""

    SPOT = "spot"
    FUTURES = "futures"
    PERPS = "perps"
