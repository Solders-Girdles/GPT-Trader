from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any


@dataclass
class MarketState:
    """Data structure for market information."""

    prices: dict[str, Decimal] = field(default_factory=dict)  # Changed from str to Decimal
    last_update: float = 0.0
    price_history: dict[str, list[Decimal]] = field(default_factory=dict)
    spreads: dict[str, Decimal] = field(default_factory=dict)  # symbol -> spread %


@dataclass
class Position:
    """Data structure for a single position.

    Supports both spot and derivatives (CFM futures).
    """

    symbol: str
    quantity: Decimal
    entry_price: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")
    mark_price: Decimal = Decimal("0")
    side: str = "long"
    # CFM/derivatives fields
    leverage: int = 1
    product_type: str = "SPOT"  # "SPOT" or "FUTURE"
    liquidation_price: Decimal | None = None
    liquidation_buffer_pct: float | None = None

    @property
    def is_futures(self) -> bool:
        """Check if this is a futures position."""
        return self.product_type == "FUTURE"


@dataclass
class PortfolioSummary:
    """Data structure for position information."""

    positions: dict[str, Position] = field(default_factory=dict)
    total_unrealized_pnl: Decimal = Decimal("0")  # Changed from str to Decimal
    equity: Decimal = Decimal("0")  # Changed from str to Decimal
    # P&L breakdown
    total_realized_pnl: Decimal = Decimal("0")  # Realized P&L from closed trades
    total_fees: Decimal = Decimal("0")  # Cumulative trading fees

    @property
    def net_pnl(self) -> Decimal:
        """Net P&L = Realized + Unrealized - Fees."""
        return self.total_realized_pnl + self.total_unrealized_pnl - self.total_fees


@dataclass
class Order:
    """Data structure for a single order."""

    order_id: str
    symbol: str
    side: str
    quantity: Decimal  # Changed from str to Decimal
    price: Decimal  # Changed from str to Decimal
    status: str
    type: str = "UNKNOWN"
    time_in_force: str = "UNKNOWN"
    creation_time: float = 0.0  # Epoch timestamp for age calculation
    filled_quantity: Decimal = Decimal("0")
    avg_fill_price: Decimal | None = None
    decision_id: str = ""  # Links to originating strategy decision


@dataclass
class ActiveOrders:
    """Data structure for order information."""

    orders: list[Order] = field(default_factory=list)


@dataclass
class Trade:
    """Data structure for a single trade."""

    trade_id: str
    symbol: str
    side: str
    quantity: Decimal  # Changed from str to Decimal
    price: Decimal  # Changed from str to Decimal
    order_id: str
    time: str
    fee: Decimal = Decimal("0")  # Changed from str to Decimal


@dataclass
class TradeHistory:
    """Data structure for trade information."""

    trades: list[Trade] = field(default_factory=list)


@dataclass
class AccountBalance:
    """Data structure for a single asset balance."""

    asset: str
    total: Decimal  # Changed from str to Decimal
    available: Decimal  # Changed from str to Decimal
    hold: Decimal = Decimal("0")  # Changed from str to Decimal


@dataclass
class AccountSummary:
    """Data structure for account metrics."""

    volume_30d: Decimal = Decimal("0")  # Changed from str to Decimal
    fees_30d: Decimal = Decimal("0")  # Changed from str to Decimal
    fee_tier: str = ""
    balances: list[AccountBalance] = field(default_factory=list)
    # Daily P&L tracking
    daily_pnl: Decimal = Decimal("0")  # P&L for current trading day
    daily_pnl_pct: Decimal = Decimal("0")  # Daily P&L as percentage of equity


@dataclass
class IndicatorContribution:
    """Contribution of a single indicator to the decision.

    Captures how much each indicator influenced the final decision,
    enabling transparency into which factors mattered most.
    """

    name: str
    value: float  # Current indicator value
    contribution: float  # -1.0 to +1.0 (negative = bearish, positive = bullish)
    weight: float = 1.0  # Relative importance weight

    @property
    def is_bullish(self) -> bool:
        """Check if this indicator is contributing bullishly."""
        return self.contribution > 0

    @property
    def is_bearish(self) -> bool:
        """Check if this indicator is contributing bearishly."""
        return self.contribution < 0

    @property
    def abs_contribution(self) -> float:
        """Absolute contribution magnitude for sorting."""
        return abs(self.contribution)


@dataclass
class DecisionData:
    """Data structure for a single strategy decision."""

    symbol: str
    action: str
    reason: str
    confidence: float
    indicators: dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0
    decision_id: str = ""  # Unique ID for linking to orders/trades
    blocked_by: str = ""  # Guard/reason that blocked execution (empty if executed)
    # Indicator contributions for transparency (sorted by impact)
    contributions: list[IndicatorContribution] = field(default_factory=list)

    @property
    def top_contributors(self) -> list[IndicatorContribution]:
        """Get top 3 contributors by absolute contribution."""
        return sorted(self.contributions, key=lambda c: c.abs_contribution, reverse=True)[:3]

    @property
    def bullish_contributors(self) -> list[IndicatorContribution]:
        """Get contributors pushing toward bullish."""
        return [c for c in self.contributions if c.is_bullish]

    @property
    def bearish_contributors(self) -> list[IndicatorContribution]:
        """Get contributors pushing toward bearish."""
        return [c for c in self.contributions if c.is_bearish]


@dataclass
class StrategyState:
    """Data structure for strategy information."""

    active_strategies: list[str] = field(default_factory=list)
    last_decisions: dict[str, DecisionData] = field(default_factory=dict)


@dataclass
class RiskGuard:
    """Data structure for a single risk guard with metadata.

    Provides detailed guard information including severity,
    last-triggered timestamp, and trigger count for debugging.
    """

    name: str
    severity: str = "MEDIUM"  # LOW, MEDIUM, HIGH, CRITICAL
    last_triggered: float = 0.0  # Epoch timestamp (0 = never triggered)
    triggered_count: int = 0  # Total times this guard has triggered
    description: str = ""  # Optional human-readable description

    @property
    def severity_order(self) -> int:
        """Return numeric value for sorting by severity (higher = more severe)."""
        severity_map = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
        return severity_map.get(self.severity.upper(), 2)


@dataclass
class RiskState:
    """Data structure for risk management information.

    Note: position_leverage field was removed as GPT-Trader focuses on spot trading
    where per-position leverage is not applicable. For perpetuals/margin trading,
    this would need to be added back with proper StatusReporter support.
    """

    max_leverage: float = 0.0
    daily_loss_limit_pct: float = 0.0
    current_daily_loss_pct: float = 0.0
    reduce_only_mode: bool = False
    reduce_only_reason: str = ""
    # Legacy: simple guard names (for backward compatibility)
    active_guards: list[str] = field(default_factory=list)
    # Enhanced: full guard objects with metadata
    guards: list[RiskGuard] = field(default_factory=list)


@dataclass
class SystemStatus:
    """Data structure for system health and brokerage connection."""

    api_latency: float = 0.0
    connection_status: str = "UNKNOWN"
    rate_limit_usage: str = "0%"
    memory_usage: str = "0MB"
    cpu_usage: str = "0%"

    # Validation failure tracking (from ValidationFailureTracker)
    validation_failures: dict[str, int] = field(default_factory=dict)
    validation_escalated: bool = False


@dataclass
class TradingStats:
    """Trading performance statistics with sample sizes.

    Computed from trade history with optional time window filtering.
    Includes sample counts for statistical context.
    """

    # Trade counts
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    break_even_trades: int = 0

    # Win/loss metrics
    win_rate: float = 0.0  # 0.0 to 1.0
    avg_win: Decimal = Decimal("0")
    avg_loss: Decimal = Decimal("0")
    profit_factor: float = 0.0  # gross_profit / gross_loss

    # P&L
    total_pnl: Decimal = Decimal("0")
    gross_profit: Decimal = Decimal("0")
    gross_loss: Decimal = Decimal("0")

    # Average trade
    avg_trade_pnl: Decimal = Decimal("0")
    avg_trade_size: Decimal = Decimal("0")

    # Time window info
    window_minutes: int = 0  # 0 = all session
    window_label: str = "All Session"

    @property
    def sample_label(self) -> str:
        """Return sample size label for display (e.g., 'n=23')."""
        return f"n={self.total_trades}"

    @property
    def has_sufficient_data(self) -> bool:
        """Check if enough trades for meaningful statistics (n >= 5)."""
        return self.total_trades >= 5


@dataclass
class ResilienceState:
    """API resilience metrics from CoinbaseClient.

    Tracks latency percentiles, error rates, caching statistics,
    and circuit breaker states for comprehensive API health monitoring.
    """

    # Latency metrics (milliseconds)
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    avg_latency_ms: float = 0.0

    # Error metrics
    error_rate: float = 0.0  # 0.0 to 1.0
    total_requests: int = 0
    total_errors: int = 0

    # Rate limiting
    rate_limit_hits: int = 0
    rate_limit_usage_pct: float = 0.0

    # Cache stats
    cache_hit_rate: float = 0.0
    cache_size: int = 0
    cache_enabled: bool = False

    # Circuit breaker
    circuit_breakers: dict[str, str] = field(default_factory=dict)  # category -> state
    any_circuit_open: bool = False

    last_update: float = 0.0


@dataclass
class ExecutionIssue:
    """A recent execution issue such as a rejection or retry."""

    timestamp: float
    symbol: str
    side: str
    quantity: float
    price: float
    reason: str
    is_retry: bool = False


@dataclass
class ExecutionMetrics:
    """Execution telemetry for order submission tracking.

    Tracks order submission performance including latency, success rates,
    and retry activity for monitoring execution health.
    """

    # Submission counts (rolling window)
    submissions_total: int = 0
    submissions_success: int = 0
    submissions_failed: int = 0
    submissions_rejected: int = 0

    # Latency metrics (milliseconds)
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    last_latency_ms: float = 0.0

    # Retry metrics
    retry_total: int = 0
    retry_rate: float = 0.0  # retries per submission

    # Recent activity
    last_submission_time: float = 0.0
    last_failure_reason: str = ""

    # Reason breakdowns (rolling window)
    rejection_reasons: dict[str, int] = field(default_factory=dict)
    retry_reasons: dict[str, int] = field(default_factory=dict)
    recent_rejections: list[ExecutionIssue] = field(default_factory=list)
    recent_retries: list[ExecutionIssue] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.submissions_total == 0:
            return 100.0
        return (self.submissions_success / self.submissions_total) * 100

    @property
    def is_healthy(self) -> bool:
        """Check if execution metrics indicate healthy state."""
        return self.success_rate >= 95.0 and self.retry_rate < 0.5

    @property
    def top_rejection_reasons(self) -> list[tuple[str, int]]:
        """Get rejection reasons sorted by count (highest first)."""
        return sorted(self.rejection_reasons.items(), key=lambda x: -x[1])

    @property
    def top_retry_reasons(self) -> list[tuple[str, int]]:
        """Get retry reasons sorted by count (highest first)."""
        return sorted(self.retry_reasons.items(), key=lambda x: -x[1])


@dataclass
class StrategyPerformance:
    """Strategy performance metrics for TUI display.

    Aggregates key performance indicators from the trading strategy
    for dashboard visualization.
    """

    # Win/loss metrics
    win_rate: float = 0.0  # 0.0 to 1.0
    profit_factor: float = 0.0  # gross_profit / gross_loss

    # Return metrics
    total_return_pct: float = 0.0  # Total return %
    daily_return_pct: float = 0.0  # Daily return %
    max_drawdown_pct: float = 0.0  # Max drawdown % (negative value)

    # Trade counts
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0

    # Risk-adjusted metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0

    # Volatility
    volatility_pct: float = 0.0  # Annualized volatility

    @property
    def win_rate_pct(self) -> float:
        """Win rate as percentage (0-100)."""
        return self.win_rate * 100

    @property
    def has_sufficient_data(self) -> bool:
        """Check if enough trades for meaningful statistics."""
        return self.total_trades >= 5


@dataclass
class RegimeData:
    """Market regime information for TUI display.

    Captures the current market regime classification and confidence
    from the regime detection system.
    """

    # Current regime
    regime: str = "UNKNOWN"  # BULL_QUIET, BULL_VOLATILE, BEAR_QUIET, etc.
    confidence: float = 0.0  # 0.0 to 1.0

    # Regime metrics
    trend_score: float = 0.0  # -1.0 (bearish) to +1.0 (bullish)
    volatility_pct: float = 0.0  # Current volatility percentile (0.0 to 1.0)
    momentum_score: float = 0.0  # -1.0 to +1.0

    # Stability
    regime_age_ticks: int = 0  # How long in current regime
    transition_probability: float = 0.0  # Likelihood of regime change

    @property
    def is_bullish(self) -> bool:
        """Check if regime is bullish."""
        return self.regime in ("BULL_QUIET", "BULL_VOLATILE")

    @property
    def is_bearish(self) -> bool:
        """Check if regime is bearish."""
        return self.regime in ("BEAR_QUIET", "BEAR_VOLATILE")

    @property
    def is_volatile(self) -> bool:
        """Check if regime is volatile."""
        return self.regime in ("BULL_VOLATILE", "BEAR_VOLATILE", "SIDEWAYS_VOLATILE", "CRISIS")

    @property
    def is_crisis(self) -> bool:
        """Check if regime is crisis mode."""
        return self.regime == "CRISIS"

    @property
    def short_label(self) -> str:
        """Short human-readable regime label."""
        labels = {
            "BULL_QUIET": "BULL",
            "BULL_VOLATILE": "BULL+",
            "BEAR_QUIET": "BEAR",
            "BEAR_VOLATILE": "BEAR+",
            "SIDEWAYS_QUIET": "RANGE",
            "SIDEWAYS_VOLATILE": "CHOP",
            "CRISIS": "CRISIS",
            "UNKNOWN": "---",
        }
        return labels.get(self.regime, self.regime)

    @property
    def icon(self) -> str:
        """Icon representing the regime."""
        icons = {
            "BULL_QUIET": "↑",
            "BULL_VOLATILE": "⇈",
            "BEAR_QUIET": "↓",
            "BEAR_VOLATILE": "⇊",
            "SIDEWAYS_QUIET": "→",
            "SIDEWAYS_VOLATILE": "↔",
            "CRISIS": "⚠",
            "UNKNOWN": "?",
        }
        return icons.get(self.regime, "?")
