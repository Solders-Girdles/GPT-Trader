"""Liquidity-aware strategy filters backed by real-time market data."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from bot_v2.utilities.logging_patterns import get_logger

__all__ = [
    "LiquidityGates",
    "MarketSnapshot",
    "LiquidityFilter",
    "RSIConfirmation",
    "LiquidityStrategyFilter",
    "create_test_snapshot",
]

logger = get_logger(__name__, component="live_trade_strategy")


@dataclass
class LiquidityGates:
    """Configurable liquidity thresholds for strategy filtering."""

    spread_bps_max: float = 50.0  # Max spread in basis points
    l1_min: Decimal = Decimal("10")  # Min L1 depth
    l10_min: Decimal = Decimal("100")  # Min L10 depth
    vol_1m_min: float = 50.0  # Min 1-minute volume


@dataclass
class MarketSnapshot:
    """Market microstructure data from WebSocket."""

    symbol: str
    mid: Decimal | None = None
    spread_bps: float | None = None
    depth_l1: Decimal | None = None
    depth_l10: Decimal | None = None
    vol_1m: float | None = None
    vol_5m: float | None = None
    last_update: str | None = None

    @property
    def is_valid(self) -> bool:
        """Check if snapshot has required data."""
        return all(
            [
                self.mid is not None,
                self.spread_bps is not None,
                self.depth_l1 is not None,
                self.depth_l10 is not None,
                self.vol_1m is not None,
            ]
        )


class LiquidityFilter:
    """Filter strategy entries based on market liquidity conditions."""

    def __init__(self, gates: LiquidityGates) -> None:
        self.gates = gates

    def should_reject_entry(self, snapshot: MarketSnapshot) -> tuple[bool, str]:
        """
        Check if entry should be rejected due to poor liquidity.

        Returns:
            (should_reject, reason)
        """
        if not snapshot.is_valid:
            return True, f"Incomplete market data for {snapshot.symbol}"

        # Check spread
        if snapshot.spread_bps is not None and snapshot.spread_bps > self.gates.spread_bps_max:
            return True, f"Spread {snapshot.spread_bps:.1f}bps > {self.gates.spread_bps_max}bps"

        # Check L1 depth
        if snapshot.depth_l1 is not None and snapshot.depth_l1 < self.gates.l1_min:
            return True, f"L1 depth {snapshot.depth_l1} < {self.gates.l1_min}"

        # Check L10 depth
        if snapshot.depth_l10 is not None and snapshot.depth_l10 < self.gates.l10_min:
            return True, f"L10 depth {snapshot.depth_l10} < {self.gates.l10_min}"

        # Check volume
        if snapshot.vol_1m is not None and snapshot.vol_1m < self.gates.vol_1m_min:
            return True, f"1m volume {snapshot.vol_1m} < {self.gates.vol_1m_min}"

        return False, "Liquidity gates passed"


class RSIConfirmation:
    """RSI confirmation for MA crossover signals."""

    def __init__(self, period: int = 14, oversold: float = 30.0, overbought: float = 70.0) -> None:
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self._price_history: dict[str, list[float]] = {}

    def update_prices(self, symbol: str, price: Decimal) -> None:
        """Update price history for RSI calculation."""
        if symbol not in self._price_history:
            self._price_history[symbol] = []

        prices = self._price_history[symbol]
        prices.append(float(price))

        # Keep only required history
        if len(prices) > self.period + 1:
            prices.pop(0)

    def calculate_rsi(self, symbol: str) -> float | None:
        """Calculate current RSI for symbol."""
        if symbol not in self._price_history:
            return None

        prices = self._price_history[symbol]
        if len(prices) < self.period + 1:
            return None

        # Calculate price changes
        deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]

        # Separate gains and losses
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]

        # Calculate average gain/loss
        avg_gain = sum(gains[-self.period :]) / self.period
        avg_loss = sum(losses[-self.period :]) / self.period

        if avg_loss == 0:
            return 100.0  # All gains

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def confirm_buy_signal(self, symbol: str) -> tuple[bool, str]:
        """Confirm buy signal with RSI (not oversold)."""
        rsi = self.calculate_rsi(symbol)

        if rsi is None:
            return False, "Insufficient price history for RSI"

        if rsi >= self.oversold:
            return True, f"RSI {rsi:.1f} confirms buy (>= {self.oversold})"
        else:
            return False, f"RSI {rsi:.1f} too low (< {self.oversold})"

    def confirm_sell_signal(self, symbol: str) -> tuple[bool, str]:
        """Confirm sell signal with RSI (not overbought)."""
        rsi = self.calculate_rsi(symbol)

        if rsi is None:
            return False, "Insufficient price history for RSI"

        if rsi <= self.overbought:
            return True, f"RSI {rsi:.1f} confirms sell (<= {self.overbought})"
        else:
            return False, f"RSI {rsi:.1f} too high (> {self.overbought})"


class LiquidityStrategyFilter:
    """Aggregate liquidity and RSI filters for strategy gating."""

    def __init__(
        self,
        liquidity_gates: LiquidityGates,
        enable_rsi: bool = True,
        per_symbol_gates: dict[str, LiquidityGates] | None = None,
    ) -> None:
        self.liquidity_filter = LiquidityFilter(liquidity_gates)
        self.rsi_confirmation = RSIConfirmation() if enable_rsi else None
        self.enable_rsi = enable_rsi
        self._per_symbol_gates = per_symbol_gates or {}

    def should_reject_entry(
        self,
        signal_side: str,
        snapshot: MarketSnapshot,  # "buy" or "sell"
    ) -> tuple[bool, str]:
        """
        Comprehensive entry filtering with liquidity gates and RSI confirmation.

        Args:
            signal_side: Direction of MA crossover signal
            snapshot: Current market data

        Returns:
            (should_reject, reason)
        """
        # Apply per-symbol liquidity gates if present
        gates = self._per_symbol_gates.get(snapshot.symbol)
        if gates:
            local_filter = LiquidityFilter(gates)
            reject_liq, reason_liq = local_filter.should_reject_entry(snapshot)
        else:
            reject_liq, reason_liq = self.liquidity_filter.should_reject_entry(snapshot)
        if reject_liq:
            return True, f"Liquidity: {reason_liq}"

        # Apply RSI confirmation if enabled
        if self.enable_rsi and self.rsi_confirmation:
            # Update RSI with current mid price
            if snapshot.mid:
                self.rsi_confirmation.update_prices(snapshot.symbol, snapshot.mid)

            if signal_side.lower() == "buy":
                confirmed, reason_rsi = self.rsi_confirmation.confirm_buy_signal(snapshot.symbol)
            else:
                confirmed, reason_rsi = self.rsi_confirmation.confirm_sell_signal(snapshot.symbol)

            if not confirmed:
                return True, f"RSI: {reason_rsi}"

        return False, "All filters passed"

    def log_filter_summary(self, snapshot: MarketSnapshot) -> None:
        """Log current market conditions for debugging."""
        logger.info(f"Market snapshot for {snapshot.symbol}:")
        logger.info(f"  Mid: {snapshot.mid}")
        logger.info(f"  Spread: {snapshot.spread_bps:.1f}bps")
        logger.info(f"  L1/L10 depth: {snapshot.depth_l1}/{snapshot.depth_l10}")
        logger.info(f"  Volume (1m/5m): {snapshot.vol_1m}/{snapshot.vol_5m}")

        if self.rsi_confirmation:
            rsi = self.rsi_confirmation.calculate_rsi(snapshot.symbol)
            rsi_msg = f"{rsi:.1f}" if rsi is not None else "N/A"
            logger.info(f"  RSI: {rsi_msg}")


# Unit test helpers for gating logic
def create_test_snapshot(
    symbol: str = "BTC-PERP",
    spread_bps: float = 10.0,
    depth_l1: float = 100.0,
    depth_l10: float = 1000.0,
    vol_1m: float = 100.0,
) -> MarketSnapshot:
    """Create test market snapshot."""
    return MarketSnapshot(
        symbol=symbol,
        mid=Decimal("50000"),
        spread_bps=spread_bps,
        depth_l1=Decimal(str(depth_l1)),
        depth_l10=Decimal(str(depth_l10)),
        vol_1m=vol_1m,
        vol_5m=vol_1m * 5,
        last_update="2024-01-01T00:00:00Z",
    )
