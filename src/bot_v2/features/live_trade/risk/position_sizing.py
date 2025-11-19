"""
Position sizing calculations and dynamic estimator integration.

Handles position size calculations with dynamic estimator fallback.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, TYPE_CHECKING

from bot_v2.features.brokerages.core.interfaces import Product
from bot_v2.persistence.event_store import EventStore

if TYPE_CHECKING:
    from bot_v2.orchestration.configuration.risk.model import RiskConfig
from bot_v2.utilities.logging_patterns import get_logger
from bot_v2.utilities.telemetry import emit_metric

logger = get_logger(__name__, component="live_trade_risk")


@dataclass
class PositionSizingContext:
    """Context for position sizing requests."""

    symbol: str = ""
    side: str = ""  # "buy" or "sell"
    equity: Decimal | None = None
    available_equity: Decimal | None = None
    current_price: Decimal | None = None
    mark_price: Decimal | None = None
    strategy_name: str = ""
    method: str = "intelligent"
    target_leverage: Decimal | None = None
    product: Product | None = None
    current_position_quantity: Decimal = Decimal("0")
    strategy_multiplier: float = 1.0
    current_positions: dict[str, Any] | None = None
    risk_metrics: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PositionSizingAdvice:
    """Advice from position sizing calculation."""

    symbol: str = ""
    side: str = ""
    target_notional: Decimal = Decimal("0")
    target_quantity: Decimal = Decimal("0")
    used_dynamic: bool = False
    reduce_only: bool = False
    reason: str | None = None
    fallback_used: bool = False
    recommended_quantity: Decimal | None = None
    max_quantity: Decimal | None = None
    min_quantity: Decimal | None = None
    confidence: float | None = None
    notes: list[str] | None = None

    def __post_init__(self) -> None:
        if self.recommended_quantity is None:
            self.recommended_quantity = self.target_quantity
        if self.max_quantity is None:
            self.max_quantity = self.target_quantity
        if self.min_quantity is None:
            self.min_quantity = Decimal("0")


@dataclass
class ImpactRequest:
    """Request for market impact assessment."""

    symbol: str
    side: str
    quantity: Decimal
    price: Decimal | None = None


@dataclass
class ImpactAssessment:
    """Assessment of market impact for a trade."""

    symbol: str = ""
    side: str = ""
    quantity: Decimal = Decimal("0")
    estimated_impact_bps: Decimal = Decimal("0")
    slippage_cost: Decimal = Decimal("0")
    price_impact: Decimal | None = None
    size_impact: Decimal | None = None
    confidence: Decimal | None = None
    liquidity_sufficient: bool = True
    reason: str | None = None
    recommended_slicing: bool | None = None
    max_slice_size: Decimal | None = None


class PositionSizer:
    """Calculates position sizes using dynamic estimator or fallback logic."""

    def __init__(
        self,
        config: "RiskConfig",
        event_store: EventStore,
        position_size_estimator: (
            Callable[[PositionSizingContext], PositionSizingAdvice] | None
        ) = None,
        impact_estimator: Callable[[ImpactRequest], ImpactAssessment] | None = None,
        is_reduce_only_mode: Callable[[], bool] | None = None,
    ):
        """
        Initialize position sizer.

        Args:
            config: Risk configuration
            event_store: Event store for sizing metrics
            position_size_estimator: Optional dynamic position sizing calculator
            impact_estimator: Optional callable returning market impact assessments
            is_reduce_only_mode: Callable to check if reduce-only mode is active
        """
        self.config = config
        self.event_store = event_store
        self._position_size_estimator = position_size_estimator
        self._impact_estimator = impact_estimator
        self._is_reduce_only_mode = is_reduce_only_mode or (lambda: False)

    def size_position(self, context: PositionSizingContext) -> PositionSizingAdvice:
        """
        Calculate position size using dynamic estimator or fallback logic.

        Args:
            context: Position sizing context with symbol, equity, price, etc.

        Returns:
            Position sizing advice with target notional and quantity
        """
        # If reduce-only mode, return zero sizing
        if self._is_reduce_only_mode():
            advice = PositionSizingAdvice(
                symbol=context.symbol,
                side=context.side,
                target_notional=Decimal("0"),
                target_quantity=Decimal("0"),
                reduce_only=True,
                reason="reduce_only_mode",
                recommended_quantity=Decimal("0"),
                max_quantity=Decimal("0"),
            )
            self._record_sizing_metric(context, advice)
            return advice

        # Try dynamic estimator if available
        if self._position_size_estimator is not None:
            try:
                advice = self._position_size_estimator(context)
                self._record_sizing_metric(context, advice)
                return advice
            except Exception as exc:
                logger.exception("Position size estimator failed for %s", context.symbol)
                emit_metric(
                    self.event_store,
                    "risk_engine",
                    {
                        "event_type": "position_sizing_error",
                        "symbol": context.symbol,
                        "error": str(exc),
                    },
                    logger=logger,
                )

        # Fallback: simple target_leverage-based sizing
        equity = context.equity
        if equity is None:
            equity = context.available_equity
        equity = equity if equity is not None else Decimal("0")

        price = context.current_price
        if price is None or price <= 0:
            price = context.mark_price
        price = price if price is not None else Decimal("0")

        leverage = context.target_leverage if context.target_leverage is not None else Decimal("1")

        target_notional = equity * leverage * Decimal(str(context.strategy_multiplier))
        target_quantity = target_notional / price if price and price > 0 else Decimal("0")

        advice = PositionSizingAdvice(
            symbol=context.symbol,
            side=context.side,
            target_notional=target_notional,
            target_quantity=target_quantity,
            fallback_used=True,
            reason="fallback",
            recommended_quantity=target_quantity,
            max_quantity=target_quantity,
        )
        self._record_sizing_metric(context, advice)
        return advice

    def _record_sizing_metric(
        self, context: PositionSizingContext, advice: PositionSizingAdvice
    ) -> None:
        """Record position sizing metrics to event store."""
        target_notional = advice.target_notional
        price_reference = context.current_price or context.mark_price
        if target_notional == 0 and advice.recommended_quantity and price_reference:
            try:
                target_notional = advice.recommended_quantity * price_reference
            except Exception:
                target_notional = Decimal("0")

        target_quantity = advice.target_quantity
        if target_quantity == 0 and advice.recommended_quantity is not None:
            target_quantity = advice.recommended_quantity

        emit_metric(
            self.event_store,
            "risk_engine",
            {
                "event_type": "position_sizing_advice",
                "symbol": context.symbol,
                "side": context.side,
                "target_notional": float(target_notional),
                "target_quantity": float(target_quantity),
                "used_dynamic": advice.used_dynamic,
                "reduce_only": advice.reduce_only,
                "fallback_used": advice.fallback_used,
                "reason": advice.reason,
            },
            logger=logger,
        )

    def set_impact_estimator(
        self, estimator: Callable[[ImpactRequest], ImpactAssessment] | None
    ) -> None:
        """Install or clear the market-impact estimator hook."""
        self._impact_estimator = estimator
