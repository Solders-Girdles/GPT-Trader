"""Base class for hybrid trading strategies.

Hybrid strategies can trade across both spot and CFM futures markets,
enabling strategies like basis trading, hedging, and leverage optimization.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from gpt_trader.features.live_trade.strategies.base import StatefulStrategyBase
from gpt_trader.features.live_trade.symbols import (
    get_cfm_symbol,
)
from gpt_trader.utilities.logging_patterns import get_logger

from .types import (
    Action,
    HybridDecision,
    HybridMarketData,
    HybridPositionState,
    HybridStrategyConfig,
    TradingMode,
)

if TYPE_CHECKING:
    from gpt_trader.core import Product
    from gpt_trader.features.live_trade.strategies.base import MarketDataContext
    from gpt_trader.features.live_trade.strategies.perps_baseline.strategy import Decision

logger = get_logger(__name__, component="hybrid_strategy")


class HybridStrategyBase(StatefulStrategyBase):
    """Abstract base class for hybrid trading strategies.

    Provides infrastructure for strategies that operate across both spot and CFM
    futures markets. Subclasses implement the decision logic while this base
    handles symbol normalization, position tracking, and market data coordination.

    Use cases:
    - Basis trading: Long spot, short futures when futures at premium
    - Hedging: Hold spot, hedge with futures
    - Leverage optimization: Use CFM for leveraged positions

    Example:
        class BasisTradingStrategy(HybridStrategyBase):
            def decide_hybrid(
                self,
                market_data: HybridMarketData,
                positions: HybridPositionState,
                equity: Decimal,
            ) -> list[HybridDecision]:
                # Implementation...
    """

    def __init__(self, config: HybridStrategyConfig) -> None:
        """Initialize hybrid strategy.

        Args:
            config: Strategy configuration.
        """
        self.config = config
        self._position_state = HybridPositionState()

        # Pre-compute symbols
        self._spot_symbol = config.spot_symbol or f"{config.base_symbol}-{config.quote_currency}"
        cfm_sym = get_cfm_symbol(config.base_symbol)
        self._cfm_symbol = config.cfm_symbol or cfm_sym or self._spot_symbol

        logger.info(
            "Initialized hybrid strategy: spot=%s, cfm=%s",
            self._spot_symbol,
            self._cfm_symbol,
        )

    @property
    def spot_symbol(self) -> str:
        """Get the spot trading symbol."""
        return self._spot_symbol

    @property
    def cfm_symbol(self) -> str:
        """Get the CFM futures trading symbol."""
        return self._cfm_symbol

    @property
    def position_state(self) -> HybridPositionState:
        """Get current position state."""
        return self._position_state

    def decide(
        self,
        symbol: str,
        current_mark: Decimal,
        position_state: dict[str, Any] | None,
        recent_marks: Sequence[Decimal],
        equity: Decimal,
        product: Product | None,
        market_data: MarketDataContext | None = None,
    ) -> Decision:
        """Standard strategy interface - delegates to decide_hybrid.

        This method adapts the hybrid strategy to the standard StrategyProtocol
        interface. For full hybrid functionality, use decide_hybrid() directly.

        Args:
            symbol: Trading symbol
            current_mark: Current mark price
            position_state: Position state dict
            recent_marks: Recent price history
            equity: Account equity
            product: Product specification
            market_data: Optional enhanced market data (orderbook depth, trade flow)

        Returns:
            Standard Decision (first actionable decision or HOLD)
        """
        # Import here to avoid circular dependency
        from gpt_trader.features.live_trade.strategies.perps_baseline.strategy import (
            Action as LegacyAction,
        )
        from gpt_trader.features.live_trade.strategies.perps_baseline.strategy import (
            Decision,
        )

        # Create market data from standard inputs
        hybrid_market_data = HybridMarketData(
            symbol=symbol,
            spot_price=current_mark,
            # Futures price not available in standard interface
        )

        # Update position state from dict if provided
        if position_state:
            self._update_position_state_from_dict(position_state)

        # Get hybrid decisions
        decisions = self.decide_hybrid(hybrid_market_data, self._position_state, equity)

        # Convert first actionable decision to standard Decision
        for decision in decisions:
            if decision.is_actionable():
                # Map hybrid Action to legacy Action
                action_map = {
                    Action.BUY: LegacyAction.BUY,
                    Action.SELL: LegacyAction.SELL,
                    Action.HOLD: LegacyAction.HOLD,
                    Action.CLOSE: LegacyAction.CLOSE,
                    Action.CLOSE_LONG: LegacyAction.CLOSE,
                    Action.CLOSE_SHORT: LegacyAction.CLOSE,
                }
                return Decision(
                    action=action_map.get(decision.action, LegacyAction.HOLD),
                    reason=decision.reason,
                    confidence=decision.confidence,
                    indicators=decision.indicators,
                )

        # No actionable decision - return HOLD
        return Decision(
            action=LegacyAction.HOLD,
            reason="No actionable hybrid decision",
            confidence=0.0,
        )

    @abstractmethod
    def decide_hybrid(
        self,
        market_data: HybridMarketData,
        positions: HybridPositionState,
        equity: Decimal,
    ) -> list[HybridDecision]:
        """Generate hybrid trading decisions.

        Subclasses must implement this method to provide strategy-specific
        decision logic. Can return decisions for both spot and CFM markets.

        Args:
            market_data: Market data with spot and futures prices.
            positions: Current position state across markets.
            equity: Total account equity for position sizing.

        Returns:
            List of decisions to execute (can be empty for HOLD).
        """
        ...

    def _update_position_state_from_dict(self, state: dict[str, Any]) -> None:
        """Update internal position state from a dictionary.

        Args:
            state: Position state dictionary (from engine or persistence).
        """
        # Handle spot position data
        if "spot" in state:
            spot = state["spot"]
            self._position_state.spot_quantity = Decimal(str(spot.get("quantity", 0)))
            if spot.get("entry_price"):
                self._position_state.spot_entry_price = Decimal(str(spot["entry_price"]))
            self._position_state.spot_side = spot.get("side", "flat")

        # Handle CFM position data
        if "cfm" in state:
            cfm = state["cfm"]
            self._position_state.cfm_quantity = Decimal(str(cfm.get("quantity", 0)))
            if cfm.get("entry_price"):
                self._position_state.cfm_entry_price = Decimal(str(cfm["entry_price"]))
            self._position_state.cfm_side = cfm.get("side", "flat")
            self._position_state.cfm_leverage = cfm.get("leverage", 1)

        # Handle flat position dict format (from some engines)
        if "quantity" in state:
            quantity = Decimal(str(state["quantity"]))
            if quantity != 0:
                side = "long" if quantity > 0 else "short"
                product_type = state.get("product_type", "SPOT")
                if product_type == "SPOT":
                    self._position_state.spot_quantity = abs(quantity)
                    self._position_state.spot_side = side
                    if state.get("entry_price"):
                        self._position_state.spot_entry_price = Decimal(str(state["entry_price"]))
                else:
                    self._position_state.cfm_quantity = abs(quantity)
                    self._position_state.cfm_side = side
                    if state.get("entry_price"):
                        self._position_state.cfm_entry_price = Decimal(str(state["entry_price"]))
                    self._position_state.cfm_leverage = state.get("leverage", 1)

    def calculate_position_size(
        self,
        equity: Decimal,
        mode: TradingMode,
        leverage: int = 1,
    ) -> Decimal:
        """Calculate position size based on config and equity.

        Args:
            equity: Available account equity.
            mode: Trading mode (spot or CFM).
            leverage: Leverage multiplier (for CFM).

        Returns:
            Position size in base currency units.
        """
        if mode == TradingMode.SPOT_ONLY:
            size_pct = self.config.spot_position_size_pct
        elif mode == TradingMode.CFM_ONLY:
            size_pct = self.config.cfm_position_size_pct
        else:
            # Hybrid - use average or specific logic
            size_pct = (self.config.spot_position_size_pct + self.config.cfm_position_size_pct) / 2

        # For CFM, leverage effectively increases buying power
        effective_equity = (
            equity * Decimal(str(leverage)) if mode == TradingMode.CFM_ONLY else equity
        )

        return effective_equity * Decimal(str(size_pct))

    def serialize_state(self) -> dict[str, Any]:
        """Serialize strategy state for persistence.

        Returns:
            Serializable state dictionary.
        """
        return {
            "position_state": self._position_state.to_dict(),
            "config": {
                "base_symbol": self.config.base_symbol,
                "quote_currency": self.config.quote_currency,
                "enable_spot": self.config.enable_spot,
                "enable_cfm": self.config.enable_cfm,
            },
        }

    def deserialize_state(self, state: dict[str, Any]) -> None:
        """Restore strategy state from serialized data.

        Args:
            state: Previously serialized state.
        """
        if "position_state" in state:
            pos = state["position_state"]
            self._position_state.spot_quantity = Decimal(str(pos.get("spot_quantity", 0)))
            if pos.get("spot_entry_price"):
                self._position_state.spot_entry_price = Decimal(str(pos["spot_entry_price"]))
            self._position_state.spot_side = pos.get("spot_side", "flat")

            self._position_state.cfm_quantity = Decimal(str(pos.get("cfm_quantity", 0)))
            if pos.get("cfm_entry_price"):
                self._position_state.cfm_entry_price = Decimal(str(pos["cfm_entry_price"]))
            self._position_state.cfm_side = pos.get("cfm_side", "flat")
            self._position_state.cfm_leverage = pos.get("cfm_leverage", 1)

    def create_spot_decision(
        self,
        action: Action,
        quantity: Decimal,
        reason: str,
        confidence: float = 0.5,
    ) -> HybridDecision:
        """Helper to create a spot market decision.

        Args:
            action: Trading action.
            quantity: Position size.
            reason: Explanation for the decision.
            confidence: Confidence score (0-1).

        Returns:
            HybridDecision for spot market.
        """
        return HybridDecision(
            action=action,
            symbol=self._spot_symbol,
            mode=TradingMode.SPOT_ONLY,
            quantity=quantity,
            leverage=1,
            reason=reason,
            confidence=confidence,
        )

    def create_cfm_decision(
        self,
        action: Action,
        quantity: Decimal,
        reason: str,
        leverage: int | None = None,
        confidence: float = 0.5,
    ) -> HybridDecision:
        """Helper to create a CFM futures decision.

        Args:
            action: Trading action.
            quantity: Position size.
            reason: Explanation for the decision.
            leverage: Leverage multiplier (uses config default if None).
            confidence: Confidence score (0-1).

        Returns:
            HybridDecision for CFM market.
        """
        return HybridDecision(
            action=action,
            symbol=self._cfm_symbol,
            mode=TradingMode.CFM_ONLY,
            quantity=quantity,
            leverage=leverage or self.config.cfm_default_leverage,
            reason=reason,
            confidence=confidence,
        )
