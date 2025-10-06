"""Factory for creating and configuring execution engines.

Responsibilities:
- Parse environment variables (SLIPPAGE_MULTIPLIERS)
- Determine which engine to use based on risk config
- Configure LiquidityService and impact estimator
- Create AdvancedExecutionEngine or LiveExecutionEngine
"""

from __future__ import annotations

import logging
import os
from decimal import Decimal
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from bot_v2.features.brokerages.core.interfaces import IBrokerage
    from bot_v2.features.live_trade.risk import LiveRiskManager

logger = logging.getLogger(__name__)


class ExecutionEngineFactory:
    """Factory for creating configured execution engines."""

    @staticmethod
    def parse_slippage_multipliers() -> dict[str, float]:
        """Parse SLIPPAGE_MULTIPLIERS env var into dict.

        Format: "BTC-USD:1.5,ETH-USD:2.0"
        Returns: {"BTC-USD": 1.5, "ETH-USD": 2.0}
        """
        slippage_env = os.environ.get("SLIPPAGE_MULTIPLIERS", "")
        slippage_map: dict[str, float] = {}
        if slippage_env:
            try:
                parts = [p for p in slippage_env.split(",") if ":" in p]
                for part in parts:
                    k, v = part.split(":", 1)
                    slippage_map[k.strip()] = float(v)
            except Exception as exc:
                logger.warning(
                    "Invalid SLIPPAGE_MULTIPLIERS entry '%s': %s",
                    slippage_env,
                    exc,
                    exc_info=True,
                )
        return slippage_map

    @staticmethod
    def should_use_advanced_engine(risk_manager: LiveRiskManager) -> bool:
        """Determine if AdvancedExecutionEngine should be used.

        Checks risk_manager.config for:
        - enable_dynamic_position_sizing
        - enable_market_impact_guard
        """
        risk_config = getattr(risk_manager, "config", None)
        if risk_config is None:
            return False

        return getattr(risk_config, "enable_dynamic_position_sizing", False) or getattr(
            risk_config, "enable_market_impact_guard", False
        )

    @staticmethod
    def create_impact_estimator(broker: IBrokerage, risk_manager: LiveRiskManager) -> Any:
        """Create market impact estimator function for risk manager.

        Returns a closure that estimates market impact using LiquidityService.
        """
        from datetime import datetime

        from bot_v2.features.live_trade.liquidity_service import LiquidityService

        liquidity_service = LiquidityService()

        def _impact_estimator(req: Any) -> Any:
            try:
                quote = broker.get_quote(req.symbol)
            except Exception:
                quote = None

            bids: list[tuple[Decimal, Decimal]]
            asks: list[tuple[Decimal, Decimal]]

            # Allow deterministic brokers to seed a custom order book
            seeded_orderbooks = getattr(broker, "order_books", None)
            if seeded_orderbooks and req.symbol in seeded_orderbooks:
                seeded = seeded_orderbooks[req.symbol]
                bids = [(Decimal(str(p)), Decimal(str(s))) for p, s in seeded[0]]
                asks = [(Decimal(str(p)), Decimal(str(s))) for p, s in seeded[1]]
            else:
                mid = None
                if quote is not None and getattr(quote, "last", None) is not None:
                    mid = Decimal(str(quote.last))
                elif (
                    quote is not None
                    and getattr(quote, "bid", None) is not None
                    and getattr(quote, "ask", None) is not None
                ):
                    mid = (Decimal(str(quote.bid)) + Decimal(str(quote.ask))) / Decimal("2")
                if mid is None:
                    mid = Decimal("100")

                tick = None
                if (
                    quote is not None
                    and getattr(quote, "ask", None) is not None
                    and getattr(quote, "bid", None) is not None
                ):
                    spread = Decimal(str(quote.ask)) - Decimal(str(quote.bid))
                    if spread > 0:
                        tick = spread / Decimal("2")
                if tick is None or tick == 0:
                    tick = mid * Decimal("0.0005")

                depth_size = max(Decimal("1000"), abs(Decimal(str(req.quantity))) * Decimal("20"))
                bids = [(mid - tick * Decimal(i + 1), depth_size) for i in range(5)]
                asks = [(mid + tick * Decimal(i + 1), depth_size) for i in range(5)]

            liquidity_service.analyze_order_book(
                req.symbol,
                bids=bids,
                asks=asks,
                timestamp=datetime.utcnow(),
            )
            return liquidity_service.estimate_market_impact(
                symbol=req.symbol,
                side=req.side,
                quantity=Decimal(str(req.quantity)),
                book_data=(bids, asks),
            )

        return _impact_estimator

    @classmethod
    def create_engine(
        cls,
        broker: IBrokerage,
        risk_manager: LiveRiskManager,
        event_store: Any,
        bot_id: str,
        enable_preview: bool,
    ) -> Any:
        """Create and configure execution engine.

        Args:
            broker: Brokerage interface
            risk_manager: Risk manager instance
            event_store: Event store for LiveExecutionEngine
            bot_id: Bot identifier
            enable_preview: Enable order preview mode

        Returns:
            AdvancedExecutionEngine or LiveExecutionEngine
        """
        use_advanced = cls.should_use_advanced_engine(risk_manager)

        # Setup impact estimator if advanced features enabled
        if use_advanced:
            try:
                impact_estimator = cls.create_impact_estimator(broker, risk_manager)
                risk_manager.set_impact_estimator(impact_estimator)
                logger.info("Configured market impact estimator for risk manager")
            except Exception as exc:
                logger.warning(
                    "Failed to initialize LiquidityService impact estimator: %s",
                    exc,
                    exc_info=True,
                )

        # Create appropriate engine
        if use_advanced:
            from bot_v2.features.live_trade.advanced_execution import AdvancedExecutionEngine

            engine = AdvancedExecutionEngine(
                broker=broker,
                risk_manager=risk_manager,
            )
            logger.info("Initialized AdvancedExecutionEngine with dynamic sizing integration")
        else:
            from bot_v2.orchestration.live_execution import LiveExecutionEngine

            slippage_map = cls.parse_slippage_multipliers()
            engine = LiveExecutionEngine(
                broker=broker,
                risk_manager=risk_manager,
                event_store=event_store,
                bot_id=bot_id,
                slippage_multipliers=slippage_map or None,
                enable_preview=enable_preview,
            )
            logger.info("Initialized LiveExecutionEngine with risk integration")

        return engine
