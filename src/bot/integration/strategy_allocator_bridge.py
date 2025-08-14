"""Strategy-Allocator Bridge for GPT-Trader.

This module provides the bridge between strategy signal generation
and portfolio allocation, connecting the strategy engine with the
capital allocation system.
"""

from __future__ import annotations

import logging
from typing import Dict

import pandas as pd

from bot.portfolio.allocator import PortfolioRules, allocate_signals
from bot.strategy.base import Strategy

logger = logging.getLogger(__name__)


class StrategyAllocatorBridge:
    """Bridge between strategy signals and portfolio allocator.

    This class connects strategy signal generation with portfolio allocation,
    providing a clean interface to convert strategy signals into position sizes.

    Attributes:
        strategy: The trading strategy instance
        rules: Portfolio allocation rules and constraints
    """

    def __init__(self, strategy: Strategy, rules: PortfolioRules) -> None:
        """Initialize the bridge with strategy and allocation rules.

        Args:
            strategy: Strategy instance that generates trading signals
            rules: Portfolio rules defining allocation constraints
        """
        self.strategy = strategy
        self.rules = rules
        logger.info(
            f"Initialized StrategyAllocatorBridge with strategy='{strategy.name}' "
            f"and max_positions={rules.max_positions}"
        )

    def process_signals(
        self, market_data: Dict[str, pd.DataFrame], equity: float
    ) -> Dict[str, int]:
        """
        Process strategy signals for all symbols and allocate capital.

        This method orchestrates the complete flow from raw market data
        to position allocations:
        1. Generate signals for each symbol using the strategy
        2. Combine signals with market data for the allocator
        3. Allocate capital based on signals and portfolio rules

        Args:
            market_data: Dict mapping symbol -> OHLCV DataFrame
            equity: Current portfolio equity for position sizing

        Returns:
            Dict mapping symbol -> position size in shares

        Raises:
            ValueError: If market_data is empty or equity is invalid
        """
        if not market_data:
            logger.warning("Empty market_data provided to process_signals")
            return {}

        if equity <= 0:
            raise ValueError(f"Invalid equity value: {equity}. Must be positive.")

        logger.debug(
            f"Processing signals for {len(market_data)} symbols with equity=${equity:,.2f}"
        )

        # Generate signals for each symbol
        signals_map = {}
        symbols_processed = 0
        symbols_with_signals = 0

        for symbol, data in market_data.items():
            try:
                # Generate strategy signals
                signals = self.strategy.generate_signals(data)

                # Combine market data with signals for the allocator
                # The allocator expects OHLCV + signal + atr columns
                combined = data.join(signals, how="left")

                # Validate that we have the required columns
                required_cols = ["Close"]
                missing_cols = [col for col in required_cols if col not in combined.columns]
                if missing_cols:
                    logger.warning(f"Symbol {symbol} missing required columns: {missing_cols}")
                    continue

                signals_map[symbol] = combined
                symbols_processed += 1

                # Check if this symbol has any signals
                if "signal" in signals.columns and signals["signal"].sum() > 0:
                    symbols_with_signals += 1

            except Exception as e:
                logger.error(f"Error processing signals for symbol {symbol}: {e}", exc_info=True)
                continue

        logger.info(
            f"Generated signals for {symbols_processed}/{len(market_data)} symbols, "
            f"{symbols_with_signals} with active signals"
        )

        if not signals_map:
            logger.warning("No valid signals generated for any symbol")
            return {}

        # Allocate capital based on signals
        try:
            allocations = allocate_signals(signals_map, equity, self.rules)

            total_positions = len([qty for qty in allocations.values() if qty > 0])
            total_capital_allocated = sum(
                qty * signals_map[symbol]["Close"].iloc[-1]
                for symbol, qty in allocations.items()
                if qty > 0 and not signals_map[symbol]["Close"].empty
            )

            logger.info(
                f"Allocated {total_positions} positions with "
                f"${total_capital_allocated:,.2f} total capital "
                f"({total_capital_allocated/equity*100:.1f}% of equity)"
            )

            return allocations

        except Exception as e:
            logger.error(f"Error in capital allocation: {e}", exc_info=True)
            return {}

    def get_strategy_info(self) -> Dict[str, any]:
        """Get information about the configured strategy.

        Returns:
            Dict containing strategy name and capabilities
        """
        return {
            "name": self.strategy.name,
            "supports_short": self.strategy.supports_short,
            "strategy_type": type(self.strategy).__name__,
        }

    def get_allocation_rules_info(self) -> Dict[str, any]:
        """Get information about the allocation rules.

        Returns:
            Dict containing key allocation parameters
        """
        return {
            "per_trade_risk_pct": self.rules.per_trade_risk_pct,
            "max_positions": self.rules.max_positions,
            "max_gross_exposure_pct": self.rules.max_gross_exposure_pct,
            "atr_k": self.rules.atr_k,
            "cost_bps": self.rules.cost_bps,
        }

    def validate_configuration(self) -> bool:
        """Validate that the bridge is properly configured.

        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            if not hasattr(self.strategy, "generate_signals"):
                logger.error("Strategy does not implement generate_signals method")
                return False

            if self.rules.per_trade_risk_pct <= 0 or self.rules.per_trade_risk_pct > 1:
                logger.error(
                    f"Invalid per_trade_risk_pct: {self.rules.per_trade_risk_pct}. "
                    "Must be between 0 and 1."
                )
                return False

            if self.rules.max_positions <= 0:
                logger.error(f"Invalid max_positions: {self.rules.max_positions}")
                return False

            logger.info("Bridge configuration validation passed")
            return True

        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
