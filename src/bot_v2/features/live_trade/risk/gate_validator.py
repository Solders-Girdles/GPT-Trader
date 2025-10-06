"""Risk gate validation for trading safety checks.

This module provides utilities for validating risk gates before executing trading
strategies, including volatility circuit breakers and market data staleness checks.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import TYPE_CHECKING

from bot_v2.features.live_trade.risk_runtime import CircuitBreakerAction

if TYPE_CHECKING:
    from collections.abc import Sequence

    from bot_v2.features.live_trade.risk_runtime import RiskManager

logger = logging.getLogger(__name__)


class RiskGateValidator:
    """Validates trading safety gates before strategy execution.

    This validator is responsible for checking multiple risk gates that determine
    whether it's safe to proceed with trading:
    1. Volatility circuit breakers (with kill switch detection)
    2. Market data staleness validation

    Example:
        >>> validator = RiskGateValidator(risk_manager)
        >>> marks = [Decimal("50000"), Decimal("50100"), ...]
        >>> if validator.validate_gates("BTC-USD", marks, lookback_window=20):
        ...     # Safe to proceed with trading
        ...     pass
    """

    def __init__(self, risk_manager: RiskManager) -> None:
        """Initialize validator with risk manager.

        Args:
            risk_manager: Risk manager instance for circuit breaker and staleness checks
        """
        self.risk_manager = risk_manager

    def validate_gates(self, symbol: str, marks: Sequence[Decimal], lookback_window: int) -> bool:
        """Validate all risk gates for a symbol.

        Runs the following checks:
        1. Volatility circuit breaker (kills trading if triggered with KILL_SWITCH action)
        2. Market data staleness (blocks trading if data is stale)

        Args:
            symbol: Trading symbol to validate
            marks: Recent mark prices for volatility analysis
            lookback_window: Number of marks to use for volatility check

        Returns:
            True if all gates pass (safe to trade), False if any gate blocks trading

        Note:
            - Exceptions during checks are logged but don't block trading
            - Only KILL_SWITCH action from circuit breaker blocks trading
            - Staleness check blocking is determined by risk_manager

        Example:
            >>> marks = [Decimal("50000"), Decimal("50100"), Decimal("50200")]
            >>> validator.validate_gates("BTC-USD", marks, lookback_window=20)
            True  # All gates pass
        """
        # Check volatility circuit breaker
        if not self._check_volatility_circuit_breaker(symbol, marks, lookback_window):
            return False

        # Check market data staleness
        if not self._check_mark_staleness(symbol):
            return False

        return True

    def _check_volatility_circuit_breaker(
        self, symbol: str, marks: Sequence[Decimal], lookback_window: int
    ) -> bool:
        """Check volatility circuit breaker for symbol.

        Args:
            symbol: Trading symbol
            marks: Recent mark prices
            lookback_window: Number of marks to analyze

        Returns:
            True if safe to proceed, False if kill switch triggered
        """
        try:
            window = marks[-lookback_window:]
            cb = self.risk_manager.check_volatility_circuit_breaker(symbol, list(window))
            if cb.triggered and cb.action is CircuitBreakerAction.KILL_SWITCH:
                logger.warning(f"Kill switch tripped by volatility CB for {symbol}")
                return False
        except Exception as exc:
            logger.debug(
                "Volatility circuit breaker check failed for %s: %s",
                symbol,
                exc,
                exc_info=True,
            )

        return True

    def _check_mark_staleness(self, symbol: str) -> bool:
        """Check if market data is stale for symbol.

        Args:
            symbol: Trading symbol

        Returns:
            True if data is fresh, False if stale
        """
        try:
            if self.risk_manager.check_mark_staleness(symbol):
                logger.warning(f"Skipping {symbol} due to stale market data")
                return False
        except Exception as exc:
            logger.debug("Mark staleness check failed for %s: %s", symbol, exc, exc_info=True)

        return True
