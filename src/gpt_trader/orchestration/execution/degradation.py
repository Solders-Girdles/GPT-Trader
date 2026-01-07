"""
Graceful degradation state management for live trading.

Tracks pause states (global and per-symbol) to enable controlled
degradation when guards trip or infrastructure fails.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from gpt_trader.orchestration.configuration.risk.model import RiskConfig

logger = get_logger(__name__, component="degradation")


@dataclass
class PauseRecord:
    """Record of a pause event."""

    until: float  # Unix timestamp when pause expires
    reason: str
    allow_reduce_only: bool = False


@dataclass
class DegradationState:
    """
    Manages graceful degradation state for trading operations.

    Tracks global pauses and per-symbol pauses with expiration times.
    Used to temporarily halt trading when guards trip or infrastructure fails.
    """

    # Global pause (all symbols)
    _global_pause: PauseRecord | None = None

    # Per-symbol pauses
    _symbol_pauses: dict[str, PauseRecord] = field(default_factory=dict)

    # Slippage failure counts per symbol (for pause-after-N logic)
    _slippage_failures: dict[str, int] = field(default_factory=dict)

    # Broker failure count (for consecutive failure tracking)
    _broker_failures: int = 0

    def pause_all(self, seconds: int, reason: str, allow_reduce_only: bool = True) -> None:
        """
        Pause all trading for a duration.

        Args:
            seconds: Duration of pause in seconds.
            reason: Human-readable reason for the pause.
            allow_reduce_only: Whether to allow reduce-only orders during pause.
        """
        until = time.time() + seconds
        self._global_pause = PauseRecord(
            until=until,
            reason=reason,
            allow_reduce_only=allow_reduce_only,
        )
        logger.warning(
            "Trading paused globally",
            reason=reason,
            duration_seconds=seconds,
            allow_reduce_only=allow_reduce_only,
            operation="degradation",
            stage="pause_all",
        )

    def pause_symbol(
        self, symbol: str, seconds: int, reason: str, allow_reduce_only: bool = True
    ) -> None:
        """
        Pause trading for a specific symbol.

        Args:
            symbol: Trading symbol to pause.
            seconds: Duration of pause in seconds.
            reason: Human-readable reason for the pause.
            allow_reduce_only: Whether to allow reduce-only orders during pause.
        """
        until = time.time() + seconds
        self._symbol_pauses[symbol] = PauseRecord(
            until=until,
            reason=reason,
            allow_reduce_only=allow_reduce_only,
        )
        logger.warning(
            "Trading paused for symbol",
            symbol=symbol,
            reason=reason,
            duration_seconds=seconds,
            allow_reduce_only=allow_reduce_only,
            operation="degradation",
            stage="pause_symbol",
        )

    def is_paused(self, symbol: str | None = None, is_reduce_only: bool = False) -> bool:
        """
        Check if trading is paused.

        Args:
            symbol: Optional symbol to check. If None, checks global pause only.
            is_reduce_only: Whether the pending order is a reduce-only order.

        Returns:
            True if trading is paused and should be blocked.
        """
        now = time.time()

        # Check global pause
        if self._global_pause is not None:
            if now < self._global_pause.until:
                # Still paused - check if reduce-only is allowed
                if is_reduce_only and self._global_pause.allow_reduce_only:
                    return False  # Allow reduce-only through
                return True
            else:
                # Pause expired - clear it
                logger.info(
                    "Global pause expired",
                    reason=self._global_pause.reason,
                    operation="degradation",
                    stage="expire_global",
                )
                self._global_pause = None

        # Check symbol-specific pause
        if symbol is not None and symbol in self._symbol_pauses:
            pause = self._symbol_pauses[symbol]
            if now < pause.until:
                # Still paused - check if reduce-only is allowed
                if is_reduce_only and pause.allow_reduce_only:
                    return False  # Allow reduce-only through
                return True
            else:
                # Pause expired - clear it
                logger.info(
                    "Symbol pause expired",
                    symbol=symbol,
                    reason=pause.reason,
                    operation="degradation",
                    stage="expire_symbol",
                )
                del self._symbol_pauses[symbol]

        return False

    def get_pause_reason(self, symbol: str | None = None) -> str | None:
        """
        Get the reason for the current pause.

        Args:
            symbol: Optional symbol to check.

        Returns:
            Pause reason string, or None if not paused.
        """
        now = time.time()

        if self._global_pause is not None and now < self._global_pause.until:
            return self._global_pause.reason

        if symbol is not None and symbol in self._symbol_pauses:
            pause = self._symbol_pauses[symbol]
            if now < pause.until:
                return pause.reason

        return None

    def record_slippage_failure(self, symbol: str, config: RiskConfig) -> bool:
        """
        Record a slippage guard failure for a symbol.

        Args:
            symbol: Trading symbol.
            config: Risk configuration with pause thresholds.

        Returns:
            True if the symbol was paused due to exceeding threshold.
        """
        self._slippage_failures[symbol] = self._slippage_failures.get(symbol, 0) + 1
        count = self._slippage_failures[symbol]

        if count >= config.slippage_failure_pause_after:
            self.pause_symbol(
                symbol=symbol,
                seconds=config.slippage_pause_seconds,
                reason=f"slippage_failures:{count}",
                allow_reduce_only=True,
            )
            # Reset counter after pause
            self._slippage_failures[symbol] = 0
            return True
        return False

    def reset_slippage_failures(self, symbol: str) -> None:
        """Reset slippage failure counter for a symbol on successful order."""
        self._slippage_failures[symbol] = 0

    def record_broker_failure(self, config: RiskConfig) -> bool:
        """
        Record a broker read failure.

        Args:
            config: Risk configuration with outage thresholds.

        Returns:
            True if trading was paused due to exceeding threshold.
        """
        self._broker_failures += 1

        if self._broker_failures >= config.broker_outage_max_failures:
            self.pause_all(
                seconds=config.broker_outage_cooldown_seconds,
                reason=f"broker_outage:{self._broker_failures}",
                allow_reduce_only=True,
            )
            # Reset counter after pause
            self._broker_failures = 0
            return True
        return False

    def reset_broker_failures(self) -> None:
        """Reset broker failure counter on successful read."""
        self._broker_failures = 0

    def clear_all(self) -> None:
        """Clear all pause states. Used for testing or manual override."""
        self._global_pause = None
        self._symbol_pauses.clear()
        self._slippage_failures.clear()
        self._broker_failures = 0
        logger.info(
            "All degradation state cleared",
            operation="degradation",
            stage="clear_all",
        )

    def get_status(self) -> dict[str, object]:
        """
        Get current degradation status for monitoring/TUI.

        Returns:
            Dictionary with pause states and failure counts.
        """
        now = time.time()
        return {
            "global_paused": self._global_pause is not None and now < self._global_pause.until,
            "global_reason": self._global_pause.reason if self._global_pause else None,
            "global_remaining_seconds": (
                max(0, int(self._global_pause.until - now)) if self._global_pause else 0
            ),
            "paused_symbols": {
                symbol: {
                    "reason": pause.reason,
                    "remaining_seconds": max(0, int(pause.until - now)),
                    "allow_reduce_only": pause.allow_reduce_only,
                }
                for symbol, pause in self._symbol_pauses.items()
                if now < pause.until
            },
            "slippage_failures": dict(self._slippage_failures),
            "broker_failures": self._broker_failures,
        }


__all__ = ["DegradationState", "PauseRecord"]
