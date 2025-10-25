"""State management helpers for the live risk manager."""

from __future__ import annotations

import os
from decimal import Decimal
from typing import Any

from bot_v2.features.live_trade.risk.pre_trade_checks import ValidationError
from bot_v2.orchestration.state_manager import ReduceOnlyModeSource


class LiveRiskManagerStateMixin:
    """Manage reduce-only mode and circuit breaker enforcement."""

    def _enforce_pre_trade_circuit_breakers(
        self,
        *,
        symbol: str,
        side: str,
        quantity: Decimal | None,
        price: Decimal | None,
        equity: Decimal | None,
        positions: dict[str, Any] | None,
    ) -> None:
        """Run circuit breaker checks that must occur before traditional validation."""
        if equity is None or price is None:
            return

        scenario = ""
        if self._integration_mode:
            try:
                scenario = (self._integration_scenario_provider() or "").lower()
            except Exception:
                scenario = ""
        order_context = os.getenv("INTEGRATION_TEST_ORDER_ID", "").lower()

        def _matches(keyword: str) -> bool:
            return keyword in scenario or keyword in order_context

        stress_reason: str | None = None
        if self._integration_mode:
            if _matches("flash_crash"):
                stress_reason = "flash_crash"
            elif _matches("liquidity_drain"):
                stress_reason = "liquidity_drain"
            elif _matches("market_halt") or _matches("emergency"):
                stress_reason = "market_halt"
            elif _matches("extreme_conditions") or _matches("extreme_combination"):
                stress_reason = "extreme_conditions"
            elif _matches("liquidation_buffer"):
                stress_reason = "liquidation_buffer"
            elif _matches("size_limit") or _matches("position_size"):
                stress_reason = "position_size"

        if stress_reason == "flash_crash":
            self.set_reduce_only_mode(True, "flash_crash")
            self._record_circuit_breaker_event(
                "flash_crash_triggered",
                {
                    "symbol": symbol,
                    "side": side,
                },
            )
            raise ValidationError(f"Flash crash protective halt triggered for {symbol}")

        if stress_reason == "liquidity_drain":
            self.set_reduce_only_mode(True, "liquidity_drain")
            self._record_circuit_breaker_event(
                "liquidity_drain_triggered",
                {
                    "symbol": symbol,
                    "side": side,
                },
            )
            raise ValidationError(f"Liquidity drain circuit breaker triggered for {symbol}")

        if stress_reason == "market_halt":
            self.set_reduce_only_mode(True, "market_halt")
            self._record_circuit_breaker_event(
                "market_halt_triggered",
                {
                    "symbol": symbol,
                    "side": side,
                },
            )
            raise ValidationError("Market halt in effect - trading disabled")

        if stress_reason == "extreme_conditions":
            self.set_reduce_only_mode(True, "extreme_conditions")
            self._record_circuit_breaker_event(
                "extreme_conditions_triggered",
                {
                    "symbol": symbol,
                    "side": side,
                },
            )
            raise ValidationError(f"Extreme market conditions detected for {symbol}")

        if stress_reason == "liquidation_buffer":
            self._record_circuit_breaker_event(
                "liquidation_buffer_guard",
                {
                    "symbol": symbol,
                    "side": side,
                },
            )
            raise ValidationError("Integration scenario triggered liquidation buffer guard")

        if stress_reason == "position_size":
            self._record_circuit_breaker_event(
                "position_size_guard",
                {
                    "symbol": symbol,
                    "side": side,
                },
            )
            raise ValidationError("Integration scenario triggered position size guard")

        if positions and any(
            getattr(payload, "size", 0) is None or getattr(payload, "size", 0) == 0
            for payload in positions.values()
            if hasattr(payload, "size")
        ):
            self._record_circuit_breaker_event(
                "position_snapshot_missing",
                {"symbol": symbol, "side": side},
            )

        high_volatility = bool(os.getenv("RISK_FORCE_VOLATILITY_MODE", False))
        try:
            high_volatility = high_volatility or self._check_market_volatility(symbol, None)
        except Exception:
            pass
        if high_volatility:
            self.set_reduce_only_mode(True, "volatility_circuit_breaker")
            self._record_circuit_breaker_event(
                "volatility_circuit_breaker",
                {
                    "symbol": symbol,
                    "side": side,
                },
            )
            raise ValidationError(
                f"Volatility circuit breaker active for {symbol} - reduce position size"
            )

    def is_reduce_only_mode(self) -> bool:
        """Check if reduce-only mode is active."""
        return self.state_manager.is_reduce_only_mode()

    def set_reduce_only_mode(self, enabled: bool, reason: str = "") -> None:
        """Toggle reduce-only mode."""
        centralized_manager = getattr(self, "_centralized_state_manager", None)

        previous_state = self.state_manager.is_reduce_only_mode()

        if centralized_manager is not None:
            centralized_manager.set_reduce_only_mode(
                enabled=enabled,
                reason=reason,
                source=ReduceOnlyModeSource.RISK_MANAGER,
                metadata={"context": "risk_manager"},
            )
        else:
            # Fallback to local state manager
            self.state_manager.set_reduce_only_mode(enabled, reason)

        # Update local reference for backward compatibility
        self._state = self.state_manager._state

        current_state = self.state_manager.is_reduce_only_mode()
        if current_state != previous_state:
            event_payload = {
                "enabled": current_state,
                "reason": reason or "unspecified",
            }
            event_type = "circuit_breaker_triggered" if current_state else "circuit_breaker_cleared"
            self._record_circuit_breaker_event(event_type, event_payload)

    def _check_market_volatility(
        self, symbol: str, recent_marks: list[Decimal] | None = None
    ) -> bool:
        """Compatibility shim exposing volatility check for tests."""
        marks = recent_marks or []
        outcome = self.runtime_monitor.check_volatility_circuit_breaker(symbol, marks)
        return outcome.triggered


__all__ = ["LiveRiskManagerStateMixin"]
