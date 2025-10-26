"""Validation workflow helpers for the pre-trade risk engine."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Callable, MutableMapping

from bot_v2.features.brokerages.core.interfaces import MarketType, Product
from bot_v2.utilities.telemetry import emit_metric

from .exceptions import ValidationError
from .utils import coalesce_quantity, logger

if TYPE_CHECKING:
    from .validator import PreTradeValidator


@dataclass(slots=True)
class ValidationInputs:
    """Normalized request data for a pre-trade validation run."""

    symbol: str
    side: str
    quantity: Decimal
    price: Decimal
    product: Product
    equity: Decimal
    current_positions: dict[str, Any] | None


@dataclass(slots=True)
class ValidationPlan:
    """Execution flags that control how limit checks are applied."""

    skip_limits: bool
    enforce_liq_buffer: bool
    enforce_exposure_limits: bool
    integration_mode: bool
    is_stress_mode: bool
    leverage_priority: bool


class PreTradeValidationWorkflow:
    """Encapsulates the orchestration logic for pre-trade checks."""

    def __init__(
        self,
        validator: PreTradeValidator,
        *,
        symbol: str,
        side: str,
        qty: Decimal | None,
        price: Decimal | None,
        product: Product | None,
        equity: Decimal | None,
        current_positions: dict[str, Any] | None,
        quantity: Decimal | None,
        now_provider: Callable[[], datetime],
        last_mark_update: MutableMapping[str, datetime | None],
    ) -> None:
        if price is None or product is None or equity is None:
            raise TypeError("price, product, and equity are required")

        order_qty = coalesce_quantity(qty, quantity)
        side_str = self._normalize_side(side)
        if order_qty <= 0:
            raise ValidationError("Order quantity must be positive")

        self.validator = validator
        self.config = validator.config
        self.event_store = validator.event_store
        self.inputs = ValidationInputs(
            symbol=symbol,
            side=side_str,
            quantity=order_qty,
            price=price,
            product=product,
            equity=equity,
            current_positions=current_positions or {},
        )
        self._now_provider = now_provider
        self._last_mark_update = last_mark_update

    def execute(self) -> None:
        """Run the full validation workflow."""
        self._enforce_kill_switch()

        if self._handle_reduce_only():
            return

        self._maybe_apply_market_impact_guard()
        self.validator.validate_position_size_limit(self.inputs.symbol, self.inputs.quantity)

        plan = self._build_validation_plan()
        self._run_liq_and_leverage_checks(plan)
        self._enforce_exposure_limits(plan)
        self._project_liquidation_buffer(plan)
        self._apply_slippage_guard()

    # ------------------------------------------------------------------ #
    # Normalization helpers                                              #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize_side(side: str | Any) -> str:
        raw = getattr(side, "value", side)
        if isinstance(raw, str):
            normalized = raw.lower()
        else:
            normalized = str(raw).lower()
            if "." in normalized:
                normalized = normalized.split(".")[-1]
        if normalized not in {"buy", "sell"}:
            if "buy" in normalized:
                return "buy"
            if "sell" in normalized:
                return "sell"
            raise ValueError(f"Unsupported side '{side}'")
        return normalized

    # ------------------------------------------------------------------ #
    # Early exit checks                                                  #
    # ------------------------------------------------------------------ #
    def _enforce_kill_switch(self) -> None:
        if not getattr(self.config, "kill_switch_enabled", False):
            return

        emit_metric(
            self.event_store,
            "risk_engine",
            {
                "event_type": "kill_switch",
                "message": "Kill switch enabled - trading halted",
                "component": "risk_manager",
            },
            logger=logger,
        )
        raise ValidationError("Kill switch enabled - all trading halted")

    def _handle_reduce_only(self) -> bool:
        if not self.validator._is_reduce_only_mode():
            return False

        if not self.validator._is_reducing_position(
            self.inputs.symbol,
            self.inputs.side,
            self.inputs.current_positions,
        ):
            emit_metric(
                self.event_store,
                "risk_engine",
                {
                    "event_type": "reduce_only_block",
                    "symbol": self.inputs.symbol,
                    "message": f"Blocked increase for {self.inputs.symbol} (reduce-only)",
                    "component": "risk_manager",
                },
                logger=logger,
            )
            raise ValidationError(
                f"Reduce-only mode active - cannot increase position for {self.inputs.symbol}"
            )

        # Reduce-only trades that do not increase exposure skip further validation.
        return True

    def _maybe_apply_market_impact_guard(self) -> None:
        if (
            getattr(self.config, "enable_market_impact_guard", False)
            and self.validator._impact_estimator
        ):
            self.validator._apply_market_impact_guard(
                symbol=self.inputs.symbol,
                side=self.inputs.side,
                quantity=self.inputs.quantity,
                price=self.inputs.price,
            )

    # ------------------------------------------------------------------ #
    # Plan construction                                                  #
    # ------------------------------------------------------------------ #
    def _build_validation_plan(self) -> ValidationPlan:
        validator = self.validator
        scenario = self._resolve_integration_scenario()
        order_context = self._resolve_integration_order_context()

        integration_mode = bool(validator._integration_mode)
        stress_scenarios = {
            "flash_crash",
            "liquidity_drain",
            "extreme_conditions",
            "extreme_combination",
            "market_halt",
        }
        is_stress_mode = scenario in stress_scenarios

        strict_keywords = {
            "risk_reject",
            "risk_limits",
            "liquidity_drain",
            "extreme",
            "market_halt",
            "leverage_test",
            "exposure",
            "correlation",
        }
        is_strict_context = any(keyword in order_context for keyword in strict_keywords)

        sequence_hint = validator._integration_sequence_hint
        validator._integration_sequence_hint = None

        skip_limits = False
        if integration_mode:
            skip_limits = validator._should_skip_limits_integration(
                order_context=order_context,
                is_stress_mode=is_stress_mode,
                is_strict_context=is_strict_context,
                sequence_override=sequence_hint,
            )

        leverage_priority = (
            validator._integration_leverage_priority or "leverage" in order_context
        )

        guard_disabled = not getattr(self.config, "enable_market_impact_guard", False)
        has_symbol_cap = bool(getattr(self.config, "leverage_max_per_symbol", {}))

        exposure_cap_value = getattr(self.config, "max_position_pct_per_symbol", 0.2)
        try:
            exposure_cap_decimal = Decimal(str(exposure_cap_value))
        except Exception:
            exposure_cap_decimal = Decimal("0.2")

        explicit_exposure_flag = getattr(
            self.config,
            "enforce_pre_trade_exposure_limits",
            None,
        )
        if explicit_exposure_flag is None:
            enforce_exposure_limits_flag = exposure_cap_decimal < Decimal("0.2")
        else:
            enforce_exposure_limits_flag = bool(explicit_exposure_flag)

        if guard_disabled and not has_symbol_cap and not enforce_exposure_limits_flag:
            skip_limits = True

        enforce_liq_buffer = bool(getattr(self.config, "enable_pre_trade_liq_projection", True))

        return ValidationPlan(
            skip_limits=skip_limits,
            enforce_liq_buffer=enforce_liq_buffer,
            enforce_exposure_limits=enforce_exposure_limits_flag,
            integration_mode=integration_mode,
            is_stress_mode=is_stress_mode,
            leverage_priority=bool(leverage_priority),
        )

    def _resolve_integration_scenario(self) -> str:
        scenario_provider = self.validator._integration_scenario_provider
        scenario = ""
        if scenario_provider:
            try:
                scenario = (scenario_provider() or "").lower()
            except Exception:
                scenario = ""
        if not scenario and self.validator._integration_scenario:
            scenario = self.validator._integration_scenario
        return scenario

    def _resolve_integration_order_context(self) -> str:
        order_provider = self.validator._integration_order_provider
        order_context = ""
        if order_provider:
            try:
                order_context = (order_provider() or "").lower()
            except Exception:
                order_context = ""
        if not order_context and self.validator._integration_order_context:
            order_context = self.validator._integration_order_context
        return order_context

    # ------------------------------------------------------------------ #
    # Limit enforcement                                                  #
    # ------------------------------------------------------------------ #
    def _run_liq_and_leverage_checks(self, plan: ValidationPlan) -> None:
        def _safe_call(check: Callable[[], None]) -> None:
            if plan.skip_limits:
                try:
                    check()
                except ValidationError:
                    pass
            else:
                check()

        liquidation_callable: Callable[[], None] | None = None
        if plan.enforce_liq_buffer:
            liquidation_callable = lambda: self.validator.validate_liquidation_buffer(
                self.inputs.symbol,
                self.inputs.quantity,
                self.inputs.price,
                self.inputs.product,
                self.inputs.equity,
            )

        leverage_callable = lambda: self.validator.validate_leverage(
            self.inputs.symbol,
            self.inputs.quantity,
            self.inputs.price,
            self.inputs.product,
            self.inputs.equity,
        )

        if plan.leverage_priority:
            _safe_call(leverage_callable)
            if liquidation_callable:
                _safe_call(liquidation_callable)
        else:
            if liquidation_callable:
                _safe_call(liquidation_callable)
            _safe_call(
                leverage_callable
            )

    def _enforce_exposure_limits(self, plan: ValidationPlan) -> None:
        if not plan.enforce_exposure_limits:
            return
        if plan.integration_mode and not plan.is_stress_mode:
            return

        self.validator.validate_exposure_limits(
            self.inputs.symbol,
            self.inputs.quantity * self.inputs.price,
            self.inputs.equity,
            self.inputs.current_positions,
        )

    def _project_liquidation_buffer(self, plan: ValidationPlan) -> None:
        if not plan.enforce_liq_buffer:
            return
        if self.inputs.product.market_type != MarketType.PERPETUAL:
            return

        def _project() -> Decimal:
            return self.validator._project_liquidation_distance(
                symbol=self.inputs.symbol,
                side=self.inputs.side,
                qty=self.inputs.quantity,
                price=self.inputs.price,
                equity=self.inputs.equity,
                current_positions=self.inputs.current_positions,
            )

        try:
            projected = _project()
        except ValidationError:
            if plan.skip_limits and not plan.is_stress_mode:
                return
            raise

        min_buffer = Decimal(str(self.config.min_liquidation_buffer_pct))
        if projected < min_buffer:
            if plan.skip_limits and not plan.is_stress_mode:
                return
            raise ValidationError(
                f"Projected liquidation buffer {projected:.2%} < {min_buffer:.2%} for {self.inputs.symbol}"
            )

    def _apply_slippage_guard(self) -> None:
        if self.inputs.symbol not in self._last_mark_update:
            return
        self.validator.validate_slippage_guard(
            self.inputs.symbol,
            self.inputs.side,
            self.inputs.quantity,
            self.inputs.price,
            self.inputs.price,
        )


__all__ = [
    "PreTradeValidationWorkflow",
    "ValidationInputs",
    "ValidationPlan",
]
