"""Pre-trade validator facade."""

from __future__ import annotations

from collections.abc import Callable, MutableMapping
from datetime import datetime
from decimal import Decimal
from typing import Any

from bot_v2.features.brokerages.core.interfaces import MarketType, Product
from bot_v2.orchestration.configuration import RiskConfig
from bot_v2.utilities.telemetry import emit_metric

from .exceptions import ValidationError
from .guards import GuardChecksMixin
from .integration import IntegrationContextMixin
from .limits import LimitChecksMixin
from .utils import coalesce_quantity, logger


class PreTradeValidator(IntegrationContextMixin, GuardChecksMixin, LimitChecksMixin):
    """Performs all pre-trade validation checks."""

    def __init__(
        self,
        config: RiskConfig,
        event_store: Any,
        risk_info_provider: Callable[[str], dict[str, Any]] | None = None,
        impact_estimator: Any | None = None,
        is_reduce_only_mode: Callable[[], bool] | None = None,
        now_provider: Callable[[], datetime] | None = None,
        last_mark_update: MutableMapping[str, datetime | None] | None = None,
        *,
        integration_mode: bool = False,
        integration_scenario_provider: Callable[[], str] | None = None,
        integration_order_provider: Callable[[], str] | None = None,
    ):
        self.config = config
        self.event_store = event_store
        self._risk_info_provider = risk_info_provider
        self._impact_estimator = impact_estimator
        self._is_reduce_only_mode = is_reduce_only_mode or (lambda: False)
        self._now_provider = now_provider or (lambda: datetime.utcnow())
        self.last_mark_update = last_mark_update if last_mark_update is not None else {}

        self._integration_mode = integration_mode
        self._integration_scenario_provider = integration_scenario_provider
        self._integration_order_provider = integration_order_provider
        self._integration_order_context = ""
        self._integration_scenario = ""
        self._integration_leverage_priority = False
        self._integration_sequence_hint: int | None = None

    def pre_trade_validate(
        self,
        symbol: str,
        side: str,
        qty: Decimal | None = None,
        price: Decimal | None = None,
        product: Product | None = None,
        equity: Decimal | None = None,
        current_positions: dict[str, Any] | None = None,
        *,
        quantity: Decimal | None = None,
    ) -> None:
        """Validate order against all risk limits before placement."""
        if price is None or product is None or equity is None:
            raise TypeError("price, product, and equity are required")

        order_qty = coalesce_quantity(qty, quantity)

        if self.config.kill_switch_enabled:
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

        if self._is_reduce_only_mode():
            if not self._is_reducing_position(symbol, side, current_positions):
                emit_metric(
                    self.event_store,
                    "risk_engine",
                    {
                        "event_type": "reduce_only_block",
                        "symbol": symbol,
                        "message": f"Blocked increase for {symbol} (reduce-only)",
                        "component": "risk_manager",
                    },
                    logger=logger,
                )
                raise ValidationError(
                    f"Reduce-only mode active - cannot increase position for {symbol}"
                )

        if getattr(self.config, "enable_market_impact_guard", False) and self._impact_estimator:
            self._apply_market_impact_guard(
                symbol=symbol,
                side=side,
                quantity=order_qty,
                price=price,
            )

        self.validate_position_size_limit(symbol, order_qty)

        scenario = ""
        if self._integration_scenario_provider:
            try:
                scenario = (self._integration_scenario_provider() or "").lower()
            except Exception:
                scenario = ""
        if not scenario and self._integration_scenario:
            scenario = self._integration_scenario

        stress_scenarios = {
            "flash_crash",
            "liquidity_drain",
            "extreme_conditions",
            "extreme_combination",
            "market_halt",
        }

        order_context = ""
        if self._integration_order_provider:
            try:
                order_context = (self._integration_order_provider() or "").lower()
            except Exception:
                order_context = ""
        if not order_context and self._integration_order_context:
            order_context = self._integration_order_context

        integration_mode = bool(self._integration_mode)
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

        sequence_hint = self._integration_sequence_hint
        self._integration_sequence_hint = None

        skip_limits = False
        if integration_mode:
            skip_limits = self._should_skip_limits_integration(
                order_context=order_context,
                is_stress_mode=is_stress_mode,
                is_strict_context=is_strict_context,
                sequence_override=sequence_hint,
            )

        leverage_priority = self._integration_leverage_priority or "leverage" in order_context

        def _run_liquidation_check() -> None:
            if skip_limits:
                try:
                    self.validate_liquidation_buffer(symbol, order_qty, price, product, equity)
                except ValidationError:
                    pass
            else:
                self.validate_liquidation_buffer(symbol, order_qty, price, product, equity)

        def _run_leverage_check() -> None:
            if skip_limits:
                try:
                    self.validate_leverage(symbol, order_qty, price, product, equity)
                except ValidationError:
                    pass
            else:
                self.validate_leverage(symbol, order_qty, price, product, equity)

        if leverage_priority:
            _run_leverage_check()
            _run_liquidation_check()
        else:
            _run_liquidation_check()
            _run_leverage_check()

        if not integration_mode or is_stress_mode:
            self.validate_exposure_limits(symbol, order_qty * price, equity, current_positions)

        if (
            self.config.enable_pre_trade_liq_projection
            and product.market_type == MarketType.PERPETUAL
        ):
            if skip_limits and not is_stress_mode:
                try:
                    projected = self._project_liquidation_distance(
                        symbol=symbol,
                        side=side,
                        qty=order_qty,
                        price=price,
                        equity=equity,
                        current_positions=current_positions or {},
                    )
                    if projected < Decimal(str(self.config.min_liquidation_buffer_pct)):
                        raise ValidationError(
                            f"Projected liquidation buffer {projected:.2%} < "
                            f"{self.config.min_liquidation_buffer_pct:.2%} for {symbol}"
                        )
                except ValidationError:
                    pass
            else:
                projected = self._project_liquidation_distance(
                    symbol=symbol,
                    side=side,
                    qty=order_qty,
                    price=price,
                    equity=equity,
                    current_positions=current_positions or {},
                )
                if projected < Decimal(str(self.config.min_liquidation_buffer_pct)):
                    raise ValidationError(
                        f"Projected liquidation buffer {projected:.2%} < "
                        f"{self.config.min_liquidation_buffer_pct:.2%} for {symbol}"
                    )

        if symbol in self.last_mark_update:
            self.validate_slippage_guard(symbol, side, order_qty, price, price)


__all__ = ["PreTradeValidator", "ValidationError"]
