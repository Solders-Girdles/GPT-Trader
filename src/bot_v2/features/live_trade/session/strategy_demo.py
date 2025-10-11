"""Demo strategy runner used by the legacy live-trade facade."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from bot_v2.features.live_trade.strategies.perps_baseline import (
    Action,
    BaselinePerpsStrategy,
    Decision,
)
from bot_v2.utilities.quantities import quantity_from

from . import account as account_ops
from .registry import (
    get_broker_client,
    get_connection,
    get_risk_manager,
)

logger = logging.getLogger(__name__)


@dataclass
class AccountContext:
    equity: Decimal
    position_map: dict[str, Any]


@dataclass
class SymbolContext:
    symbol: str
    current_mark: Decimal
    equity: Decimal
    position_state: dict[str, Any] | None
    recent_marks: list[Decimal] | None
    product: Any


def run_strategy(  # pragma: no cover - legacy demo path
    symbols: list[str],
    strategy_name: str = "baseline_perps",
    iterations: int = 3,
    mark_cache: dict[str, Decimal] | None = None,
    mark_windows: dict[str, list[Decimal]] | None = None,
    *,
    strategy_override: Any | None = None,
) -> dict[str, Decision]:
    """Run the legacy baseline strategy for demonstration purposes."""

    broker = get_broker_client()
    connection = get_connection()
    risk_manager = get_risk_manager()
    if not broker or not connection or not connection.is_connected or not risk_manager:
        raise RuntimeError("Broker connection not initialized")

    strategy = strategy_override or BaselinePerpsStrategy(
        environment="simulated", risk_manager=risk_manager
    )

    decisions_accumulator: dict[str, Decision] = {}

    for _ in range(iterations):
        account_info = account_ops.get_account()
        if not account_info:
            break

        account_ctx = AccountContext(
            equity=Decimal(str(account_info.equity)),
            position_map={pos.symbol: pos for pos in broker.get_positions()},
        )

        for symbol in symbols:
            try:
                symbol_ctx = _prepare_symbol_context(
                    symbol=symbol,
                    account_context=account_ctx,
                    mark_cache=mark_cache,
                    mark_windows=mark_windows,
                )

                if symbol_ctx is None:
                    continue

                decision = _generate_strategy_decision(strategy, symbol_ctx)
                decisions_accumulator[symbol] = decision
                _execute_decision_if_actionable(decision, symbol_ctx)
            except Exception as exc:  # pragma: no cover - demo resilience
                logger.error("Error processing %s: %s", symbol, exc)

    return decisions_accumulator


def _prepare_symbol_context(
    *,
    symbol: str,
    account_context: AccountContext,
    mark_cache: dict[str, Decimal] | None,
    mark_windows: dict[str, list[Decimal]] | None,
) -> SymbolContext | None:
    current_mark = None
    if mark_cache and symbol in mark_cache:
        current_mark = mark_cache[symbol]
    else:
        quote = account_ops.get_quote(symbol)
        if quote:
            current_mark = Decimal(str(quote.last))

    if not current_mark:
        logger.warning("No mark price for %s", symbol)
        return None

    position_state = None
    if symbol in account_context.position_map:
        position = account_context.position_map[symbol]
        resolved_quantity = _quantity_from_position(position)
        position_state = {
            "quantity": Decimal(str(resolved_quantity)),
            "side": position.side,
            "entry": Decimal(str(position.entry_price)),
        }

    recent_marks = mark_windows.get(symbol) if mark_windows else None
    product = _build_template_product(symbol)

    return SymbolContext(
        symbol=symbol,
        current_mark=current_mark,
        equity=account_context.equity,
        position_state=position_state,
        recent_marks=recent_marks,
        product=product,
    )


def _generate_strategy_decision(
    strategy: BaselinePerpsStrategy, context: SymbolContext
) -> Decision:
    decision = strategy.decide(
        symbol=context.symbol,
        current_mark=context.current_mark,
        position_state=context.position_state,
        recent_marks=context.recent_marks,
        equity=context.equity,
        product=context.product,
    )
    logger.info("%s decision: %s - %s", context.symbol, decision.action.value, decision.reason)
    return decision


def _execute_decision_if_actionable(decision: Decision, context: SymbolContext) -> None:
    symbol = context.symbol
    current_mark = context.current_mark
    product = context.product

    if decision.action in (Action.BUY, Action.SELL):
        side = "buy" if decision.action == Action.BUY else "sell"
        quantity = decision.quantity
        if quantity is None and decision.target_notional:
            quantity = decision.target_notional / current_mark

        if quantity is not None:
            from bot_v2.features.brokerages.coinbase.utilities import enforce_perp_rules

            quantized_quantity, _ = enforce_perp_rules(
                product=product,
                quantity=quantity,
                price=current_mark,
            )

            logger.info("Would place %s order: %s %s @ market", side, quantized_quantity, symbol)

    elif decision.action == Action.CLOSE and context.position_state:
        close_side = "sell" if context.position_state.get("side") == "long" else "buy"
        quantity = decision.quantity
        if quantity is None:
            quantity = _quantity_from_position(context.position_state)

        if quantity is not None:
            from bot_v2.features.brokerages.coinbase.utilities import enforce_perp_rules

            quantized_quantity, _ = enforce_perp_rules(
                product=product,
                quantity=quantity,
                price=current_mark,
            )
            logger.info(
                "Would place reduce-only %s order: %s %s", close_side, quantized_quantity, symbol
            )
        else:
            logger.info("Skipping close for %s; no quantity resolved from decision/state", symbol)


def _quantity_from_position(position: Any | None) -> Decimal:
    resolved = quantity_from(position, default=Decimal("0"))
    return resolved if resolved is not None else Decimal("0")


def _build_template_product(symbol: str) -> Any:
    from bot_v2.features.brokerages.core.interfaces import MarketType, Product

    base_asset = symbol.split("-")[0]
    return Product(
        symbol=symbol,
        base_asset=base_asset,
        quote_asset="USD",
        market_type=MarketType.PERPETUAL,
        step_size=Decimal("0.001"),
        min_size=Decimal("0.001"),
        price_increment=Decimal("0.01"),
        min_notional=Decimal("10"),
    )


__all__ = ["run_strategy"]
