"""Decision execution mixin leveraging modular workflow helpers."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from bot_v2.features.brokerages.core.interfaces import Product

from . import workflow


class DecisionExecutionMixin:
    """Execute strategy decisions through the execution coordinator."""

    async def execute_decision(
        self,
        symbol: str,
        decision: Any,
        mark: Decimal,
        product: Product,
        position_state: dict[str, Any] | None,
    ) -> None:
        await workflow.execute_decision(
            self,
            symbol=symbol,
            decision=decision,
            mark=mark,
            product=product,
            position_state=position_state,
        )


__all__ = ["DecisionExecutionMixin"]
