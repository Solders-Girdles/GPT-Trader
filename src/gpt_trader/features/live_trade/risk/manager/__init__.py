from collections import defaultdict
from decimal import Decimal
from typing import Any


class ValidationError(Exception):
    pass


class LiveRiskManager:
    def __init__(self, config: Any = None) -> None:
        self.config = config
        self.positions: defaultdict[str, dict[str, Any]] = defaultdict(dict)

    def check_order(self, order: Any) -> bool:
        return True

    def update_position(self, position: Any) -> None:
        pass

    def check_liquidation_buffer(self, symbol: str, position: Any, equity: Decimal) -> bool:
        if not self.config:
            return False

        try:
            # Handle object or dict
            if isinstance(position, dict):
                liq_price = position.get("liquidation_price")
                mark_price = position.get("mark") or position.get("mark_price")
            else:
                liq_price = getattr(position, "liquidation_price", None)
                mark_price = getattr(position, "mark_price", None) or getattr(
                    position, "mark", None
                )

            if not liq_price or not mark_price:
                return False

            mark_price = Decimal(str(mark_price))
            liq_price = Decimal(str(liq_price))

            if mark_price == 0:
                return False

            buffer_pct = abs(mark_price - liq_price) / mark_price

            if buffer_pct < self.config.min_liquidation_buffer_pct:
                self.positions[symbol]["reduce_only"] = True
                return True

        except (AttributeError, TypeError, ZeroDivisionError, ValueError):
            pass

        return False

    def pre_trade_validate(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        price: Decimal,
        product: Any,
        equity: Decimal,
        current_positions: dict[str, Any],
    ) -> None:
        if self.config:
            notional = quantity * price
            if equity > 0:
                leverage = notional / equity
                if leverage > self.config.max_leverage:
                    raise ValidationError(
                        f"Leverage {leverage} exceeds max {self.config.max_leverage}"
                    )
