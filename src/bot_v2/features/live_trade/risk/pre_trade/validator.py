"""Pre-trade validator facade."""

from __future__ import annotations

from collections.abc import Callable, MutableMapping
from datetime import datetime
from decimal import Decimal
from typing import Any, TYPE_CHECKING

from bot_v2.features.brokerages.core.interfaces import Product

if TYPE_CHECKING:
    from bot_v2.orchestration.configuration import RiskConfig

from .exceptions import ValidationError
from .guards import GuardChecksMixin
from .integration import IntegrationContextMixin
from .limits import LimitChecksMixin
from .workflow import PreTradeValidationWorkflow


class PreTradeValidator(IntegrationContextMixin, GuardChecksMixin, LimitChecksMixin):
    """Performs all pre-trade validation checks."""

    def __init__(
        self,
        config: "RiskConfig",
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
        quantity: Decimal | None = None,
        price: Decimal | None = None,
        product: Product | None = None,
        equity: Decimal | None = None,
        current_positions: dict[str, Any] | None = None,
    ) -> None:
        """Validate order against all risk limits before placement."""
        workflow = PreTradeValidationWorkflow(
            self,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            product=product,
            equity=equity,
            current_positions=current_positions,
            now_provider=self._now_provider,
            last_mark_update=self.last_mark_update,
        )
        workflow.execute()


__all__ = ["PreTradeValidator", "ValidationError"]
