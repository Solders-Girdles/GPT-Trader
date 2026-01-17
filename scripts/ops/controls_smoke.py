#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

from gpt_trader.app.config import BotConfig, BotRiskConfig
from gpt_trader.app.container import (
    ApplicationContainer,
    clear_application_container,
    set_application_container,
)
from gpt_trader.features.live_trade.engines.base import CoordinatorContext
from gpt_trader.features.live_trade.engines.strategy import TradingEngine
from gpt_trader.features.live_trade.execution.submission_result import OrderSubmissionStatus
from gpt_trader.features.live_trade.strategies.perps_baseline import Action, Decision

OUTPUT_DIR = Path("var/ops")


@dataclass
class SmokeResult:
    name: str
    status: str
    detail: dict[str, Any]


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _build_engine() -> TradingEngine:
    risk = BotRiskConfig(position_fraction=Decimal("0.1"))
    config = BotConfig(symbols=["BTC-USD"], interval=1, risk=risk)
    config.profile = "dev"
    container = ApplicationContainer(config)
    set_application_container(container)
    risk_manager = MagicMock()
    risk_manager.check_mark_staleness.return_value = False
    risk_manager.is_reduce_only_mode.return_value = False
    risk_manager.config = MagicMock()
    risk_manager.config.kill_switch_enabled = False
    risk_manager.check_order.return_value = True
    risk_manager._daily_pnl_triggered = False

    broker = MagicMock()
    broker.get_market_snapshot.return_value = {"spread_bps": 5, "depth_l1": 10000}
    broker.list_balances.return_value = []
    broker.list_positions.return_value = []

    context = CoordinatorContext(
        config=config,
        broker=broker,
        risk_manager=risk_manager,
        event_store=container.event_store,
        orders_store=container.orders_store,
        container=container,
    )
    engine = TradingEngine(context)

    engine._state_collector = MagicMock()
    engine._state_collector.require_product.return_value = MagicMock()
    engine._state_collector.resolve_effective_price.return_value = Decimal("50000")
    engine._state_collector.build_positions_dict.return_value = {}

    engine._guard_manager = MagicMock()
    engine._guard_manager.run_runtime_guards.return_value = None
    engine._guard_manager._api_error_window = []
    engine._guard_manager._api_error_window_seconds = 60
    engine._guard_manager._api_error_rate_threshold = 0.5
    engine._guard_manager._api_rate_limit_usage_threshold = 0.9

    engine._order_validator = MagicMock()
    engine._order_validator.validate_exchange_rules.return_value = (Decimal("1"), None)
    engine._order_validator.enforce_slippage_guard.return_value = None
    engine._order_validator.ensure_mark_is_fresh.return_value = None
    engine._order_validator.run_pre_trade_validation.return_value = None
    engine._order_validator.maybe_preview_order.return_value = None
    engine._order_validator.finalize_reduce_only_flag.return_value = False
    engine._order_validator.enable_order_preview = True

    failure_tracker = MagicMock()
    failure_tracker.get_failure_count.return_value = 0

    security_result = MagicMock()
    security_result.is_valid = True
    security_result.errors = []
    engine._order_validator.validate_security_constraints.return_value = security_result

    engine._order_submitter = MagicMock()
    engine._order_submitter.submit_order.return_value = "order-1"
    engine._order_submitter.generate_client_order_id.return_value = "decision-1"
    return engine


def _run_kill_switch(engine: TradingEngine) -> SmokeResult:
    risk_manager = cast(MagicMock, engine.context.risk_manager)
    risk_manager.config.kill_switch_enabled = True
    decision = Decision(Action.BUY, "ops controls smoke")

    result = asyncio.run(
        engine._validate_and_place_order(
            symbol="BTC-USD",
            decision=decision,
            price=Decimal("50000"),
            equity=Decimal("10000"),
            quantity_override=Decimal("1"),
        )
    )

    detail = {
        "status": result.status.value,
        "reason": str(result.reason) if result.reason is not None else None,
        "decision_id": result.decision_trace.decision_id if result.decision_trace else None,
    }
    if (
        result.status == OrderSubmissionStatus.BLOCKED
        and result.reason == "kill_switch"
        and detail["decision_id"]
    ):
        return SmokeResult("kill_switch", "passed", detail)
    return SmokeResult("kill_switch", "failed", detail)


def _run_reduce_only_blocks_entry(engine: TradingEngine) -> SmokeResult:
    risk_manager = cast(MagicMock, engine.context.risk_manager)
    risk_manager.config.kill_switch_enabled = False
    risk_manager.is_reduce_only_mode.return_value = True
    risk_manager.check_order.return_value = False
    risk_manager._daily_pnl_triggered = False
    engine._current_positions = {}

    decision = Decision(Action.BUY, "ops controls smoke")
    result = asyncio.run(
        engine._validate_and_place_order(
            symbol="BTC-USD",
            decision=decision,
            price=Decimal("50000"),
            equity=Decimal("10000"),
            quantity_override=Decimal("1"),
        )
    )

    detail = {
        "status": result.status.value,
        "reason": str(result.reason) if result.reason is not None else None,
    }
    if result.status == OrderSubmissionStatus.BLOCKED:
        return SmokeResult("reduce_only_block", "passed", detail)
    return SmokeResult("reduce_only_block", "failed", detail)


def _ensure_serializable(engine: TradingEngine) -> None:
    order_submitter = cast(MagicMock, engine._order_submitter)

    def _record_decision_trace(*_: Any, **__: Any) -> None:
        return None

    def _record_preview(*_: Any, **__: Any) -> None:
        return None

    def _record_rejection(*_: Any, **__: Any) -> None:
        return None

    order_submitter.record_decision_trace.side_effect = _record_decision_trace
    order_submitter.record_preview.side_effect = _record_preview
    order_submitter.record_rejection.side_effect = _record_rejection
    order_submitter.record_decision_trace.return_value = None
    order_submitter.record_preview.return_value = None
    order_submitter.record_rejection.return_value = None


def _run_reduce_only_allows_exit(engine: TradingEngine) -> SmokeResult:
    risk_manager = cast(MagicMock, engine.context.risk_manager)
    risk_manager.config.kill_switch_enabled = False
    risk_manager.is_reduce_only_mode.return_value = True
    risk_manager.check_order.return_value = True
    risk_manager._daily_pnl_triggered = False
    risk_manager.check_mark_staleness.return_value = False
    engine._current_positions = {"BTC-USD": MagicMock(quantity=Decimal("1"), side="long")}

    order_validator = cast(MagicMock, engine._order_validator)
    order_validator.validate_exchange_rules.return_value = (Decimal("0.1"), None)
    order_validator.enforce_slippage_guard.return_value = None
    order_validator.ensure_mark_is_fresh.return_value = None
    order_validator.run_pre_trade_validation.return_value = None
    order_validator.maybe_preview_order.return_value = None
    order_validator.finalize_reduce_only_flag.return_value = False

    decision = Decision(Action.SELL, "ops controls smoke")
    result = asyncio.run(
        engine._validate_and_place_order(
            symbol="BTC-USD",
            decision=decision,
            price=Decimal("50000"),
            equity=Decimal("10000"),
            quantity_override=Decimal("1"),
        )
    )

    detail = {
        "status": result.status.value,
        "order_id": str(result.order_id) if result.order_id is not None else None,
    }
    if result.status == OrderSubmissionStatus.SUCCESS:
        return SmokeResult("reduce_only_exit", "passed", detail)
    return SmokeResult("reduce_only_exit", "failed", detail)


def main() -> int:
    with (
        patch(
            "gpt_trader.features.live_trade.engines.strategy.create_strategy",
            return_value=MagicMock(),
        ),
        patch(
            "gpt_trader.security.security_validator.get_validator",
        ) as validator,
    ):
        engine = _build_engine()
        security_result = MagicMock()
        security_result.is_valid = True
        security_result.errors = []
        validator.return_value.validate_order_request.return_value = security_result
        _ensure_serializable(engine)

    async def _bypass_order_guards(
        *_: Any,
        price: Decimal,
        quantity: Decimal,
        reduce_only_flag: bool,
        **__: Any,
    ) -> tuple[Decimal, Decimal, bool, None]:
        return quantity, price, reduce_only_flag, None

    with (
        patch(
            "gpt_trader.features.live_trade.engines.strategy.get_failure_tracker",
        ) as get_failure_tracker,
        patch(
            "gpt_trader.features.live_trade.engines.strategy.TradingEngine._run_security_validation",
            return_value=None,
        ),
        patch(
            "gpt_trader.features.live_trade.engines.strategy.TradingEngine._run_order_validator_guards",
            new=_bypass_order_guards,
        ),
    ):
        failure_tracker = MagicMock()
        failure_tracker.get_failure_count.return_value = 0
        get_failure_tracker.return_value = failure_tracker
        results = [
            _run_kill_switch(engine),
            _run_reduce_only_blocks_entry(engine),
            _run_reduce_only_allows_exit(engine),
        ]

    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": [
            {"name": result.name, "status": result.status, "detail": result.detail}
            for result in results
        ],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"controls_smoke_{_utc_timestamp()}.json"
    output_path.write_text(json.dumps(output, indent=2))
    print(f"output_path={output_path}")

    if any(result.status != "passed" for result in results):
        clear_application_container()
        return 1

    clear_application_container()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
