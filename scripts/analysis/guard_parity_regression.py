"""
Guard parity regression runner.

Usage:
    uv run python scripts/analysis/guard_parity_regression.py --profile canary
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any

from gpt_trader.app.config import BotConfig
from gpt_trader.backtesting.engine.guarded_execution import (
    BacktestDecisionContext,
    BacktestExecutionContext,
    BacktestGuardedExecutor,
)
from gpt_trader.backtesting.simulation.broker import SimulatedBroker
from gpt_trader.backtesting.validation.decision_logger import DecisionLogger
from gpt_trader.core import Candle, MarketType, OrderSide, OrderType, Product
from gpt_trader.features.live_trade.execution.submission_result import OrderSubmissionStatus
from gpt_trader.features.live_trade.execution.validation import ValidationFailureTracker
from gpt_trader.features.live_trade.risk import ValidationError
from gpt_trader.features.live_trade.risk.config import RiskConfig
from gpt_trader.features.live_trade.risk.manager import LiveRiskManager
from gpt_trader.persistence.event_store import EventStore

DEFAULT_EQUITY = Decimal("1000")
DEFAULT_PRICE = Decimal("100")


class PreviewBehavior(Enum):
    SUCCESS = "success"
    VALIDATION_ERROR = "validation_error"
    ERROR = "error"


class PreviewSimulatedBroker(SimulatedBroker):
    def __init__(
        self,
        *,
        initial_equity_usd: Decimal,
        preview_behavior: PreviewBehavior,
        preview_error_message: str | None = None,
    ) -> None:
        super().__init__(initial_equity_usd=initial_equity_usd)
        self._preview_behavior = preview_behavior
        self._preview_error_message = preview_error_message

    def preview_order(self, **kwargs: Any) -> dict[str, Any]:
        if self._preview_behavior is PreviewBehavior.SUCCESS:
            return {"status": "ok", "preview": kwargs}
        if self._preview_behavior is PreviewBehavior.VALIDATION_ERROR:
            message = self._preview_error_message or "Preview blocked"
            raise ValidationError(message)
        message = self._preview_error_message or "Preview failed"
        raise RuntimeError(message)

    def edit_order_preview(self, order_id: str, **kwargs: Any) -> dict[str, Any]:
        preview_payload = {"order_id": order_id, **kwargs}
        return self.preview_order(**preview_payload)


@dataclass(frozen=True)
class GuardParityScenario:
    name: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    price: Decimal | None
    expected_status: OrderSubmissionStatus
    expected_failure_code: str | None
    expected_reason_contains: str | None
    risk_max_leverage: int | None = None
    risk_max_exposure_pct: float | None = None
    risk_slippage_guard_bps: int | None = None
    enable_order_preview: bool | None = None
    preview_behavior: PreviewBehavior | None = None
    preview_error_message: str | None = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Guard parity regression runner")
    parser.add_argument("--profile", default="canary", help="Runtime profile under runtime_data/.")
    parser.add_argument("--symbol", default="BTC-USD", help="Trading symbol.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output dir (default: runtime_data/<profile>/reports).",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        help="Optional run identifier to include in report filenames.",
    )
    return parser.parse_args()


def _build_config(symbol: str) -> BotConfig:
    config = BotConfig()
    config.symbols = [symbol]
    config.enable_order_preview = False
    return config


def _build_broker(
    symbol: str,
    price: Decimal,
    initial_equity: Decimal,
    preview_behavior: PreviewBehavior | None = None,
    preview_error_message: str | None = None,
) -> tuple[SimulatedBroker, Product]:
    if preview_behavior is None:
        broker = SimulatedBroker(initial_equity_usd=initial_equity)
    else:
        broker = PreviewSimulatedBroker(
            initial_equity_usd=initial_equity,
            preview_behavior=preview_behavior,
            preview_error_message=preview_error_message,
        )
    base_asset, quote_asset = symbol.split("-", 1) if "-" in symbol else (symbol, "USD")
    product = Product(
        symbol=symbol,
        base_asset=base_asset,
        quote_asset=quote_asset,
        market_type=MarketType.SPOT,
        min_size=Decimal("0.01"),
        step_size=Decimal("0.01"),
        min_notional=Decimal("10"),
        price_increment=Decimal("0.01"),
        leverage_max=5,
    )
    broker.register_product(product)
    bar_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    broker.update_bar(
        symbol,
        Candle(
            ts=bar_time,
            open=price,
            high=price,
            low=price,
            close=price,
            volume=Decimal("1000"),
        ),
    )
    return broker, product


def _serialize(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    return value


def _evaluate_scenario(
    *,
    scenario: GuardParityScenario,
    result: Any,
    decision: Any | None,
) -> dict[str, Any]:
    decision_failures = decision.risk_check_failures if decision else []
    decision_passed = decision.risk_checks_passed if decision else None
    decision_logged = decision is not None

    expected_passed = scenario.expected_status == OrderSubmissionStatus.SUCCESS
    status_match = result.status == scenario.expected_status

    failure_match = True
    if scenario.expected_failure_code:
        failure_match = scenario.expected_failure_code in decision_failures
    else:
        failure_match = not decision_failures

    reason_text = result.reason or result.error or ""
    reason_match = True
    if scenario.expected_reason_contains:
        reason_match = scenario.expected_reason_contains.lower() in reason_text.lower()

    decision_passed_match = True
    if decision_logged:
        decision_passed_match = decision_passed == expected_passed

    order_id_match = True
    if expected_passed:
        order_id_match = result.order_id is not None

    passed = all(
        [
            decision_logged,
            status_match,
            failure_match,
            reason_match,
            decision_passed_match,
            order_id_match,
        ]
    )

    expected_payload = {
        "status": scenario.expected_status.value,
        "failure_code": scenario.expected_failure_code,
        "reason_contains": scenario.expected_reason_contains,
        "side": scenario.side.value,
        "order_type": scenario.order_type.value,
        "quantity": scenario.quantity,
        "price": scenario.price,
        "risk_max_leverage": scenario.risk_max_leverage,
        "risk_max_exposure_pct": scenario.risk_max_exposure_pct,
        "risk_slippage_guard_bps": scenario.risk_slippage_guard_bps,
        "enable_order_preview": scenario.enable_order_preview,
        "preview_behavior": scenario.preview_behavior,
    }
    actual_payload: dict[str, Any] = {
        "status": result.status.value,
        "order_id": result.order_id,
        "reason": result.reason,
        "error": result.error,
        "decision_logged": decision_logged,
        "decision_risk_passed": decision_passed,
        "decision_failures": decision_failures,
    }
    if decision is not None:
        actual_payload["decision_id"] = decision.decision_id
        actual_payload["decision_cycle_id"] = decision.cycle_id

    return {
        "name": scenario.name,
        "expected": expected_payload,
        "actual": actual_payload,
        "passed": passed,
    }


def _run_scenario(
    *,
    scenario: GuardParityScenario,
    symbol: str,
    run_identifier: str,
    base_price: Decimal,
    initial_equity: Decimal,
) -> dict[str, Any]:
    config = _build_config(symbol)
    if scenario.enable_order_preview is not None:
        config.enable_order_preview = scenario.enable_order_preview
    if scenario.preview_behavior is not None:
        config.enable_order_preview = True

    broker, product = _build_broker(
        symbol,
        base_price,
        initial_equity,
        preview_behavior=scenario.preview_behavior,
        preview_error_message=scenario.preview_error_message,
    )

    risk_config = RiskConfig()
    if scenario.risk_max_leverage is not None:
        risk_config.max_leverage = scenario.risk_max_leverage
    if scenario.risk_max_exposure_pct is not None:
        risk_config.max_exposure_pct = scenario.risk_max_exposure_pct
    if scenario.risk_slippage_guard_bps is not None:
        risk_config.slippage_guard_bps = scenario.risk_slippage_guard_bps
    risk_manager = LiveRiskManager(config=risk_config, event_store=None, state_file=None)

    decision_logger = DecisionLogger()
    event_store = EventStore()
    context = BacktestExecutionContext(
        config=config,
        broker=broker,
        risk_manager=risk_manager,
        event_store=event_store,
        decision_logger=decision_logger,
        failure_tracker=ValidationFailureTracker(),
    )
    executor = BacktestGuardedExecutor(context)
    decision_context = BacktestDecisionContext(
        cycle_id=run_identifier,
        strategy_name="guard_parity",
        strategy_params={"scenario": scenario.name},
        recent_marks=[base_price],
        mark_price=base_price,
    )

    result = executor.submit_order(
        symbol=symbol,
        side=scenario.side,
        order_type=scenario.order_type,
        quantity=scenario.quantity,
        price=scenario.price,
        product=product,
        decision_context=decision_context,
        reason=scenario.name,
    )

    decisions = decision_logger.get_decisions()
    decision = decisions[0] if decisions else None

    return _evaluate_scenario(scenario=scenario, result=result, decision=decision)


def main() -> int:
    args = _parse_args()

    profile = str(args.profile)
    symbol = str(args.symbol)
    output_dir = args.output_dir or Path("runtime_data") / profile / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_identifier = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    scenarios = [
        GuardParityScenario(
            name="exchange_rules_min_notional",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1"),
            price=Decimal("1"),
            expected_status=OrderSubmissionStatus.BLOCKED,
            expected_failure_code="exchange_rules",
            expected_reason_contains="min_notional",
        ),
        GuardParityScenario(
            name="pre_trade_max_leverage",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.2"),
            price=DEFAULT_PRICE,
            expected_status=OrderSubmissionStatus.BLOCKED,
            expected_failure_code="pre_trade_validation",
            expected_reason_contains="Leverage",
            risk_max_leverage=0,
        ),
        GuardParityScenario(
            name="slippage_guard_trip",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.2"),
            price=DEFAULT_PRICE,
            expected_status=OrderSubmissionStatus.BLOCKED,
            expected_failure_code="slippage_guard",
            expected_reason_contains="slippage",
            risk_slippage_guard_bps=10,
        ),
        GuardParityScenario(
            name="order_preview_blocked",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.2"),
            price=DEFAULT_PRICE,
            expected_status=OrderSubmissionStatus.BLOCKED,
            expected_failure_code="order_preview",
            expected_reason_contains="preview",
            risk_slippage_guard_bps=1_000_000,
            enable_order_preview=True,
            preview_behavior=PreviewBehavior.VALIDATION_ERROR,
            preview_error_message="Preview blocked by broker",
        ),
        GuardParityScenario(
            name="insufficient_funds",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("30"),
            price=DEFAULT_PRICE,
            expected_status=OrderSubmissionStatus.FAILED,
            expected_failure_code="insufficient_funds",
            expected_reason_contains="insufficient",
            risk_max_leverage=100,
            risk_max_exposure_pct=100.0,
            risk_slippage_guard_bps=20_000_000,
        ),
        GuardParityScenario(
            name="happy_path_limit",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.2"),
            price=DEFAULT_PRICE,
            expected_status=OrderSubmissionStatus.SUCCESS,
            expected_failure_code=None,
            expected_reason_contains=None,
            risk_slippage_guard_bps=1_000_000,
        ),
        GuardParityScenario(
            name="order_preview_ok",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.2"),
            price=DEFAULT_PRICE,
            expected_status=OrderSubmissionStatus.SUCCESS,
            expected_failure_code=None,
            expected_reason_contains=None,
            risk_slippage_guard_bps=1_000_000,
            enable_order_preview=True,
            preview_behavior=PreviewBehavior.SUCCESS,
        ),
    ]

    scenario_results = [
        _run_scenario(
            scenario=scenario,
            symbol=symbol,
            run_identifier=run_identifier,
            base_price=DEFAULT_PRICE,
            initial_equity=DEFAULT_EQUITY,
        )
        for scenario in scenarios
    ]

    overall_passed = all(result["passed"] for result in scenario_results)
    timestamp = datetime.now(timezone.utc).isoformat()

    report = {
        "run_id": run_identifier,
        "timestamp": timestamp,
        "profile": profile,
        "symbol": symbol,
        "overall_passed": overall_passed,
        "scenarios": scenario_results,
    }

    json_path = output_dir / f"guard_parity_{run_identifier}.json"
    text_path = output_dir / f"guard_parity_{run_identifier}.txt"

    with json_path.open("w") as handle:
        json.dump(_serialize(report), handle, indent=2)

    lines = [
        f"guard_parity_run_id: {run_identifier}",
        f"timestamp: {timestamp}",
        f"profile: {profile}",
        f"symbol: {symbol}",
        f"overall_status: {'PASS' if overall_passed else 'FAIL'}",
    ]
    for result in scenario_results:
        status = "PASS" if result["passed"] else "FAIL"
        expected_status = result["expected"]["status"]
        actual_status = result["actual"]["status"]
        lines.append(
            f"{result['name']}: {status} (expected {expected_status}, actual {actual_status})"
        )
    lines.append(f"json_report: {json_path}")

    text_path.write_text("\n".join(lines) + "\n")

    for line in lines:
        print(line)

    return 0 if overall_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
