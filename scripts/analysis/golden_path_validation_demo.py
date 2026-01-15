"""
Golden-path validation demo for backtesting.

Usage:
    uv run python scripts/analysis/golden_path_validation_demo.py
    uv run python scripts/analysis/golden_path_validation_demo.py --mismatch --export /tmp/decisions.json
    uv run python scripts/analysis/golden_path_validation_demo.py --log-dir /tmp/decision_logs
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from gpt_trader.backtesting.validation.decision_logger import DecisionLogger, StrategyDecision
from gpt_trader.backtesting.validation.validator import GoldenPathValidator


def _build_decision(
    *,
    cycle_id: str,
    timestamp: datetime,
    symbol: str,
    action: str,
    quantity: Decimal = Decimal("0"),
    price: Decimal | None = None,
    reason: str = "demo",
) -> StrategyDecision:
    decision = StrategyDecision.create(
        cycle_id=cycle_id,
        symbol=symbol,
        equity=Decimal("100000"),
        position_quantity=Decimal("0"),
        position_side=None,
        mark_price=price or Decimal("50000"),
        recent_marks=[price or Decimal("50000")],
    )
    decision.timestamp = timestamp
    decision.with_strategy("demo_strategy", {"mode": "golden_path"})
    decision.with_action(
        action=action,
        quantity=quantity,
        price=price,
        order_type="MARKET",
        reason=reason,
    )
    decision.with_risk_result(True, [])
    return decision


def main() -> int:
    parser = argparse.ArgumentParser(description="Golden-path validation demo")
    parser.add_argument(
        "--mismatch",
        action="store_true",
        help="Introduce a mismatch between live and simulated decisions",
    )
    parser.add_argument(
        "--export",
        type=Path,
        help="Write decisions JSON to this path",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        help="Write JSONL decision logs to this directory",
    )
    parser.add_argument(
        "--cycle-id",
        type=str,
        help="Override the generated decision cycle id",
    )
    args = parser.parse_args()

    logger = DecisionLogger(storage_path=args.log_dir)
    cycle_id = args.cycle_id or logger.start_cycle()
    start_time = datetime.now(timezone.utc)

    live_decisions = [
        _build_decision(
            cycle_id=cycle_id,
            timestamp=start_time,
            symbol="BTC-USD",
            action="HOLD",
            reason="warmup",
        ),
        _build_decision(
            cycle_id=cycle_id,
            timestamp=start_time + timedelta(minutes=5),
            symbol="BTC-USD",
            action="BUY",
            quantity=Decimal("0.01"),
            price=Decimal("50000"),
            reason="signal",
        ),
    ]

    for decision in live_decisions:
        logger.log_decision(decision)

    simulated_decisions = [
        _build_decision(
            cycle_id=cycle_id,
            timestamp=live_decisions[0].timestamp,
            symbol="BTC-USD",
            action="HOLD",
            reason="simulated",
        ),
        _build_decision(
            cycle_id=cycle_id,
            timestamp=live_decisions[1].timestamp,
            symbol="BTC-USD",
            action="BUY",
            quantity=Decimal("0.01"),
            price=Decimal("50000"),
            reason="simulated",
        ),
    ]

    if args.mismatch:
        simulated_decisions[1].with_action(
            action="SELL",
            quantity=Decimal("0.02"),
            price=Decimal("50010"),
            order_type="MARKET",
            reason="forced_mismatch",
        )

    validator = GoldenPathValidator()
    results = [
        validator.validate_decision(live, simulated)
        for live, simulated in zip(live_decisions, simulated_decisions)
    ]

    report = validator.generate_report(cycle_id)

    print("Golden-path validation results")
    for result in results:
        status = "MATCH" if result.matches else "MISMATCH"
        live_action = result.live_decision.action
        sim_action = result.sim_decision.action if result.sim_decision else "N/A"
        print(f"- {status}: {live_action} vs {sim_action}")

    print(f"Match rate: {report.match_rate:.2f}%")
    print(f"Divergences: {len(report.divergences)}")

    if args.export:
        exported = logger.export_to_json(args.export)
        print(f"Exported {exported} decisions to {args.export}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
