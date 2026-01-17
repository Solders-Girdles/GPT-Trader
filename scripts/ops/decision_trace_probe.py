#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import logging
import sqlite3
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from gpt_trader.app.config import BotConfig
from gpt_trader.app.container import (
    clear_application_container,
    create_application_container,
    set_application_container,
)
from gpt_trader.config.types import Profile
from gpt_trader.core import OrderSide

PROBE_REASON = "ops: decision_id probe (quantity_override=0)"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Emit a decision trace with decision_id via dry-run guard stack."
    )
    parser.add_argument("--profile", default="canary", help="Profile name (default: canary)")
    parser.add_argument("--symbol", default="BTC-USD", help="Symbol to use (default: BTC-USD)")
    parser.add_argument(
        "--side",
        default="buy",
        choices=("buy", "sell"),
        help="Order side (default: buy)",
    )
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=Path("."),
        help="Repo/runtime root (default: .)",
    )
    return parser.parse_args()


def _read_latest_trace(events_db: Path) -> tuple[int | None, str | None]:
    if not events_db.exists():
        return None, None
    connection = sqlite3.connect(str(events_db))
    try:
        cursor = connection.execute(
            """
            SELECT id, timestamp
            FROM events
            WHERE event_type = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            ("order_decision_trace",),
        )
        row = cursor.fetchone()
        if row is None:
            return None, None
        return row[0], row[1]
    finally:
        connection.close()


def _format_timestamp(value: str | None) -> str:
    if not value:
        return "-"
    try:
        dt = datetime.fromisoformat(value.replace(" ", "T"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()
    except ValueError:
        return value


def _build_config(profile_name: str, runtime_root: Path) -> BotConfig:
    profile = Profile(profile_name.lower())
    config = BotConfig.from_profile(profile=profile, dry_run=True, mock_broker=True)
    config.runtime_root = str(runtime_root)
    config.dry_run = True
    config.mock_broker = True
    return config


async def _run_probe(
    *,
    symbol: str,
    side: OrderSide,
    config: BotConfig,
) -> tuple[str, str | None, str | None]:
    container = create_application_container(config)
    set_application_container(container)
    bot = container.create_bot()
    try:
        result = await bot.engine.submit_order(
            symbol=symbol,
            side=side,
            price=Decimal("1"),
            equity=Decimal("1000"),
            quantity_override=Decimal("0"),
            reason=PROBE_REASON,
        )
        decision_id = None
        if result.decision_trace is not None:
            decision_id = result.decision_trace.decision_id
        return result.status.value, decision_id, result.reason
    finally:
        await bot.shutdown()
        clear_application_container()


def main() -> int:
    args = _parse_args()

    logging.getLogger("gpt_trader").setLevel(logging.WARNING)

    side = OrderSide.BUY if args.side == "buy" else OrderSide.SELL
    config = _build_config(args.profile, args.runtime_root)

    events_db = Path(config.runtime_root) / "runtime_data" / args.profile / "events.db"

    async def runner() -> int:
        status, decision_id, reason = await _run_probe(
            symbol=args.symbol,
            side=side,
            config=config,
        )
        latest_id, latest_ts = _read_latest_trace(events_db)
        print(f"status={status}")
        print(f"decision_id={decision_id or '-'}")
        if reason:
            print(f"blocked_reason={reason}")
        if latest_id is not None:
            print(f"latest_trace_id={latest_id}")
        print(f"latest_trace_ts={_format_timestamp(latest_ts)}")
        return 0

    return asyncio.run(runner())


if __name__ == "__main__":
    raise SystemExit(main())
