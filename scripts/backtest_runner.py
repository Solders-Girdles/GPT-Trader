"""
Backtest runner that produces a readiness evidence artifact.

This script runs a production-style strategy over historical candles and writes:
- JSON: runtime_data/<profile>/reports/backtest_<run_id>.json
- Text: runtime_data/<profile>/reports/backtest_<run_id>.txt
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import deque
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from gpt_trader.app.config.bot_config import BotConfig
from gpt_trader.app.config.profile_loader import ProfileLoader
from gpt_trader.backtesting.data.manager import create_coinbase_data_provider
from gpt_trader.backtesting.metrics.risk import calculate_risk_metrics
from gpt_trader.backtesting.metrics.statistics import calculate_trade_statistics
from gpt_trader.backtesting.simulation.broker import SimulatedBroker
from gpt_trader.backtesting.types import FeeTier, SimulationConfig
from gpt_trader.core import OrderSide, OrderType
from gpt_trader.features.brokerages.coinbase.client.client import CoinbaseClient
from gpt_trader.features.live_trade.factory import create_strategy
from gpt_trader.features.live_trade.strategies.perps_baseline import Action
from gpt_trader.features.live_trade.strategies.perps_baseline.strategy import SpotStrategy
from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="backtest_runner")


def _iso_utc(dt: datetime) -> str:
    dt_utc = dt if dt.tzinfo else dt.replace(tzinfo=UTC)
    return dt_utc.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _parse_dt(value: str) -> datetime:
    raw = value.strip()
    if len(raw) == 10:  # YYYY-MM-DD
        dt = datetime.fromisoformat(raw).replace(tzinfo=UTC)
        return dt
    dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _serialize(value: Any) -> Any:
    if isinstance(value, datetime):
        return _iso_utc(value)
    if isinstance(value, Decimal):
        return str(value)
    if hasattr(value, "value") and isinstance(getattr(value, "value"), str):
        # Enum
        return getattr(value, "value")
    if is_dataclass(value):
        return {k: _serialize(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _serialize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(v) for v in value]
    return value


def _load_profile_config(profile_name: str) -> BotConfig:
    from gpt_trader.config.types import Profile

    profile_enum = Profile(profile_name)
    loader = ProfileLoader()
    schema = loader.load(profile_enum)
    kwargs = loader.to_bot_config_kwargs(schema, profile_enum)
    return BotConfig(**kwargs)


def _build_position_state(position: Any) -> dict[str, Any] | None:
    if position is None:
        return None
    return {
        "quantity": position.quantity,
        "entry_price": position.entry_price,
        "unrealized_pnl": position.unrealized_pnl,
        "side": position.side,
        "leverage": position.leverage,
    }


def _select_strategy(config: BotConfig) -> Any:
    strategy = create_strategy(config)

    # Backtesting defaults to spot-safe behavior when shorts are disabled.
    if config.strategy_type == "baseline" and not config.enable_shorts:
        return SpotStrategy(config=config.strategy)

    return strategy


def _lookback_bars(strategy: Any) -> int:
    config = getattr(strategy, "config", None)
    if config is None:
        return 64

    long_ma = getattr(config, "long_ma_period", getattr(config, "long_ma", 20))
    rsi_period = getattr(config, "rsi_period", 14)
    required = max(int(long_ma), int(rsi_period) + 1) + 10
    return max(required, 64)


def _execute_decision(
    *,
    broker: SimulatedBroker,
    symbol: str,
    action: Action,
    equity: Decimal,
    price: Decimal,
    leverage: int,
    position_fraction: Decimal,
) -> None:
    position = broker.get_position(symbol)

    if action == Action.CLOSE:
        if position is None:
            return
        side = OrderSide.SELL if position.quantity > 0 else OrderSide.BUY
        broker.place_order(
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=abs(position.quantity),
            reduce_only=True,
            leverage=leverage,
        )
        return

    if action not in (Action.BUY, Action.SELL):
        return

    if price <= 0:
        return

    target_notional = equity * position_fraction * Decimal(leverage)
    quantity = target_notional / price
    if quantity <= 0:
        return

    side = OrderSide.BUY if action == Action.BUY else OrderSide.SELL
    broker.place_order(
        symbol=symbol,
        side=side,
        order_type=OrderType.MARKET,
        quantity=quantity,
        leverage=leverage,
    )


async def run_backtest(
    *,
    profile: str,
    symbol: str,
    granularity: str,
    start: datetime,
    end: datetime,
    initial_equity_usd: Decimal,
    fee_tier: FeeTier,
    position_fraction: Decimal,
    max_leverage: int,
    cache_dir: Path,
) -> tuple[dict[str, Any], str]:
    config = _load_profile_config(profile)
    config.symbols = [symbol]

    # Use public market endpoints by default (no auth required).
    client = CoinbaseClient(api_mode=config.coinbase_api_mode)
    data_provider = create_coinbase_data_provider(
        client=client,
        cache_dir=cache_dir,
        validate_quality=True,
    )

    candles = await data_provider.get_candles(
        symbol=symbol,
        granularity=granularity,
        start=start,
        end=end,
    )
    candles = sorted(candles, key=lambda c: c.ts)
    if not candles:
        raise RuntimeError(f"No candles returned for {symbol} {granularity} {start}..{end}")

    sim_config = SimulationConfig(
        start_date=start,
        end_date=end,
        granularity=granularity,
        initial_equity_usd=initial_equity_usd,
        fee_tier=fee_tier,
    )
    broker = SimulatedBroker(
        initial_equity_usd=initial_equity_usd,
        fee_tier=fee_tier,
        config=sim_config,
    )
    broker.connect()

    strategy = _select_strategy(config)
    leverage = max(1, min(max_leverage, int(getattr(config.risk, "target_leverage", 1))))
    mark_history: deque[Decimal] = deque(maxlen=_lookback_bars(strategy))

    for idx, candle in enumerate(candles, start=1):
        broker.update_bar(symbol, candle)
        broker.update_equity_curve()

        mark_history.append(candle.close)
        recent_marks = list(mark_history)
        current_mark = candle.close
        equity = broker.equity
        decision = strategy.decide(
            symbol=symbol,
            current_mark=current_mark,
            position_state=_build_position_state(broker.get_position(symbol)),
            recent_marks=recent_marks,
            equity=equity,
            product=None,
        )

        _execute_decision(
            broker=broker,
            symbol=symbol,
            action=decision.action,
            equity=equity,
            price=current_mark,
            leverage=leverage,
            position_fraction=position_fraction,
        )

        if idx % 5000 == 0:
            logger.info(
                "Backtest progress",
                processed=idx,
                total=len(candles),
                progress_pct=f"{(idx / len(candles)) * 100:.1f}",
            )

    # Force-close open position to realize PnL for reporting.
    if broker.get_position(symbol) is not None:
        _execute_decision(
            broker=broker,
            symbol=symbol,
            action=Action.CLOSE,
            equity=broker.equity,
            price=candles[-1].close,
            leverage=leverage,
            position_fraction=position_fraction,
        )
        broker.update_equity_curve()

    risk = calculate_risk_metrics(broker)
    stats = calculate_trade_statistics(broker)
    broker_stats = broker.get_statistics()

    quality_report = data_provider.get_quality_report(symbol, granularity)

    payload: dict[str, Any] = {
        "run_id": f"backtest_{datetime.now(tz=UTC).strftime('%Y%m%d_%H%M%S')}",
        "generated_at": _iso_utc(datetime.now(tz=UTC)),
        "profile": profile,
        "symbol": symbol,
        "granularity": granularity,
        "start_date": _iso_utc(start),
        "end_date": _iso_utc(end),
        "candles_loaded": len(candles),
        "strategy_type": config.strategy_type,
        "strategy_class": strategy.__class__.__name__,
        "initial_equity_usd": str(initial_equity_usd),
        "fee_tier": fee_tier.value,
        "leverage": leverage,
        "position_fraction": str(position_fraction),
        "broker_stats": _serialize(broker_stats),
        "risk_metrics": _serialize(risk),
        "trade_statistics": _serialize(stats),
        "data_quality": _serialize(quality_report) if quality_report is not None else None,
    }

    summary_lines = [
        "BACKTEST SUMMARY",
        f"Profile: {profile}",
        f"Symbol: {symbol}",
        f"Granularity: {granularity}",
        f"Range: {_iso_utc(start)} .. {_iso_utc(end)}",
        f"Candles: {len(candles)}",
        "",
        "PERFORMANCE",
        f"Final Equity: {broker_stats['final_equity']}",
        f"Total Return %: {broker_stats['total_return_pct']}",
        "",
        "RISK",
        f"Max Drawdown %: {risk.max_drawdown_pct}",
        f"Sharpe: {risk.sharpe_ratio}",
        f"Sortino: {risk.sortino_ratio}",
        "",
        "TRADES",
        f"Total Trades: {stats.total_trades}",
        f"Win Rate %: {stats.win_rate}",
        f"Profit Factor: {stats.profit_factor}",
    ]
    summary = "\n".join(summary_lines) + "\n"
    return payload, summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a strategy backtest and save evidence artifacts."
    )
    parser.add_argument("--profile", default="canary", help="Profile name (default: canary)")
    parser.add_argument(
        "--symbol", default=None, help="Symbol to backtest (default: profile first)"
    )
    parser.add_argument("--granularity", default="FIVE_MINUTE", help="Candle granularity")
    parser.add_argument(
        "--start", type=str, default=None, help="Start datetime (ISO) or YYYY-MM-DD"
    )
    parser.add_argument("--end", type=str, default=None, help="End datetime (ISO) or YYYY-MM-DD")
    parser.add_argument(
        "--days", type=int, default=30, help="Lookback days when --start/--end omitted"
    )
    parser.add_argument(
        "--initial-equity-usd",
        type=str,
        default="100000",
        help="Initial equity in USD (default: 100000)",
    )
    parser.add_argument(
        "--fee-tier",
        type=str,
        default=FeeTier.TIER_2.name,
        choices=[t.name for t in FeeTier],
        help="Fee tier (default: TIER_2)",
    )
    parser.add_argument(
        "--position-fraction",
        type=str,
        default="0.10",
        help="Position fraction per trade (default: 0.10)",
    )
    parser.add_argument(
        "--max-leverage",
        type=int,
        default=1,
        help="Max leverage to use in simulation (default: 1)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Candle cache dir (default: runtime_data/<profile>/cache/candles)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output dir (default: runtime_data/<profile>/reports)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    profile = args.profile
    config = _load_profile_config(profile)

    symbol = args.symbol or (config.symbols[0] if config.symbols else "BTC-USD")
    granularity = args.granularity

    end = _parse_dt(args.end) if args.end else datetime.now(tz=UTC)
    start = _parse_dt(args.start) if args.start else end - timedelta(days=int(args.days))

    cache_dir = args.cache_dir or Path("runtime_data") / profile / "cache" / "candles"
    output_dir = args.output_dir or Path("runtime_data") / profile / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    initial_equity = Decimal(args.initial_equity_usd)
    fee_tier = FeeTier[args.fee_tier]
    position_fraction = Decimal(args.position_fraction)

    payload, summary = asyncio.run(
        run_backtest(
            profile=profile,
            symbol=symbol,
            granularity=granularity,
            start=start,
            end=end,
            initial_equity_usd=initial_equity,
            fee_tier=fee_tier,
            position_fraction=position_fraction,
            max_leverage=int(args.max_leverage),
            cache_dir=cache_dir,
        )
    )

    run_id = payload["run_id"]
    json_path = output_dir / f"{run_id}.json"
    txt_path = output_dir / f"{run_id}.txt"

    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    txt_path.write_text(summary)

    logger.info(
        "Backtest complete", run_id=run_id, json_path=str(json_path), txt_path=str(txt_path)
    )
    print(summary)
    print(f"Saved: {json_path}")
    print(f"Saved: {txt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
