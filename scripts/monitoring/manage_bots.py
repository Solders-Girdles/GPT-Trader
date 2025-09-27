#!/usr/bin/env python3
"""
Minimal CLI to manage paper bots (create/list/start/stop/snapshot).

Usage examples:
  python scripts/manage_bots.py create --id bot1 --name scalp-bot \
      --strategy scalp --symbols BTC-USD,ETH-USD,SOL-USD \
      --capital 15000 --max-positions 8 --max-position-size 0.2

  python scripts/manage_bots.py start --id bot1
  python scripts/manage_bots.py list
  python scripts/manage_bots.py snapshot --id bot1
  python scripts/manage_bots.py stop --id bot1
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Load production env (CDP keys) if present for market data
env_file = Path(__file__).parent.parent / '.env.production'
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                value = value.strip().strip('"')
                if key == 'COINBASE_CDP_PRIVATE_KEY':
                    private_key_lines = [value] if value else []
                    for next_line in f:
                        next_line = next_line.strip()
                        private_key_lines.append(next_line)
                        if 'END EC PRIVATE KEY' in next_line:
                            break
                    value = '\n'.join(private_key_lines)
                os.environ[key] = value

from bot_v2.orchestration.bot_manager import BotManager, BotConfig, RiskConfig
from bot_v2.orchestration.session_store import SessionStore


MANAGER = BotManager()
STORE = SessionStore()


def _parse_kv_params(items):
    params = {}
    for it in items or []:
        if '=' not in it:
            continue
        k, v = it.split('=', 1)
        k = k.strip()
        v = v.strip()
        # Try to cast to int/float/bool
        vl = v.lower()
        if vl in ('true', 'false'):
            params[k] = (vl == 'true')
            continue
        try:
            if '.' in v:
                params[k] = float(v)
            else:
                params[k] = int(v)
            continue
        except ValueError:
            params[k] = v
    return params


def cmd_create(args):
    symbols = [s.strip() for s in args.symbols.split(',') if s.strip()]
    risk = RiskConfig(
        max_positions=args.max_positions,
        max_position_size=args.max_position_size,
        stop_loss=args.stop_loss,
        take_profit=args.take_profit,
        commission=args.commission,
        slippage=args.slippage,
    )
    # Strategy params based on selected strategy
    sp: dict = {}
    if args.strategy == 'momentum':
        if args.momentum_period is not None:
            sp['momentum_period'] = args.momentum_period
        if args.momentum_threshold is not None:
            sp['threshold'] = args.momentum_threshold
    elif args.strategy == 'mean_reversion':
        if args.bb_period is not None:
            sp['bb_period'] = args.bb_period
        if args.bb_std is not None:
            sp['bb_std'] = args.bb_std
    elif args.strategy == 'breakout':
        if args.breakout_period is not None:
            sp['breakout_period'] = args.breakout_period
        if args.breakout_threshold is not None:
            sp['threshold_pct'] = args.breakout_threshold
    elif args.strategy == 'ma_crossover':
        if args.ma_fast is not None:
            sp['fast_period'] = args.ma_fast
        if args.ma_slow is not None:
            sp['slow_period'] = args.ma_slow
    elif args.strategy == 'volatility':
        if args.vol_period is not None:
            sp['vol_period'] = args.vol_period
        if args.vol_threshold is not None:
            sp['vol_threshold'] = args.vol_threshold
    elif args.strategy == 'scalp':
        if args.scalp_bp_threshold is not None:
            sp['bp_threshold'] = args.scalp_bp_threshold

    # Generic passthrough params: --param k=v
    sp.update(_parse_kv_params(args.param))

    cfg = BotConfig(
        bot_id=args.id,
        name=args.name,
        symbols=symbols,
        strategy=args.strategy,
        capital=args.capital,
        strategy_params=sp,
        risk=risk,
        loop_sleep=args.loop_sleep,
    )
    MANAGER.add_bot(cfg)
    print(f"✅ Created bot {args.id} ({args.name}) with strategy {args.strategy} and params {sp or '{}'}")


def cmd_start(args):
    MANAGER.start(args.id)
    print(f"🚀 Started bot {args.id}")


def cmd_stop(args):
    MANAGER.stop(args.id)
    print(f"🛑 Stopped bot {args.id}")


def cmd_list(args):
    bots = MANAGER.list_bots()
    if not bots:
        print("No bots")
        return
    for b in bots:
        print(f"- {b.config.bot_id}: {b.config.name} [{b.status}] symbols={','.join(b.config.symbols)} strategy={b.config.strategy}")
        m = b.metrics
        print(f"    trades/hr={m.trades_per_hour:.2f} signals/hr={m.signals_per_hour:.2f} exec%={m.execution_rate:.1f}")
        if b.config.strategy_params:
            print(f"    params={b.config.strategy_params}")


def cmd_snapshot(args):
    bot = MANAGER.get_bot(args.id)
    path = STORE.save_bot_snapshot(bot)
    print(f"💾 Snapshot saved: {path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Bot Manager CLI")
    sub = p.add_subparsers(dest='cmd', required=True)

    pc = sub.add_parser('create', help='Create a bot')
    pc.add_argument('--id', required=True)
    pc.add_argument('--name', required=True)
    pc.add_argument('--strategy', default='scalp')
    pc.add_argument('--symbols', required=True, help='Comma-separated symbols')
    pc.add_argument('--capital', type=float, default=10000)
    pc.add_argument('--max-positions', type=int, default=6)
    pc.add_argument('--max-position-size', type=float, default=0.2)
    pc.add_argument('--stop-loss', type=float, default=0.04)
    pc.add_argument('--take-profit', type=float, default=0.08)
    pc.add_argument('--commission', type=float, default=0.004)
    pc.add_argument('--slippage', type=float, default=0.001)
    pc.add_argument('--loop-sleep', type=float, default=5.0)
    # Strategy-specific knobs
    pc.add_argument('--momentum-period', type=int, default=None)
    pc.add_argument('--momentum-threshold', type=float, default=None)
    pc.add_argument('--bb-period', type=int, default=None)
    pc.add_argument('--bb-std', type=float, default=None)
    pc.add_argument('--breakout-period', type=int, default=None)
    pc.add_argument('--breakout-threshold', type=float, default=None, help='fraction, e.g., 0.01 = 1%')
    pc.add_argument('--ma-fast', type=int, default=None)
    pc.add_argument('--ma-slow', type=int, default=None)
    pc.add_argument('--vol-period', type=int, default=None)
    pc.add_argument('--vol-threshold', type=float, default=None)
    pc.add_argument('--scalp-bp-threshold', type=float, default=None, help='fraction, e.g., 0.0005 = 5 bps')
    pc.add_argument('--param', action='append', help='Extra strategy param as key=value', default=None)
    pc.set_defaults(func=cmd_create)

    ps = sub.add_parser('start', help='Start a bot')
    ps.add_argument('--id', required=True)
    ps.set_defaults(func=cmd_start)

    pst = sub.add_parser('stop', help='Stop a bot')
    pst.add_argument('--id', required=True)
    pst.set_defaults(func=cmd_stop)

    pl = sub.add_parser('list', help='List bots')
    pl.set_defaults(func=cmd_list)

    psnap = sub.add_parser('snapshot', help='Save bot snapshot to results/managed')
    psnap.add_argument('--id', required=True)
    psnap.set_defaults(func=cmd_snapshot)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
