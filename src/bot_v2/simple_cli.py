#!/usr/bin/env python3
"""
Simple CLI for GPT-Trader V2 Feature Slices
Direct access to trading system features without complex orchestration.

Usage:
    python simple_cli.py backtest --symbol BTC-USD
    python simple_cli.py analyze --symbol BTC-USD
    python simple_cli.py optimize --symbol BTC-USD
"""

import argparse
import sys

# Import feature slices with error handling
try:
    from features.backtest import run_backtest
    BACKTEST_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Backtest module not available: {e}")
    BACKTEST_AVAILABLE = False

try:
    from features.analyze import analyze_symbol
    ANALYZE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Analyze module not available: {e}")
    ANALYZE_AVAILABLE = False

try:
    from features.optimize import optimize_strategy
    OPTIMIZE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Optimize module not available: {e}")
    OPTIMIZE_AVAILABLE = False

# Readiness and fees are always available via core modules
import asyncio
import os
from decimal import Decimal
from bot_v2.orchestration.broker_factory import create_brokerage
from bot_v2.features.brokerages.core.interfaces import MarketType
from bot_v2.config.live_trade_config import RiskConfig
from bot_v2.features.live_trade.fees_engine import create_fees_engine
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except Exception:
    pass


def cmd_backtest(args) -> None:
    """Run backtest for a given symbol."""
    if not BACKTEST_AVAILABLE:
        print("❌ Backtest module not available")
        sys.exit(1)
    
    print(f"🔄 Running backtest for {args.symbol}...")
    try:
        from datetime import datetime
        result = run_backtest(
            strategy="MomentumStrategy",
            symbol=args.symbol, 
            start=datetime(2023, 1, 1),
            end=datetime(2024, 1, 1),
            initial_capital=10000
        )
        print(f"✅ Backtest completed for {args.symbol}")
        print(f"📊 Results: {result}")
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        sys.exit(1)


def cmd_analyze(args) -> None:
    """Analyze market data for a given symbol."""
    if not ANALYZE_AVAILABLE:
        print("❌ Analyze module not available")
        sys.exit(1)
    
    print(f"🔍 Analyzing {args.symbol}...")
    try:
        result = analyze_symbol(symbol=args.symbol, lookback_days=60)
        print(f"✅ Analysis completed for {args.symbol}")
        print(f"📈 Results: {result}")
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        sys.exit(1)


def cmd_optimize(args) -> None:
    """Optimize strategy parameters for a given symbol."""
    if not OPTIMIZE_AVAILABLE:
        print("❌ Optimize module not available")
        sys.exit(1)
    
    print(f"⚡ Optimizing strategy for {args.symbol}...")
    try:
        from datetime import datetime
        result = optimize_strategy(
            strategy="Momentum",
            symbol=args.symbol,
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2024, 1, 1)
        )
        print(f"✅ Optimization completed for {args.symbol}")
        print(f"🎯 Results: {result}")
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        sys.exit(1)


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        description="GPT-Trader V2 Simple CLI",
        epilog="Examples:\n  %(prog)s backtest --symbol AAPL\n  %(prog)s analyze --symbol TSLA\n  %(prog)s optimize --symbol SPY",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Run historical backtest')
    backtest_parser.add_argument('--symbol', required=True, help='Symbol (e.g., BTC-USD or AAPL)')
    backtest_parser.set_defaults(func=cmd_backtest)
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze market data')
    analyze_parser.add_argument('--symbol', required=True, help='Symbol (e.g., BTC-USD or TSLA)')
    analyze_parser.set_defaults(func=cmd_analyze)
    
    # Optimize command
    optimize_parser = subparsers.add_parser('optimize', help='Optimize strategy parameters')
    optimize_parser.add_argument('--symbol', required=True, help='Symbol (e.g., BTC-USD or SPY)')
    optimize_parser.set_defaults(func=cmd_optimize)
    
    # Broker smoke command
    broker_parser = subparsers.add_parser('broker', help='Broker quick smoke (mock or sandbox)')
    broker_parser.add_argument('--broker', default=None, help='Broker name (e.g., coinbase)')
    broker_parser.add_argument('--sandbox', action='store_true', help='Use sandbox environment if supported')
    broker_parser.add_argument('--run-order-tests', action='store_true', help='Place and cancel a tiny test order (use with caution)')
    broker_parser.add_argument('--symbol', default='BTC-USD', help='Symbol for order test')
    broker_parser.add_argument('--limit-price', type=float, default=10.0, help='Limit price for test order')
    broker_parser.add_argument('--qty', type=float, default=0.001, help='Quantity for test order')
    broker_parser.set_defaults(func=cmd_broker_smoke)

    # Perps command (wraps the Phase 7 runner)
    perps_parser = subparsers.add_parser('perps', help='Run the perpetuals trading bot')
    perps_parser.add_argument('--profile', choices=['dev', 'demo', 'prod'], default='dev', help='Configuration profile')
    perps_parser.add_argument('--dry-run', action='store_true', help='Run without placing real orders')
    perps_parser.add_argument('--symbols', nargs='+', help='Symbols to trade (e.g., BTC-PERP ETH-PERP)')
    perps_parser.add_argument('--interval', type=int, default=5, help='Update interval in seconds')
    perps_parser.add_argument('--leverage', type=int, default=2, help='Target leverage')
    perps_parser.add_argument('--reduce-only', action='store_true', help='Enable reduce-only mode')
    perps_parser.add_argument('--dev-fast', action='store_true', help='Run single cycle and exit (for testing)')
    perps_parser.set_defaults(func=cmd_perps)

    # Readiness audit
    ready_parser = subparsers.add_parser('readiness', help='Audit env, broker access, risk config, fees')
    ready_parser.add_argument('--symbol', default='BTC-PERP', help='Symbol to probe (default: BTC-PERP)')
    ready_parser.set_defaults(func=cmd_readiness)

    # Breakeven calculator
    be_parser = subparsers.add_parser('breakeven', help='Compute fee-aware break-even exit price')
    be_parser.add_argument('--side', choices=['long', 'short'], required=True, help='Position side')
    be_parser.add_argument('--entry', type=float, required=True, help='Entry price')
    be_parser.add_argument('--safety-bps', type=float, default=10.0, help='Extra safety margin in bps (default 10)')
    be_parser.add_argument('--symbol', default='BTC-PERP', help='Symbol for tier lookup (default: BTC-PERP)')
    be_parser.set_defaults(func=cmd_breakeven)

    return parser


def cmd_broker_smoke(args) -> None:
    """Quick smoke for broker integration using factory (supports Coinbase)."""
    import os
    from decimal import Decimal
    from bot_v2.orchestration.broker_factory import create_brokerage
    from bot_v2.features.brokerages.core.interfaces import OrderSide, OrderType, TimeInForce

    # Respect explicit flag or env
    if args.broker:
        os.environ['BROKER'] = args.broker
    if args.sandbox:
        os.environ['COINBASE_SANDBOX'] = '1'

    print("🔌 Creating brokerage via factory...")
    broker = create_brokerage()
    print("🔗 Connecting...")
    if not broker.connect():
        print("❌ Failed to connect.")
        return
    print("✅ Connected.")

    print("📦 Listing products (first 5)...")
    products = broker.list_products()
    print([p.symbol for p in products[:5]])

    print("💰 Listing balances (first 5)...")
    balances = broker.list_balances()
    print([f"{b.asset}:{b.available}" for b in balances[:5]])

    if args.run_order_tests:
        print("🧪 Placing test limit order (and cancelling)...")
        order = broker.place_order(
            symbol=args.symbol,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            qty=Decimal(str(args.qty)),
            price=Decimal(str(args.limit_price)),
            tif=TimeInForce.GTC,
        )
        print(f"Order ID: {order.id}")
        cancelled = broker.cancel_order(order.id)
        print(f"Cancelled: {cancelled}")


def _bool_env(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).lower() in ('1', 'true', 'yes', 'on')


def cmd_readiness(args) -> None:
    """Audit environment, broker access, risk config, and fees tier."""
    symbol = args.symbol
    print("🔎 Readiness Audit")
    print("—" * 40)

    # 1) Env checks
    print("1) Environment")
    broker_env = os.getenv('BROKER', 'coinbase')
    api_mode = os.getenv('COINBASE_API_MODE', 'advanced')
    sandbox = os.getenv('COINBASE_SANDBOX', '0')
    deriv = os.getenv('COINBASE_ENABLE_DERIVATIVES', '0')
    paper = _bool_env('PERPS_PAPER', False)
    print(f"   BROKER={broker_env} | MODE={api_mode} | SANDBOX={sandbox} | DERIV={deriv} | PAPER={paper}")
    creds_ok = bool(os.getenv('COINBASE_PROD_CDP_API_KEY') and os.getenv('COINBASE_PROD_CDP_PRIVATE_KEY'))
    if not paper and not creds_ok:
        print("   ⚠️  Missing CDP credentials for live derivatives (set COINBASE_PROD_CDP_API_KEY/PRIVATE_KEY)")
    else:
        print("   ✅ Credentials present (or paper mode)")

    # 2) Broker connectivity and product probe (non-destructive)
    print("2) Broker probe")
    try:
        broker = create_brokerage()
        ok = broker.validate_connection() if hasattr(broker, 'validate_connection') else broker.connect()
        if not ok:
            print("   ❌ Could not validate/connect to broker")
        else:
            print("   ✅ Broker connection validated")
            prods = broker.list_products(market=MarketType.PERPETUAL)
            print(f"   📦 Perpetual products visible: {len(prods)}")
            if prods:
                syms = ', '.join(p.symbol for p in prods[:5])
                print(f"   → Sample: {syms}")
            try:
                q = broker.get_quote(symbol)
                if q and q.bid and q.ask:
                    spread_bps = float((q.ask - q.bid) / ((q.ask + q.bid)/2) * 10000)
                    print(f"   💬 {symbol} quote: bid={q.bid} ask={q.ask} spread≈{spread_bps:.1f}bps")
            except Exception:
                pass
    except Exception as e:
        print(f"   ❌ Broker probe error: {e}")

    # 3) Risk config sanity
    print("3) Risk configuration")
    rc = RiskConfig.from_env()
    print(f"   max_leverage={rc.max_leverage} per_symbol={rc.leverage_max_per_symbol or {}}")
    print(f"   min_liq_buffer={rc.min_liquidation_buffer_pct:.2f} daily_loss_limit={rc.daily_loss_limit}")
    print(f"   exposure cap={rc.max_exposure_pct:.2f} per_symbol_cap={rc.max_position_pct_per_symbol:.2f}")
    if rc.max_leverage > 3:
        print("   ⚠️  Consider starting with max_leverage ≤ 3")
    if rc.min_liquidation_buffer_pct < 0.10:
        print("   ⚠️  Raise min_liquidation_buffer_pct to ≥ 0.10 for safety")
    if rc.daily_loss_limit <= 0:
        print("   ⚠️  Set a positive RISK_DAILY_LOSS_LIMIT (e.g., 50–200)")
    if rc.kill_switch_enabled:
        print("   🛑 Kill switch: ENABLED")
    if rc.reduce_only_mode:
        print("   🧯 Reduce-only mode: ENABLED")

    # 4) Fees tier
    print("4) Fees & break-even")
    try:
        engine = asyncio.run(create_fees_engine())
        tier = asyncio.run(engine.tier_resolver.get_current_tier())
        print(f"   Tier: {tier.tier_name} | maker={float(tier.maker_rate):.4f} taker={float(tier.taker_rate):.4f}")
        entry = Decimal('50000')
        min_exit_long = asyncio.run(engine.get_minimum_profit_target(entry, 'long', symbol))
        print(f"   Example: Long @ {entry} → min exit ≈ {min_exit_long:.2f} (taker+taker)")
    except Exception as e:
        print(f"   ⚠️  Fee tier check skipped: {e}")

    print("\n✅ Readiness audit complete. Next: --dry-run perps, then tiny size.")


def cmd_breakeven(args) -> None:
    """Compute fee-aware break-even exit price given an entry."""
    side = args.side.lower()
    entry = Decimal(str(args.entry))
    safety = Decimal(str(args.safety_bps / 10000.0))
    symbol = args.symbol
    try:
        engine = asyncio.run(create_fees_engine())
        min_exit = asyncio.run(engine.get_minimum_profit_target(entry, side, symbol=symbol, safety_margin=safety))
        if side == 'long':
            print(f"Entry {entry} long → min exit {min_exit:.2f} (incl. taker fees + {args.safety_bps}bps)")
        else:
            print(f"Entry {entry} short → max exit {min_exit:.2f} (incl. taker fees + {args.safety_bps}bps)")
    except Exception as e:
        print(f"❌ Breakeven calculation failed: {e}")


def cmd_perps(args) -> None:
    """Run the perpetuals trading bot (Phase 7 runner)."""
    import subprocess
    from pathlib import Path

    script = Path(__file__).resolve().parents[2] / 'scripts' / 'run_perps_bot.py'
    if not script.exists():
        print(f"❌ Runner script not found at {script}")
        sys.exit(1)

    cmd = [sys.executable, str(script), '--profile', args.profile, '--interval', str(args.interval), '--leverage', str(args.leverage)]
    if args.dry_run:
        cmd.append('--dry-run')
    if args.reduce_only:
        cmd.append('--reduce-only')
    if hasattr(args, 'dev_fast') and args.dev_fast:
        cmd.append('--dev-fast')
    if args.symbols:
        cmd += ['--symbols', *args.symbols]

    print(f"🏁 Launching perps bot: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Perps bot exited with error: {e}")
        sys.exit(e.returncode)


def main() -> None:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Display system status
    print("🚀 GPT-Trader V2 Simple CLI")
    print(f"📦 Backtest: {'✅' if BACKTEST_AVAILABLE else '❌'} | "
          f"Analyze: {'✅' if ANALYZE_AVAILABLE else '❌'} | "
          f"Optimize: {'✅' if OPTIMIZE_AVAILABLE else '❌'}")
    print()
    
    # Execute the command
    try:
        args.func(args)
    except AttributeError:
        print("❌ Unknown command")
        parser.print_help()
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
