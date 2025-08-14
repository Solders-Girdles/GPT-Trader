from __future__ import annotations

import argparse
import asyncio
import sys

from bot.config import get_config
from bot.exec.alpaca_paper import AlpacaPaperBroker
from bot.live.trading_engine import LiveTradingEngine
from bot.portfolio.allocator import PortfolioRules
from bot.strategy.trend_breakout import TrendBreakoutParams, TrendBreakoutStrategy
from rich.panel import Panel

from .cli_utils import CLITheme, console


def add_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> argparse.ArgumentParser:
    p = subparsers.add_parser(
        "paper",
        help="Run paper trading with live market data",
        description="""
        Execute paper trading strategies using live market data from Alpaca.

        Examples:
            # Basic paper trading
            gpt-trader paper --symbols AAPL,MSFT,GOOGL --risk-pct 1.0

            # Paper trading with custom parameters
            gpt-trader paper --symbols SPY,QQQ --risk-pct 0.5 --max-positions 5 \\
                --donchian 55 --atr 20 --atr-k 2.0 --rebalance-interval 300
        """,
    )

    # Strategy configuration
    strategy_group = p.add_argument_group("Strategy Configuration")
    strategy_group.add_argument(
        "--strategy",
        default="trend_breakout",
        choices=["trend_breakout"],
        help="Trading strategy to use (default: trend_breakout)",
    )
    strategy_group.add_argument(
        "--donchian",
        type=int,
        default=55,
        metavar="DAYS",
        help="Donchian channel lookback period (default: 55)",
    )
    strategy_group.add_argument(
        "--atr",
        type=int,
        default=20,
        metavar="DAYS",
        help="ATR period for volatility calculation (default: 20)",
    )
    strategy_group.add_argument(
        "--atr-k",
        type=float,
        default=2.0,
        metavar="MULT",
        help="ATR multiplier for stop-loss (default: 2.0)",
    )

    # Trading parameters
    trading_group = p.add_argument_group("Trading Parameters")
    trading_group.add_argument(
        "--symbols",
        required=True,
        metavar="SYMBOLS",
        help="Comma-separated list of symbols to trade (e.g., AAPL,MSFT,GOOGL)",
    )
    trading_group.add_argument(
        "--risk-pct",
        type=float,
        default=0.5,
        metavar="PCT",
        help="Per-trade risk as percent of equity (default: 0.5)",
    )
    trading_group.add_argument(
        "--max-positions",
        type=int,
        default=10,
        metavar="N",
        help="Maximum concurrent positions (default: 10)",
    )
    trading_group.add_argument(
        "--rebalance-interval",
        type=int,
        default=300,
        metavar="SECONDS",
        help="Rebalance interval in seconds (default: 300)",
    )

    # Execution cost/turnover controls (can also come from profile)
    exec_group = p.add_argument_group("Execution Cost & Turnover Controls")
    exec_group.add_argument(
        "--cost-adjusted-sizing",
        action="store_true",
        help="Reduce risk sizing by expected costs/slippage",
    )
    exec_group.add_argument(
        "--slippage-bps", type=float, default=None, help="Estimated one-way slippage in bps"
    )
    exec_group.add_argument(
        "--exec-max-turnover",
        type=float,
        default=None,
        help="Execution-level turnover cap (L1, 0-1) per rebalance",
    )

    p.set_defaults(func=_handle_enhanced)
    return p


def _handle_enhanced(args: argparse.Namespace) -> None:
    """Enhanced paper trading handler with better UX."""

    # Load application configuration
    app_config = get_config()

    # Validate Alpaca credentials
    if not app_config.alpaca.api_key_id or not app_config.alpaca.api_secret_key:
        console.print(
            CLITheme.error(
                "Alpaca credentials not found. Set ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY."
            )
        )
        sys.exit(1)

    # Parse symbols
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        console.print(CLITheme.error("No valid symbols provided"))
        sys.exit(1)

    # Display configuration
    console.print(
        Panel(
            f"[bold]Paper Trading Configuration[/bold]\n"
            f"Strategy: {args.strategy}\n"
            f"Symbols: {', '.join(symbols)}\n"
            f"Risk per trade: {args.risk_pct}%\n"
            f"Max positions: {args.max_positions}\n"
            f"Rebalance interval: {args.rebalance_interval}s\n"
            f"Cost-adjusted sizing: {'ON' if getattr(args, 'cost_adjusted_sizing', False) else 'OFF'}\n"
            f"Slippage (bps): {args.slippage_bps if args.slippage_bps is not None else 0.0}\n"
            f"Exec max turnover: {args.exec_max_turnover if args.exec_max_turnover is not None else '—'}",
            title="[bold cyan]Starting Paper Trading",
            border_style="cyan",
        )
    )

    # Strategy and rules
    strategy = TrendBreakoutStrategy(
        TrendBreakoutParams(
            donchian_lookback=args.donchian,
            atr_period=args.atr,
            atr_k=args.atr_k,
        )
    )
    rules = PortfolioRules(
        per_trade_risk_pct=args.risk_pct / 100.0,
        atr_k=args.atr_k,
        max_positions=args.max_positions,
        cost_bps=5.0,
        cost_adjusted_sizing=bool(getattr(args, "cost_adjusted_sizing", False)),
        slippage_bps=(
            float(args.slippage_bps) if getattr(args, "slippage_bps", None) is not None else 0.0
        ),
        max_turnover_per_rebalance=(
            float(args.exec_max_turnover)
            if getattr(args, "exec_max_turnover", None) is not None
            else None
        ),
    )

    # Broker and engine
    broker = AlpacaPaperBroker(
        api_key=app_config.alpaca.api_key_id,
        secret_key=app_config.alpaca.api_secret_key,
        base_url=app_config.alpaca.paper_base_url,
    )

    engine = LiveTradingEngine(
        broker=broker,
        strategy=strategy,
        rules=rules,
        symbols=symbols,
        rebalance_interval=args.rebalance_interval,
        max_positions=args.max_positions,
    )

    console.print(CLITheme.success("Paper trading engine initialized. Press Ctrl+C to stop."))

    try:
        asyncio.run(engine.start())
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Stopping paper trading...[/yellow]")
        asyncio.run(engine.stop())
        console.print(CLITheme.success("Paper trading stopped."))


# Keep the original handler for backward compatibility
def _handle(args: argparse.Namespace) -> None:
    """Original paper trading handler for backward compatibility."""
    _handle_enhanced(args)
