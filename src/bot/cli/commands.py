"""
GPT-Trader CLI Commands
All command implementations in one place
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from ..backtest import run_backtest
from ..optimization import run_optimization
from .cli_utils import (
    confirm_action,
    format_percentage,
    print_error,
    print_success,
    print_table,
    print_warning,
)


class BaseCommand:
    """Base class for all commands"""

    def __init__(self, args: argparse.Namespace):
        self.args = args

    def execute(self) -> int:
        """Execute the command"""
        raise NotImplementedError

    @classmethod
    def add_parser(cls, subparsers: argparse._SubParsersAction) -> None:
        """Add command parser"""
        raise NotImplementedError


class BacktestCommand(BaseCommand):
    """Run backtests on trading strategies"""

    @classmethod
    def add_parser(cls, subparsers: argparse._SubParsersAction) -> None:
        parser = subparsers.add_parser(
            "backtest",
            help="Run historical backtests",
            description="Run comprehensive backtests on trading strategies",
        )

        # Data selection
        data_group = parser.add_mutually_exclusive_group(required=True)
        data_group.add_argument("--symbol", help="Single symbol to test")
        data_group.add_argument("--symbol-list", type=Path, help="CSV file with symbols")

        # Date range
        parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
        parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")

        # Strategy configuration
        parser.add_argument(
            "--strategy",
            choices=["demo_ma", "trend_breakout", "mean_reversion"],
            default="trend_breakout",
            help="Strategy to test",
        )

        # Risk parameters
        parser.add_argument("--risk-pct", type=float, default=0.5, help="Risk per trade (%)")
        parser.add_argument(
            "--max-positions", type=int, default=10, help="Max concurrent positions"
        )
        parser.add_argument("--cost-bps", type=float, default=0, help="Transaction costs (bps)")

        # Output options
        parser.add_argument("--output-dir", type=Path, help="Output directory")
        parser.add_argument("--no-plot", action="store_true", help="Skip chart generation")

    def execute(self) -> int:
        """Execute backtest command"""
        try:
            print_success(f"Starting backtest for {self.args.strategy} strategy")

            # Prepare configuration
            config = {
                "symbol": self.args.symbol or self.args.symbol_list,
                "start_date": self.args.start,
                "end_date": self.args.end,
                "strategy": self.args.strategy,
                "risk_pct": self.args.risk_pct,
                "max_positions": self.args.max_positions,
                "transaction_costs": self.args.cost_bps / 10000,
            }

            # Run backtest
            results = run_backtest(**config)

            # Display results
            self._display_results(results)

            # Save results if output directory specified
            if self.args.output_dir:
                self._save_results(results, self.args.output_dir)

            print_success("Backtest completed successfully")
            return 0

        except Exception as e:
            print_error(f"Backtest failed: {e}")
            return 1

    def _display_results(self, results: dict[str, Any]) -> None:
        """Display backtest results"""
        metrics = [
            ["Total Return", format_percentage(results.get("total_return", 0))],
            ["Sharpe Ratio", f"{results.get('sharpe_ratio', 0):.2f}"],
            ["Max Drawdown", format_percentage(results.get("max_drawdown", 0))],
            ["Win Rate", format_percentage(results.get("win_rate", 0))],
            ["Total Trades", str(results.get("total_trades", 0))],
        ]

        print("\nBacktest Results:")
        print_table(["Metric", "Value"], metrics)

    def _save_results(self, results: dict[str, Any], output_dir: Path) -> None:
        """Save results to file"""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save as CSV
        import pandas as pd

        df = pd.DataFrame([results])
        output_file = output_dir / f"backtest_{datetime.now():%Y%m%d_%H%M%S}.csv"
        df.to_csv(output_file, index=False)

        print(f"Results saved to {output_file}")


class OptimizeCommand(BaseCommand):
    """Optimize strategy parameters"""

    @classmethod
    def add_parser(cls, subparsers: argparse._SubParsersAction) -> None:
        parser = subparsers.add_parser(
            "optimize",
            help="Optimize strategy parameters",
            description="Find optimal parameters for trading strategies",
        )

        parser.add_argument("--symbol", required=True, help="Symbol to optimize")
        parser.add_argument("--start", required=True, help="Start date")
        parser.add_argument("--end", required=True, help="End date")
        parser.add_argument("--strategy", required=True, help="Strategy to optimize")

        # Optimization parameters
        parser.add_argument(
            "--param",
            action="append",
            nargs=4,
            metavar=("NAME", "MIN", "MAX", "STEP"),
            help="Parameter to optimize",
        )
        parser.add_argument(
            "--metric",
            default="sharpe",
            choices=["sharpe", "returns", "calmar"],
            help="Optimization metric",
        )
        parser.add_argument(
            "--method",
            default="grid",
            choices=["grid", "random", "bayesian"],
            help="Optimization method",
        )

    def execute(self) -> int:
        """Execute optimization command"""
        try:
            print_success(f"Starting optimization for {self.args.strategy}")

            # Run optimization
            results = run_optimization(
                symbol=self.args.symbol,
                start_date=self.args.start,
                end_date=self.args.end,
                strategy=self.args.strategy,
                parameters=self.args.param,
                metric=self.args.metric,
                method=self.args.method,
            )

            # Display results
            print("\nOptimization Results:")
            print(f"Best {self.args.metric}: {results['best_score']:.3f}")
            print(f"Best parameters: {results['best_params']}")

            return 0

        except Exception as e:
            print_error(f"Optimization failed: {e}")
            return 1


class LiveCommand(BaseCommand):
    """Run live trading (use with extreme caution!)"""

    @classmethod
    def add_parser(cls, subparsers: argparse._SubParsersAction) -> None:
        parser = subparsers.add_parser(
            "live", help="Run live trading", description="⚠️  WARNING: Live trading uses real money!"
        )

        parser.add_argument("--symbol", required=True, help="Symbol(s) to trade")
        parser.add_argument("--strategy", required=True, help="Strategy to use")
        parser.add_argument("--capital", type=float, required=True, help="Capital to allocate")
        parser.add_argument("--risk-pct", type=float, default=0.5, help="Risk per trade")
        parser.add_argument("--confirm", action="store_true", help="Confirm live trading")

    def execute(self) -> int:
        """Execute live trading command"""
        if not self.args.confirm:
            print_error("Live trading requires --confirm flag")
            print_warning("⚠️  WARNING: Live trading uses real money!")
            print_warning("Test with paper trading first")
            return 1

        # Double confirmation for live trading
        if not confirm_action("Are you SURE you want to start LIVE trading with REAL money?"):
            print("Live trading cancelled")
            return 0

        try:
            print_success(f"Starting live trading with ${self.args.capital:,.2f}")

            # Start live trading engine
            from ..live import LiveTradingEngine

            engine = LiveTradingEngine(
                symbols=[self.args.symbol],
                strategy=self.args.strategy,
                capital=self.args.capital,
                risk_pct=self.args.risk_pct,
            )

            engine.start()

            print("Live trading started. Press Ctrl+C to stop")

            # Keep running until interrupted
            try:
                engine.wait()
            except KeyboardInterrupt:
                print("\nStopping live trading...")
                engine.stop()

            return 0

        except Exception as e:
            print_error(f"Live trading failed: {e}")
            return 1


class PaperCommand(BaseCommand):
    """Run paper trading simulation"""

    @classmethod
    def add_parser(cls, subparsers: argparse._SubParsersAction) -> None:
        parser = subparsers.add_parser(
            "paper", help="Run paper trading", description="Test strategies with simulated trading"
        )

        parser.add_argument("--symbol", help="Symbol(s) to trade")
        parser.add_argument("--portfolio", help="Portfolio name")
        parser.add_argument("--strategy", default="trend_breakout", help="Strategy to use")
        parser.add_argument("--capital", type=float, default=100000, help="Starting capital")

    def execute(self) -> int:
        """Execute paper trading command"""
        try:
            print_success(f"Starting paper trading with ${self.args.capital:,.2f}")

            # Check if in demo mode
            import os

            if os.getenv("DEMO_MODE", "false").lower() == "true":
                print_warning("Paper trading requires API credentials (not available in demo mode)")
                print_warning("Set DEMO_MODE=false and configure API keys to use paper trading")
                return 1

            # Start paper trading
            from ..paper_trading import PaperTradingEngine

            engine = PaperTradingEngine(
                symbols=[self.args.symbol] if self.args.symbol else [],
                portfolio=self.args.portfolio,
                strategy=self.args.strategy,
                starting_capital=self.args.capital,
            )

            engine.start()

            print("Paper trading started. Press Ctrl+C to stop")

            try:
                engine.wait()
            except KeyboardInterrupt:
                print("\nStopping paper trading...")
                engine.stop()

            return 0

        except Exception as e:
            print_error(f"Paper trading failed: {e}")
            return 1


class MonitorCommand(BaseCommand):
    """Monitor running strategies and positions"""

    @classmethod
    def add_parser(cls, subparsers: argparse._SubParsersAction) -> None:
        parser = subparsers.add_parser(
            "monitor",
            help="Monitor trading activity",
            description="Real-time monitoring of strategies and positions",
        )

        parser.add_argument("--refresh", type=int, default=5, help="Refresh interval in seconds")
        parser.add_argument(
            "--metrics",
            nargs="+",
            default=["pnl", "positions", "trades"],
            help="Metrics to display",
        )
        parser.add_argument("--export", type=Path, help="Export metrics to file")

    def execute(self) -> int:
        """Execute monitor command"""
        try:
            print_success("Starting monitoring dashboard")

            from ..monitor import TradingMonitor

            monitor = TradingMonitor(refresh_interval=self.args.refresh, metrics=self.args.metrics)

            if self.args.export:
                monitor.enable_export(self.args.export)

            monitor.start()

            print(f"Monitoring active (refresh: {self.args.refresh}s). Press Ctrl+C to stop")

            try:
                monitor.wait()
            except KeyboardInterrupt:
                print("\nStopping monitor...")
                monitor.stop()

            return 0

        except Exception as e:
            print_error(f"Monitor failed: {e}")
            return 1


class DashboardCommand(BaseCommand):
    """Launch web dashboard"""

    @classmethod
    def add_parser(cls, subparsers: argparse._SubParsersAction) -> None:
        parser = subparsers.add_parser(
            "dashboard",
            help="Launch web dashboard",
            description="Start the Streamlit web dashboard",
        )

        parser.add_argument("--port", type=int, default=8501, help="Dashboard port")
        parser.add_argument("--host", default="localhost", help="Dashboard host")
        parser.add_argument(
            "--no-browser", action="store_true", help="Don't open browser automatically"
        )

    def execute(self) -> int:
        """Execute dashboard command"""
        try:
            print_success(f"Launching dashboard on http://{self.args.host}:{self.args.port}")

            # Get dashboard path
            dashboard_path = Path(__file__).parent.parent / "dashboard" / "app.py"

            if not dashboard_path.exists():
                print_error(f"Dashboard not found at {dashboard_path}")
                return 1

            # Launch Streamlit
            cmd = [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(dashboard_path),
                "--server.port",
                str(self.args.port),
                "--server.address",
                self.args.host,
            ]

            if self.args.no_browser:
                cmd.extend(["--server.headless", "true"])

            subprocess.run(cmd)
            return 0

        except KeyboardInterrupt:
            print("\nDashboard stopped")
            return 0
        except Exception as e:
            print_error(f"Dashboard failed: {e}")
            return 1


class WizardCommand(BaseCommand):
    """Setup wizard for configuration"""

    @classmethod
    def add_parser(cls, subparsers: argparse._SubParsersAction) -> None:
        parser = subparsers.add_parser(
            "wizard",
            help="Run setup wizard",
            description="Interactive setup wizard for configuration",
        )

        parser.add_argument("--skip-api", action="store_true", help="Skip API configuration")

    def execute(self) -> int:
        """Execute setup wizard"""
        try:
            print_success("Starting setup wizard")
            print("This wizard will help you configure GPT-Trader\n")

            config = {}

            # API Configuration
            if not self.args.skip_api:
                print("\n=== API Configuration ===")
                config["api"] = self._configure_api()

            # Strategy Selection
            print("\n=== Strategy Configuration ===")
            config["strategy"] = self._configure_strategy()

            # Risk Parameters
            print("\n=== Risk Management ===")
            config["risk"] = self._configure_risk()

            # Save configuration
            self._save_config(config)

            print_success("\nConfiguration complete!")
            print("You can now run: gpt-trader backtest --help")

            return 0

        except KeyboardInterrupt:
            print("\n\nSetup cancelled")
            return 130
        except Exception as e:
            print_error(f"Setup failed: {e}")
            return 1

    def _configure_api(self) -> dict[str, str]:
        """Configure API settings"""
        config = {}

        print("Choose your data/execution provider:")
        print("1. Alpaca (recommended)")
        print("2. Interactive Brokers")
        print("3. Demo mode (no API required)")

        choice = input("\nSelect option [1-3]: ").strip()

        if choice == "3":
            config["provider"] = "demo"
            config["demo_mode"] = "true"
        elif choice == "2":
            config["provider"] = "ibkr"
            config["tws_port"] = input("TWS Port [7497]: ").strip() or "7497"
        else:
            config["provider"] = "alpaca"
            config["api_key"] = input("Alpaca API Key: ").strip()
            config["api_secret"] = input("Alpaca API Secret: ").strip()

            use_paper = input("Use paper trading? [Y/n]: ").strip().lower()
            config["paper_trading"] = "false" if use_paper == "n" else "true"

        return config

    def _configure_strategy(self) -> dict[str, Any]:
        """Configure strategy settings"""
        config = {}

        print("\nSelect default strategy:")
        print("1. Trend Breakout (recommended)")
        print("2. Moving Average Crossover")
        print("3. Mean Reversion")

        choice = input("\nSelect option [1-3]: ").strip()

        strategies = {"1": "trend_breakout", "2": "demo_ma", "3": "mean_reversion"}

        config["default"] = strategies.get(choice, "trend_breakout")

        return config

    def _configure_risk(self) -> dict[str, Any]:
        """Configure risk parameters"""
        config = {}

        config["risk_per_trade"] = float(input("\nRisk per trade (%) [1.0]: ").strip() or "1.0")

        config["max_positions"] = int(input("Maximum concurrent positions [10]: ").strip() or "10")

        config["stop_loss"] = float(input("Stop loss (%) [2.0]: ").strip() or "2.0")

        return config

    def _save_config(self, config: dict[str, Any]) -> None:
        """Save configuration to file"""
        import json

        config_dir = Path.home() / ".gpt-trader"
        config_dir.mkdir(exist_ok=True)

        config_file = config_dir / "config.json"

        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)

        print(f"\nConfiguration saved to {config_file}")

        # Also create .env.local if API configured
        if "api" in config and config["api"].get("provider") != "demo":
            env_file = Path(".env.local")
            with open(env_file, "w") as f:
                if config["api"]["provider"] == "alpaca":
                    f.write(f"ALPACA_API_KEY_ID={config['api']['api_key']}\n")
                    f.write(f"ALPACA_API_SECRET_KEY={config['api']['api_secret']}\n")

                    if config["api"].get("paper_trading") == "true":
                        f.write("ALPACA_PAPER_BASE_URL=https://paper-api.alpaca.markets\n")

            print(f"API credentials saved to {env_file}")
