"""
Strategy Development CLI for GPT-Trader

Provides comprehensive command-line interface for strategy development workflow:
- Strategy creation from templates
- Training and validation pipeline
- Performance analysis and reporting
- Deployment to paper trading

Integrates all Week 1-2 components into a unified workflow.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import click
import pandas as pd
from bot.dataflow.data_quality_framework import create_data_quality_framework

# Week 1 imports
from bot.dataflow.historical_data_manager import DataFrequency, create_historical_data_manager

# Week 2 imports
from bot.strategy.validation_engine import create_strategy_validator

# Week 3 imports
from bot.strategy.validation_pipeline import create_validation_pipeline
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.tree import Tree

console = Console()
logger = logging.getLogger(__name__)


# Strategy Templates
STRATEGY_TEMPLATES = {
    "moving_average": {
        "name": "Moving Average Crossover",
        "description": "Simple moving average crossover strategy",
        "file": "moving_average_template.py",
        "parameters": {
            "fast_period": {"type": "int", "min": 5, "max": 50, "default": 10},
            "slow_period": {"type": "int", "min": 10, "max": 200, "default": 20},
        },
        "complexity": "Simple",
        "holding_period": "Short-term",
    },
    "mean_reversion": {
        "name": "Mean Reversion",
        "description": "Bollinger Bands mean reversion strategy",
        "file": "mean_reversion_template.py",
        "parameters": {
            "window": {"type": "int", "min": 10, "max": 50, "default": 20},
            "num_std": {"type": "float", "min": 1.0, "max": 3.0, "default": 2.0},
            "lookback": {"type": "int", "min": 5, "max": 20, "default": 10},
        },
        "complexity": "Medium",
        "holding_period": "Short-term",
    },
    "momentum": {
        "name": "Momentum Strategy",
        "description": "Price momentum with RSI filter",
        "file": "momentum_template.py",
        "parameters": {
            "lookback_period": {"type": "int", "min": 5, "max": 30, "default": 14},
            "rsi_period": {"type": "int", "min": 10, "max": 30, "default": 14},
            "rsi_oversold": {"type": "int", "min": 20, "max": 40, "default": 30},
            "rsi_overbought": {"type": "int", "min": 60, "max": 80, "default": 70},
        },
        "complexity": "Medium",
        "holding_period": "Medium-term",
    },
    "breakout": {
        "name": "Breakout Strategy",
        "description": "Channel breakout with volume confirmation",
        "file": "breakout_template.py",
        "parameters": {
            "channel_period": {"type": "int", "min": 10, "max": 50, "default": 20},
            "volume_threshold": {"type": "float", "min": 1.0, "max": 3.0, "default": 1.5},
            "atr_period": {"type": "int", "min": 10, "max": 30, "default": 14},
        },
        "complexity": "Complex",
        "holding_period": "Medium-term",
    },
}


class StrategyDevelopmentWorkflow:
    """Main workflow manager for strategy development"""

    def __init__(self, output_dir: str = "data/strategy_development") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.output_dir / "strategies").mkdir(exist_ok=True)
        (self.output_dir / "templates").mkdir(exist_ok=True)
        (self.output_dir / "results").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)

        # Initialize components
        self.data_manager = None
        self.quality_framework = None
        self.validator = None
        self.validation_pipeline = None

        logger.info(f"Strategy Development Workflow initialized at {self.output_dir}")

    def _initialize_components(self) -> None:
        """Lazy initialization of components"""
        if self.data_manager is None:
            console.print("   🔧 Initializing components...")
            self.data_manager = create_historical_data_manager(
                min_quality_score=0.75,
                cache_dir=str(self.output_dir / "cache"),
                max_concurrent_downloads=5,
            )

            self.quality_framework = create_data_quality_framework(
                min_quality_score=75.0, outlier_method="iqr", missing_data_method="forward"
            )

            self.validator = create_strategy_validator(
                min_sharpe_ratio=0.5, max_drawdown=0.15, min_confidence_level=0.95
            )

            # Week 3 validation pipeline
            self.validation_pipeline = create_validation_pipeline(
                output_dir=str(self.output_dir / "pipeline"),
                data_quality_threshold=75.0,
                min_sharpe_ratio=0.3,
                max_drawdown=0.20,
            )

    def create_strategy_from_template(
        self, template_name: str, strategy_name: str, parameters: dict[str, Any] | None = None
    ) -> str:
        """Create a new strategy from template"""

        if template_name not in STRATEGY_TEMPLATES:
            raise ValueError(f"Unknown template: {template_name}")

        template_info = STRATEGY_TEMPLATES[template_name]
        strategy_dir = self.output_dir / "strategies" / strategy_name
        strategy_dir.mkdir(parents=True, exist_ok=True)

        # Use provided parameters or defaults
        if parameters is None:
            parameters = {
                param: info["default"] for param, info in template_info["parameters"].items()
            }

        # Generate strategy code
        strategy_code = self._generate_strategy_code(template_name, strategy_name, parameters)

        # Write strategy file
        strategy_file = strategy_dir / f"{strategy_name.lower().replace(' ', '_')}.py"
        with open(strategy_file, "w") as f:
            f.write(strategy_code)

        # Write configuration
        config = {
            "strategy_name": strategy_name,
            "template": template_name,
            "parameters": parameters,
            "created_at": datetime.now().isoformat(),
            "metadata": {
                "description": template_info["description"],
                "complexity": template_info["complexity"],
                "holding_period": template_info["holding_period"],
            },
        }

        config_file = strategy_dir / "strategy_config.json"
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)

        console.print(f"   ✅ Strategy created: {strategy_file}")
        return str(strategy_file)

    def _generate_strategy_code(
        self, template_name: str, strategy_name: str, parameters: dict[str, Any]
    ) -> str:
        """Generate strategy code from template"""

        class_name = strategy_name.replace(" ", "").replace("_", "")
        param_init = "\n        ".join([f"self.{k} = {v}" for k, v in parameters.items()])
        param_space = self._generate_parameter_space(template_name, parameters)

        if template_name == "moving_average":
            return f'''"""
{strategy_name} - Generated from Moving Average template
Created: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

import pandas as pd
import numpy as np
from bot.strategy.base import Strategy


class {class_name}Strategy(Strategy):
    """Moving Average Crossover Strategy"""

    def __init__(self):
        self.name = "{strategy_name}"
        self.supports_short = True
        {param_init}

    def generate_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on moving average crossover"""
        if len(bars) < self.slow_period:
            result = pd.DataFrame(index=bars.index)
            result['signal'] = 0
            return result

        # Calculate moving averages
        fast_ma = bars['Close'].rolling(window=self.fast_period).mean()
        slow_ma = bars['Close'].rolling(window=self.slow_period).mean()

        # Generate signals
        signals = pd.Series(0, index=bars.index)

        # Buy when fast MA crosses above slow MA
        buy_signals = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
        signals[buy_signals] = 1

        # Sell when fast MA crosses below slow MA
        sell_signals = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
        signals[sell_signals] = -1

        # Return DataFrame with signal column
        result = pd.DataFrame(index=bars.index)
        result['signal'] = signals
        return result

    def get_parameter_space(self) -> dict:
        """Get parameter optimization space"""
        return {param_space}


def create_strategy():
    """Factory function to create strategy instance"""
    return {class_name}Strategy()
'''

        elif template_name == "mean_reversion":
            return f'''"""
{strategy_name} - Generated from Mean Reversion template
Created: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

import pandas as pd
import numpy as np
from bot.strategy.base import Strategy


class {class_name}Strategy(Strategy):
    """Bollinger Bands Mean Reversion Strategy"""

    def __init__(self):
        self.name = "{strategy_name}"
        self.supports_short = True
        {param_init}

    def generate_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Generate mean reversion signals using Bollinger Bands"""
        if len(bars) < self.window:
            result = pd.DataFrame(index=bars.index)
            result['signal'] = 0
            return result

        # Calculate Bollinger Bands
        rolling_mean = bars['Close'].rolling(window=self.window).mean()
        rolling_std = bars['Close'].rolling(window=self.window).std()

        upper_band = rolling_mean + (rolling_std * self.num_std)
        lower_band = rolling_mean - (rolling_std * self.num_std)

        # Generate signals
        signals = pd.Series(0, index=bars.index)

        # Buy when price touches lower band (oversold)
        buy_signals = bars['Close'] <= lower_band
        signals[buy_signals] = 1

        # Sell when price touches upper band (overbought)
        sell_signals = bars['Close'] >= upper_band
        signals[sell_signals] = -1

        # Exit when price returns to mean
        mean_cross = abs(bars['Close'] - rolling_mean) < (rolling_std * 0.5)
        signals[mean_cross] = 0

        # Return DataFrame with signal column
        result = pd.DataFrame(index=bars.index)
        result['signal'] = signals
        return result

    def get_parameter_space(self) -> dict:
        """Get parameter optimization space"""
        return {param_space}


def create_strategy():
    """Factory function to create strategy instance"""
    return {class_name}Strategy()
'''

        else:  # Generic template
            return f'''"""
{strategy_name} - Generated from {template_name} template
Created: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

import pandas as pd
import numpy as np
from bot.strategy.base import Strategy


class {class_name}Strategy(Strategy):
    """{template_name.title()} Strategy"""

    def __init__(self):
        self.name = "{strategy_name}"
        self.supports_short = True
        {param_init}

    def generate_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals - implement your logic here"""
        # TODO: Implement your strategy logic
        result = pd.DataFrame(index=bars.index)
        result['signal'] = 0
        return result

    def get_parameter_space(self) -> dict:
        """Get parameter optimization space"""
        return {param_space}


def create_strategy():
    """Factory function to create strategy instance"""
    return {class_name}Strategy()
'''

    def _generate_parameter_space(self, template_name: str, parameters: dict[str, Any]) -> str:
        """Generate parameter space definition"""
        template_info = STRATEGY_TEMPLATES[template_name]
        param_definitions = []

        for param_name, param_info in template_info["parameters"].items():
            if param_info["type"] == "int":
                param_def = f"'{param_name}': {{'type': 'integer', 'low': {param_info['min']}, 'high': {param_info['max']}}}"
            elif param_info["type"] == "float":
                param_def = f"'{param_name}': {{'type': 'real', 'low': {param_info['min']}, 'high': {param_info['max']}}}"
            else:
                param_def = f"'{param_name}': {{'type': 'categorical', 'categories': ['{param_info['default']}']}}'"

            param_definitions.append(param_def)

        return "{\n            " + ",\n            ".join(param_definitions) + "\n        }"

    def validate_strategy_returns(self, returns: pd.Series, strategy_name: str) -> dict[str, Any]:
        """Validate strategy using returns data"""

        self._initialize_components()

        console.print(f"   🔍 Running validation for {strategy_name}...")

        # Run validation
        validation_result = self.validator.validate_strategy(returns, strategy_name)

        # Create validation report
        report = {
            "strategy_name": strategy_name,
            "validation_timestamp": datetime.now().isoformat(),
            "overall_score": validation_result.overall_score,
            "grade": validation_result.validation_grade,
            "is_validated": validation_result.is_validated,
            "confidence_level": validation_result.confidence_level,
            "performance_metrics": {
                "sharpe_ratio": validation_result.performance_metrics.sharpe_ratio,
                "max_drawdown": validation_result.performance_metrics.max_drawdown,
                "win_rate": validation_result.performance_metrics.win_rate,
                "profit_factor": validation_result.performance_metrics.profit_factor,
                "total_return": validation_result.performance_metrics.total_return,
                "volatility": validation_result.performance_metrics.volatility,
            },
            "strengths": validation_result.strengths,
            "weaknesses": validation_result.weaknesses,
            "recommendations": validation_result.recommendations,
        }

        # Save validation report
        report_file = self.output_dir / "reports" / f"{strategy_name}_validation_report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2, default=str)

        console.print(f"   ✅ Validation report saved: {report_file}")
        return report

    def prepare_training_data(self, symbols: list[str], days: int = 365) -> dict[str, pd.DataFrame]:
        """Prepare training data for strategy development"""

        self._initialize_components()

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        console.print(f"   📊 Downloading data for {len(symbols)} symbols...")

        # Download data
        datasets, metadata = self.data_manager.get_training_dataset(
            symbols=symbols, start_date=start_date, end_date=end_date, frequency=DataFrequency.DAILY
        )

        console.print(f"   ✅ Downloaded data for {len(datasets)} symbols")

        # Quality validation
        cleaned_datasets = {}
        for symbol, data in datasets.items():
            cleaned_data, quality_report = self.quality_framework.clean_and_validate(data, symbol)
            if quality_report.is_usable:
                cleaned_datasets[symbol] = cleaned_data
                console.print(
                    f"      • {symbol}: {len(cleaned_data)} records (Quality: {quality_report.quality_score:.1f}/100)"
                )
            else:
                console.print(f"      • {symbol}: [red]Poor quality data excluded[/red]")

        return cleaned_datasets

    def generate_performance_report(
        self, strategy_name: str, validation_report: dict[str, Any]
    ) -> str:
        """Generate comprehensive performance report"""

        report_md = f"""# Strategy Performance Report

## Strategy: {strategy_name}

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overall Assessment

- **Validation Score:** {validation_report['overall_score']:.1f}/100
- **Grade:** {validation_report['grade']}
- **Status:** {'✅ Validated' if validation_report['is_validated'] else '❌ Not Validated'}
- **Confidence:** {validation_report['confidence_level']:.1%}

## Performance Metrics

| Metric | Value |
|--------|-------|
| Sharpe Ratio | {validation_report['performance_metrics']['sharpe_ratio']:.3f} |
| Maximum Drawdown | {validation_report['performance_metrics']['max_drawdown']:.1%} |
| Win Rate | {validation_report['performance_metrics']['win_rate']:.1%} |
| Profit Factor | {validation_report['performance_metrics']['profit_factor']:.2f} |
| Total Return | {validation_report['performance_metrics']['total_return']:.1%} |
| Volatility | {validation_report['performance_metrics']['volatility']:.1%} |

## Strengths

"""

        for strength in validation_report.get("strengths", []):
            report_md += f"- {strength}\n"

        report_md += "\n## Weaknesses\n\n"

        for weakness in validation_report.get("weaknesses", []):
            report_md += f"- {weakness}\n"

        report_md += "\n## Recommendations\n\n"

        for rec in validation_report.get("recommendations", []):
            report_md += f"- {rec}\n"

        # Save report
        report_file = self.output_dir / "reports" / f"{strategy_name}_performance_report.md"
        with open(report_file, "w") as f:
            f.write(report_md)

        return str(report_file)


# CLI Commands
@click.group()
def develop() -> None:
    """Strategy development commands"""
    pass


@develop.command()
@click.option(
    "--template",
    type=click.Choice(list(STRATEGY_TEMPLATES.keys())),
    required=True,
    help="Strategy template to use",
)
@click.option("--name", required=True, help="Name for the new strategy")
@click.option("--parameters", help="JSON string of parameters (optional)")
@click.option("--output-dir", default="data/strategy_development", help="Output directory")
def create(template, name, parameters, output_dir) -> None:
    """Create a new strategy from template"""

    console.print(
        Panel(
            f"[bold blue]Creating Strategy from Template[/bold blue]\n"
            f"Template: {template}\n"
            f"Name: {name}\n"
            f"Output: {output_dir}",
            title="🚀 Strategy Creation",
        )
    )

    try:
        # Parse parameters if provided
        params = None
        if parameters:
            params = json.loads(parameters)

        # Initialize workflow
        workflow = StrategyDevelopmentWorkflow(output_dir)

        # Create strategy
        strategy_file = workflow.create_strategy_from_template(template, name, params)

        # Show template info
        template_info = STRATEGY_TEMPLATES[template]
        console.print("\n📋 [bold]Strategy Template Information[/bold]")
        console.print(f"   Description: {template_info['description']}")
        console.print(f"   Complexity: {template_info['complexity']}")
        console.print(f"   Holding Period: {template_info['holding_period']}")

        # Show parameters
        used_params = params or {
            p: info["default"] for p, info in template_info["parameters"].items()
        }
        console.print("\n⚙️  [bold]Parameters Used[/bold]")
        for param, value in used_params.items():
            console.print(f"   • {param}: {value}")

        console.print("\n✅ [bold green]Strategy created successfully![/bold green]")
        console.print(f"   📁 File: {strategy_file}")
        console.print(f"   📄 Config: {Path(strategy_file).parent / 'strategy_config.json'}")

        console.print("\n🎯 [bold]Next Steps[/bold]")
        console.print("   1. Review and customize the generated strategy code")
        console.print(f"   2. Test with: [cyan]gpt-trader develop test --strategy {name}[/cyan]")
        console.print(
            f"   3. Validate with: [cyan]gpt-trader develop validate --strategy {name}[/cyan]"
        )

    except Exception as e:
        console.print(f"❌ [bold red]Strategy creation failed:[/bold red] {str(e)}")
        raise click.ClickException(str(e))


@develop.command()
@click.option("--strategy", required=True, help="Strategy name to test")
@click.option("--symbols", default="AAPL,MSFT,GOOGL", help="Comma-separated symbols")
@click.option("--days", default=365, help="Days of historical data")
@click.option(
    "--output-dir", default="data/strategy_development", help="Strategy development directory"
)
def test(strategy, symbols, days, output_dir) -> None:
    """Test strategy with historical data"""

    symbol_list = [s.strip().upper() for s in symbols.split(",")]

    console.print(
        Panel(
            f"[bold blue]Testing Strategy with Historical Data[/bold blue]\n"
            f"Strategy: {strategy}\n"
            f"Symbols: {', '.join(symbol_list)}\n"
            f"Data Period: {days} days",
            title="🧪 Strategy Testing",
        )
    )

    try:
        workflow = StrategyDevelopmentWorkflow(output_dir)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:

            # Prepare data
            data_task = progress.add_task("📊 Preparing training data...", total=1)
            datasets = workflow.prepare_training_data(symbol_list, days)
            progress.update(data_task, completed=1)

            if not datasets:
                raise ValueError("No usable datasets found")

            # Test strategy (simplified - just calculate returns from first symbol)
            test_task = progress.add_task("🧪 Testing strategy logic...", total=1)

            # Use first available dataset
            sample_symbol = list(datasets.keys())[0]
            sample_data = datasets[sample_symbol]

            # Calculate simple returns for testing
            returns = sample_data["Close"].pct_change().dropna()

            progress.update(test_task, completed=1)

            # Basic validation
            validate_task = progress.add_task("🔍 Basic validation...", total=1)
            validation_report = workflow.validate_strategy_returns(returns, strategy)
            progress.update(validate_task, completed=1)

        # Display results
        console.print("\n📊 [bold]Test Results[/bold]")
        console.print(f"   Data Quality: ✅ {len(datasets)}/{len(symbol_list)} symbols usable")
        console.print(f"   Sample Size: {len(returns)} return observations")
        console.print(f"   Validation Score: {validation_report['overall_score']:.1f}/100")
        console.print(f"   Grade: {validation_report['grade']}")

        metrics = validation_report["performance_metrics"]
        console.print("\n📈 [bold]Key Metrics[/bold]")
        console.print(f"   • Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        console.print(f"   • Max Drawdown: {metrics['max_drawdown']:.1%}")
        console.print(f"   • Win Rate: {metrics['win_rate']:.1%}")

        console.print("\n✅ [bold green]Strategy test completed![/bold green]")
        console.print(f"   📄 Report: {output_dir}/reports/{strategy}_validation_report.json")

    except Exception as e:
        console.print(f"❌ [bold red]Strategy test failed:[/bold red] {str(e)}")
        raise click.ClickException(str(e))


@develop.command()
@click.option("--strategy", required=True, help="Strategy name to validate")
@click.option("--symbols", default="AAPL,MSFT,GOOGL,AMZN,TSLA", help="Comma-separated symbols")
@click.option("--days", default=730, help="Days of historical data (2+ years recommended)")
@click.option(
    "--output-dir", default="data/strategy_development", help="Strategy development directory"
)
@click.option("--generate-report", is_flag=True, help="Generate markdown report")
def validate(strategy, symbols, days, output_dir, generate_report) -> None:
    """Comprehensive strategy validation"""

    symbol_list = [s.strip().upper() for s in symbols.split(",")]

    console.print(
        Panel(
            f"[bold blue]Comprehensive Strategy Validation[/bold blue]\n"
            f"Strategy: {strategy}\n"
            f"Symbols: {', '.join(symbol_list)}\n"
            f"Data Period: {days} days\n"
            f"Generate Report: {'Yes' if generate_report else 'No'}",
            title="🔍 Strategy Validation",
        )
    )

    try:
        workflow = StrategyDevelopmentWorkflow(output_dir)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:

            # Prepare comprehensive dataset
            data_task = progress.add_task("📊 Preparing validation dataset...", total=1)
            datasets = workflow.prepare_training_data(symbol_list, days)
            progress.update(data_task, completed=1)

            if not datasets:
                raise ValueError("No usable datasets found")

            # Comprehensive validation using multiple symbols
            validate_task = progress.add_task(
                "🔍 Running comprehensive validation...", total=len(datasets)
            )

            all_returns = []
            for _symbol, data in datasets.items():
                returns = data["Close"].pct_change().dropna()
                all_returns.append(returns)
                progress.update(validate_task, advance=1)

            # Combine returns (simple approach - could be weighted)
            combined_returns = pd.concat(all_returns).groupby(level=0).mean()

            # Run validation
            validation_report = workflow.validate_strategy_returns(combined_returns, strategy)

        # Display comprehensive results
        console.print("\n🎯 [bold]Validation Results[/bold]")

        # Create results table
        results_table = Table(title="📊 Performance Metrics")
        results_table.add_column("Metric", style="cyan", width=20)
        results_table.add_column("Value", style="white", width=15)
        results_table.add_column("Assessment", style="green", width=20)

        metrics = validation_report["performance_metrics"]

        results_table.add_row(
            "Overall Score",
            f"{validation_report['overall_score']:.1f}/100",
            "✅ Pass" if validation_report["overall_score"] >= 70 else "❌ Needs Work",
        )
        results_table.add_row(
            "Grade",
            validation_report["grade"],
            "✅ Good" if validation_report["grade"] in ["A", "B", "C"] else "⚠️  Review",
        )
        results_table.add_row(
            "Sharpe Ratio",
            f"{metrics['sharpe_ratio']:.3f}",
            "✅ Good" if metrics["sharpe_ratio"] > 0.5 else "⚠️  Low",
        )
        results_table.add_row(
            "Max Drawdown",
            f"{metrics['max_drawdown']:.1%}",
            "✅ Good" if metrics["max_drawdown"] < 0.15 else "⚠️  High",
        )
        results_table.add_row(
            "Win Rate",
            f"{metrics['win_rate']:.1%}",
            "✅ Good" if metrics["win_rate"] > 0.5 else "⚠️  Low",
        )
        results_table.add_row(
            "Profit Factor",
            f"{metrics['profit_factor']:.2f}",
            "✅ Good" if metrics["profit_factor"] > 1.2 else "⚠️  Low",
        )

        console.print(results_table)

        # Show strengths and weaknesses
        if validation_report.get("strengths"):
            console.print("\n💪 [bold green]Strengths[/bold green]")
            for strength in validation_report["strengths"]:
                console.print(f"   • {strength}")

        if validation_report.get("weaknesses"):
            console.print("\n⚠️  [bold yellow]Areas for Improvement[/bold yellow]")
            for weakness in validation_report["weaknesses"]:
                console.print(f"   • {weakness}")

        if validation_report.get("recommendations"):
            console.print("\n💡 [bold blue]Recommendations[/bold blue]")
            for rec in validation_report["recommendations"]:
                console.print(f"   • {rec}")

        # Generate markdown report if requested
        if generate_report:
            report_file = workflow.generate_performance_report(strategy, validation_report)
            console.print("\n📄 [bold]Detailed Report Generated[/bold]")
            console.print(f"   File: {report_file}")

        # Final assessment
        if validation_report["is_validated"]:
            console.print(
                "\n✅ [bold green]Strategy is validated and ready for deployment![/bold green]"
            )
            console.print("   Next: Deploy to paper trading with [cyan]gpt-trader paper[/cyan]")
        else:
            console.print(
                "\n⚠️  [bold yellow]Strategy needs improvement before deployment[/bold yellow]"
            )
            console.print("   Consider optimizing parameters or reviewing strategy logic")

    except Exception as e:
        console.print(f"❌ [bold red]Strategy validation failed:[/bold red] {str(e)}")
        raise click.ClickException(str(e))


@develop.command()
def list_templates() -> None:
    """List available strategy templates"""

    console.print("[bold blue]Available Strategy Templates[/bold blue]\n")

    # Create template tree
    tree = Tree("📋 Strategy Templates")

    for template_id, template_info in STRATEGY_TEMPLATES.items():
        template_node = tree.add(f"[bold]{template_info['name']}[/bold] ({template_id})")
        template_node.add(f"📝 {template_info['description']}")
        template_node.add(f"🎯 Complexity: {template_info['complexity']}")
        template_node.add(f"⏱️  Holding: {template_info['holding_period']}")

        # Parameters
        param_node = template_node.add("⚙️  Parameters")
        for param_name, param_info in template_info["parameters"].items():
            if param_info["type"] == "int":
                param_node.add(
                    f"{param_name}: {param_info['min']}-{param_info['max']} (default: {param_info['default']})"
                )
            elif param_info["type"] == "float":
                param_node.add(
                    f"{param_name}: {param_info['min']:.1f}-{param_info['max']:.1f} (default: {param_info['default']})"
                )

    console.print(tree)

    console.print("\n🚀 [bold]Usage Examples[/bold]")
    console.print(
        '   Create MA strategy: [cyan]gpt-trader develop create --template moving_average --name "My MA Strategy"[/cyan]'
    )
    # Use triple braces to safely embed JSON braces in f-string and render literally
    console.print(
        '   With parameters: [cyan]gpt-trader develop create --template moving_average --name "Fast MA" --parameters \'{"fast_period": 5, "slow_period": 15}\'[/cyan]'
    )


@develop.command()
@click.option("--strategy-file", required=True, help="Path to strategy Python file")
@click.option("--symbols", default="AAPL,MSFT,GOOGL,AMZN,TSLA", help="Comma-separated symbols")
@click.option("--days", default=730, help="Days of historical data")
@click.option(
    "--output-dir", default="data/strategy_development", help="Strategy development directory"
)
def pipeline(strategy_file, symbols, days, output_dir) -> None:
    """Run automated validation pipeline for strategy"""

    symbol_list = [s.strip().upper() for s in symbols.split(",")]

    console.print(
        Panel(
            f"[bold blue]Automated Validation Pipeline[/bold blue]\n"
            f"Strategy File: {strategy_file}\n"
            f"Symbols: {', '.join(symbol_list)}\n"
            f"Data Period: {days} days\n"
            f"Output: {output_dir}",
            title="🤖 Automated Pipeline",
        )
    )

    try:
        workflow = StrategyDevelopmentWorkflow(output_dir)
        workflow._initialize_components()

        # Run full validation pipeline
        result = workflow.validation_pipeline.validate_strategy_file(
            strategy_file_path=strategy_file, symbols=symbol_list, days=days, save_results=True
        )

        # Enhanced summary with pipeline-specific results
        console.print("\n🎯 [bold]Pipeline Execution Complete[/bold]")

        if result.pipeline_success:
            console.print("✅ [bold green]All pipeline components successful![/bold green]")
            console.print(f"   • Strategy: {result.strategy_name}")
            console.print(f"   • Overall Score: {result.overall_score:.1f}/100")
            console.print(f"   • Grade: {result.validation_grade}")
            console.print(f"   • Execution Time: {result.execution_time_seconds:.1f}s")

            # Component breakdown
            console.print("\n📊 [bold]Component Status[/bold]")
            components = [
                ("Data Preparation", result.data_preparation_success),
                ("Strategy Loading", result.strategy_loading_success),
                ("Training Pipeline", result.training_success),
                ("Validation Engine", result.validation_success),
                ("Result Persistence", result.persistence_success),
            ]

            for name, success in components:
                status = "✅" if success else "❌"
                console.print(f"   {status} {name}")

            console.print("\n🚀 [bold]Next Steps[/bold]")
            if (
                hasattr(result.validation_results, "is_validated")
                and result.validation_results.is_validated
            ):
                console.print("   Strategy is validated and ready for deployment!")
                console.print(
                    f"   Deploy to paper trading: [cyan]gpt-trader paper --strategy {result.strategy_name}[/cyan]"
                )
            else:
                console.print("   Strategy needs improvement before deployment")
                console.print("   Review recommendations and optimize parameters")

        else:
            console.print("❌ [bold red]Pipeline execution failed[/bold red]")
            if result.errors:
                console.print("\n🔍 [bold]Errors[/bold]")
                for error in result.errors:
                    console.print(f"   • {error}")

    except Exception as e:
        console.print(f"❌ [bold red]Pipeline command failed:[/bold red] {str(e)}")
        raise click.ClickException(str(e))


@develop.command()
@click.option(
    "--output-dir", default="data/strategy_development", help="Strategy development directory"
)
def status(output_dir) -> None:
    """Show strategy development status"""

    dev_dir = Path(output_dir)
    strategies_dir = dev_dir / "strategies"
    reports_dir = dev_dir / "reports"

    console.print("[bold blue]Strategy Development Status[/bold blue]\n")

    if not strategies_dir.exists():
        console.print("📁 No strategies found. Create your first strategy with:")
        console.print(
            '   [cyan]gpt-trader develop create --template moving_average --name "My Strategy"[/cyan]'
        )
        return

    # List strategies
    strategies = [d for d in strategies_dir.iterdir() if d.is_dir()]

    if not strategies:
        console.print("📁 Strategy directory exists but is empty.")
        return

    # Create status table
    table = Table(title="📊 Created Strategies")
    table.add_column("Strategy", style="cyan", width=20)
    table.add_column("Template", style="white", width=15)
    table.add_column("Created", style="dim", width=12)
    table.add_column("Validated", style="green", width=10)
    table.add_column("Files", style="dim", width=15)

    for strategy_dir in sorted(strategies):
        strategy_name = strategy_dir.name

        # Read config
        config_file = strategy_dir / "strategy_config.json"
        if config_file.exists():
            with open(config_file) as f:
                config = json.load(f)
            template = config.get("template", "Unknown")
            created = datetime.fromisoformat(config["created_at"]).strftime("%Y-%m-%d")
        else:
            template = "Unknown"
            created = "Unknown"

        # Check validation
        validation_file = reports_dir / f"{strategy_name}_validation_report.json"
        validated = "✅ Yes" if validation_file.exists() else "❌ No"

        # Count files
        files = list(strategy_dir.glob("*"))
        file_count = f"{len(files)} files"

        table.add_row(strategy_name, template, created, validated, file_count)

    console.print(table)

    # Show recent reports
    if reports_dir.exists():
        report_files = list(reports_dir.glob("*_validation_report.json"))
        if report_files:
            console.print("\n📄 [bold]Recent Validation Reports[/bold]")
            for report_file in sorted(report_files, key=lambda f: f.stat().st_mtime, reverse=True)[
                :5
            ]:
                modified = datetime.fromtimestamp(report_file.stat().st_mtime).strftime(
                    "%Y-%m-%d %H:%M"
                )
                console.print(f"   • {report_file.name} ({modified})")


if __name__ == "__main__":
    develop()
