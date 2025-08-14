"""
Validation Pipeline Integration for GPT-Trader Strategy Development

Provides automated end-to-end validation workflows integrating:
- Historical Data Manager (Week 1)
- Data Quality Framework (Week 1)
- Strategy Training Pipeline (Week 2)
- Strategy Validation Engine (Week 2)
- Strategy Development CLI (Week 3)

This creates a unified pipeline for automated strategy testing and validation.
"""

import importlib.util
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from bot.dataflow.data_quality_framework import create_data_quality_framework

# Week 1 imports
from bot.dataflow.historical_data_manager import DataFrequency, create_historical_data_manager

# Week 3 imports
from bot.strategy.base import Strategy
from bot.strategy.persistence import create_strategy_persistence_manager

# Week 2 imports
from bot.strategy.training_pipeline import create_strategy_training_pipeline
from bot.strategy.validation_engine import create_strategy_validator
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table

console = Console()
logger = logging.getLogger(__name__)


class ValidationPipelineError(Exception):
    """Base exception for validation pipeline errors"""

    pass


class StrategyLoadError(ValidationPipelineError):
    """Error loading strategy for validation"""

    pass


class ValidationPipelineResult:
    """Result of validation pipeline execution"""

    def __init__(self) -> None:
        self.strategy_name: str = ""
        self.pipeline_success: bool = False
        self.data_preparation_success: bool = False
        self.strategy_loading_success: bool = False
        self.training_success: bool = False
        self.validation_success: bool = False
        self.persistence_success: bool = False

        # Detailed results
        self.data_quality_scores: dict[str, float] = {}
        self.training_results: Any | None = None
        self.validation_results: Any | None = None
        self.persistence_results: Any | None = None

        # Performance metrics
        self.overall_score: float = 0.0
        self.validation_grade: str = "F"
        self.sharpe_ratio: float = 0.0
        self.max_drawdown: float = 0.0

        # Execution metadata
        self.execution_time_seconds: float = 0.0
        self.timestamp: datetime = datetime.now()
        self.errors: list[str] = []
        self.warnings: list[str] = []


class ValidationPipeline:
    """Automated validation pipeline for strategy development"""

    def __init__(
        self,
        output_dir: str = "data/validation_pipeline",
        data_quality_threshold: float = 75.0,
        min_sharpe_ratio: float = 0.3,
        max_drawdown: float = 0.20,
    ) -> None:

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.output_dir / "data").mkdir(exist_ok=True)
        (self.output_dir / "strategies").mkdir(exist_ok=True)
        (self.output_dir / "results").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)

        # Configuration
        self.data_quality_threshold = data_quality_threshold
        self.min_sharpe_ratio = min_sharpe_ratio
        self.max_drawdown = max_drawdown

        # Components (lazy initialization)
        self.data_manager = None
        self.quality_framework = None
        self.training_pipeline = None
        self.validation_engine = None
        self.persistence_manager = None

        # Pipeline state
        self.current_result: ValidationPipelineResult | None = None
        self.validation_history: list[ValidationPipelineResult] = []

        logger.info(f"Validation Pipeline initialized at {self.output_dir}")

    def _initialize_components(self) -> None:
        """Lazy initialization of all pipeline components"""
        if self.data_manager is None:
            console.print("   🔧 Initializing pipeline components...")

            # Week 1 components
            self.data_manager = create_historical_data_manager(
                min_quality_score=self.data_quality_threshold / 100.0,
                cache_dir=str(self.output_dir / "data" / "cache"),
                max_concurrent_downloads=3,
            )

            self.quality_framework = create_data_quality_framework(
                min_quality_score=self.data_quality_threshold,
                outlier_method="iqr",
                missing_data_method="forward",
            )

            # Week 2 components
            self.training_pipeline = create_strategy_training_pipeline(
                validation_method="walk_forward",
                train_ratio=0.7,
                optimization_method="bayesian",
                max_iterations=50,
            )

            self.validation_engine = create_strategy_validator(
                min_sharpe_ratio=self.min_sharpe_ratio,
                max_drawdown=self.max_drawdown,
                min_confidence_level=0.90,
            )

            self.persistence_manager = create_strategy_persistence_manager(
                storage_backend="filesystem",
                base_path=str(self.output_dir / "strategies"),
                enable_versioning=True,
            )

            console.print("   ✅ Pipeline components initialized")

    def validate_strategy_file(
        self,
        strategy_file_path: str,
        symbols: list[str] = None,
        days: int = 730,
        save_results: bool = True,
    ) -> ValidationPipelineResult:
        """
        Execute full validation pipeline for a strategy file

        Args:
            strategy_file_path: Path to strategy Python file
            symbols: List of symbols to test with (default: tech stocks)
            days: Days of historical data (default: 2 years)
            save_results: Whether to save detailed results

        Returns:
            ValidationPipelineResult with comprehensive test results
        """

        if symbols is None:
            symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

        # Initialize result tracking
        result = ValidationPipelineResult()
        result.strategy_name = Path(strategy_file_path).stem
        start_time = datetime.now()

        try:
            console.print("🚀 [bold blue]Starting Validation Pipeline[/bold blue]")
            console.print(f"   Strategy: {result.strategy_name}")
            console.print(f"   Symbols: {', '.join(symbols)}")
            console.print(f"   Data Period: {days} days")

            self._initialize_components()

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
                console=console,
            ) as progress:

                # Step 1: Data Preparation
                data_task = progress.add_task("📊 Preparing data...", total=1)
                datasets = self._prepare_validation_data(symbols, days, result)
                progress.update(data_task, completed=1)

                if not result.data_preparation_success:
                    return result

                # Step 2: Strategy Loading
                strategy_task = progress.add_task("🔌 Loading strategy...", total=1)
                strategy = self._load_strategy_from_file(strategy_file_path, result)
                progress.update(strategy_task, completed=1)

                if not result.strategy_loading_success:
                    return result

                # Step 3: Strategy Training
                training_task = progress.add_task("🏋️ Training strategy...", total=1)
                self._execute_strategy_training(strategy, datasets, result)
                progress.update(training_task, completed=1)

                # Step 4: Strategy Validation
                validation_task = progress.add_task("🔍 Validating strategy...", total=1)
                self._execute_strategy_validation(strategy, datasets, result)
                progress.update(validation_task, completed=1)

                # Step 5: Strategy Persistence
                persistence_task = progress.add_task("💾 Persisting results...", total=1)
                self._execute_strategy_persistence(strategy, result)
                progress.update(persistence_task, completed=1)

            # Finalize results
            result.pipeline_success = all(
                [
                    result.data_preparation_success,
                    result.strategy_loading_success,
                    result.training_success,
                    result.validation_success,
                    result.persistence_success,
                ]
            )

            result.execution_time_seconds = (datetime.now() - start_time).total_seconds()

            # Save results if requested
            if save_results:
                self._save_pipeline_results(result)

            # Add to history
            self.current_result = result
            self.validation_history.append(result)

            # Display summary
            self._display_pipeline_summary(result)

            return result

        except Exception as e:
            result.errors.append(str(e))
            result.execution_time_seconds = (datetime.now() - start_time).total_seconds()
            logger.exception(f"Validation pipeline failed for {result.strategy_name}")
            console.print(f"❌ [bold red]Pipeline failed:[/bold red] {str(e)}")
            return result

    def _prepare_validation_data(
        self, symbols: list[str], days: int, result: ValidationPipelineResult
    ) -> dict[str, pd.DataFrame]:
        """Prepare and validate data for strategy testing"""

        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            # Download historical data
            console.print(f"   📊 Downloading data for {len(symbols)} symbols...")
            datasets, metadata = self.data_manager.get_training_dataset(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                frequency=DataFrequency.DAILY,
                force_refresh=False,
            )

            # Quality assessment and cleaning
            console.print("   🧹 Assessing and cleaning data quality...")
            cleaned_datasets = {}

            for symbol, raw_data in datasets.items():
                cleaned_data, quality_report = self.quality_framework.clean_and_validate(
                    raw_data, symbol
                )

                if quality_report.is_usable:
                    cleaned_datasets[symbol] = cleaned_data
                    result.data_quality_scores[symbol] = quality_report.quality_score
                    console.print(
                        f"      • {symbol}: {len(cleaned_data)} records (Quality: {quality_report.quality_score:.1f}/100)"
                    )
                else:
                    result.warnings.append(f"Poor quality data excluded for {symbol}")
                    console.print(f"      • {symbol}: [yellow]Poor quality data excluded[/yellow]")

            if not cleaned_datasets:
                result.errors.append("No usable datasets after quality filtering")
                return {}

            avg_quality = np.mean(list(result.data_quality_scores.values()))
            console.print(
                f"   ✅ Data preparation complete. Average quality: {avg_quality:.1f}/100"
            )

            result.data_preparation_success = True
            return cleaned_datasets

        except Exception as e:
            result.errors.append(f"Data preparation failed: {str(e)}")
            result.data_preparation_success = False
            return {}

    def _load_strategy_from_file(
        self, strategy_file_path: str, result: ValidationPipelineResult
    ) -> Strategy | None:
        """Load strategy class from Python file"""

        try:
            strategy_path = Path(strategy_file_path)
            if not strategy_path.exists():
                raise StrategyLoadError(f"Strategy file not found: {strategy_file_path}")

            # Import strategy module
            spec = importlib.util.spec_from_file_location("strategy_module", strategy_path)
            if spec is None or spec.loader is None:
                raise StrategyLoadError(f"Cannot load strategy module from {strategy_path}")

            strategy_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(strategy_module)

            # Find strategy class or create_strategy function
            strategy_instance = None

            if hasattr(strategy_module, "create_strategy"):
                # Use factory function
                strategy_instance = strategy_module.create_strategy()
                console.print("   ✅ Loaded strategy using create_strategy() factory")
            else:
                # Find Strategy class
                strategy_classes = [
                    getattr(strategy_module, attr)
                    for attr in dir(strategy_module)
                    if (
                        isinstance(getattr(strategy_module, attr), type)
                        and issubclass(getattr(strategy_module, attr), Strategy)
                        and getattr(strategy_module, attr) != Strategy
                    )
                ]

                if not strategy_classes:
                    raise StrategyLoadError("No Strategy class found in module")

                if len(strategy_classes) > 1:
                    result.warnings.append(
                        f"Multiple Strategy classes found, using {strategy_classes[0].__name__}"
                    )

                strategy_instance = strategy_classes[0]()
                console.print(f"   ✅ Loaded strategy class: {strategy_classes[0].__name__}")

            if strategy_instance is None:
                raise StrategyLoadError("Failed to create strategy instance")

            # Validate strategy interface
            if not hasattr(strategy_instance, "generate_signals"):
                raise StrategyLoadError("Strategy missing generate_signals method")

            result.strategy_loading_success = True
            return strategy_instance

        except Exception as e:
            result.errors.append(f"Strategy loading failed: {str(e)}")
            result.strategy_loading_success = False
            return None

    def _execute_strategy_training(
        self,
        strategy: Strategy,
        datasets: dict[str, pd.DataFrame],
        result: ValidationPipelineResult,
    ) -> None:
        """Execute strategy training pipeline"""

        try:
            console.print(f"   🏋️ Training strategy with {len(datasets)} datasets...")

            # Use first dataset for training (could be enhanced to use multiple)
            primary_symbol = list(datasets.keys())[0]

            # Get parameter space from strategy if available
            parameter_space = {}
            if hasattr(strategy, "get_parameter_space"):
                parameter_space = strategy.get_parameter_space()

            if parameter_space:
                # Run parameter optimization
                training_result = self.training_pipeline.train_strategy(
                    strategy=strategy,
                    symbols=[primary_symbol],
                    parameter_space=parameter_space,
                    strategy_id=f"{result.strategy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                )

                result.training_results = training_result
                console.print(
                    f"   ✅ Training complete. Best score: {training_result.best_score:.3f}"
                )
            else:
                # No parameters to optimize, mark as successful
                console.print("   ✅ Strategy has no parameters to optimize")

            result.training_success = True

        except Exception as e:
            result.errors.append(f"Strategy training failed: {str(e)}")
            result.training_success = False

    def _execute_strategy_validation(
        self,
        strategy: Strategy,
        datasets: dict[str, pd.DataFrame],
        result: ValidationPipelineResult,
    ) -> None:
        """Execute comprehensive strategy validation"""

        try:
            console.print("   🔍 Validating strategy performance...")

            # Generate combined returns for validation
            all_returns = []

            for symbol, data in datasets.items():
                try:
                    # Generate signals using strategy
                    signals = strategy.generate_signals(data)

                    # Calculate returns (simplified approach)
                    price_changes = data["Close"].pct_change().fillna(0)

                    # Apply signals to returns (assuming signals are 1, 0, -1)
                    if "signal" in signals.columns:
                        strategy_returns = signals["signal"].shift(1).fillna(0) * price_changes
                        all_returns.append(strategy_returns)

                except Exception as e:
                    result.warnings.append(f"Signal generation failed for {symbol}: {str(e)}")
                    continue

            if not all_returns:
                raise ValueError("No valid returns generated from any dataset")

            # Combine returns (equal weight across symbols)
            combined_returns = pd.concat(all_returns, axis=1).mean(axis=1).dropna()

            if len(combined_returns) < 50:
                raise ValueError("Insufficient return data for validation")

            # Run validation
            validation_result = self.validation_engine.validate_strategy(
                returns=combined_returns, strategy_id=f"{result.strategy_name}_validation"
            )

            result.validation_results = validation_result
            result.overall_score = validation_result.overall_score
            result.validation_grade = validation_result.validation_grade

            # Extract key metrics
            if hasattr(validation_result, "performance_metrics"):
                metrics = validation_result.performance_metrics
                result.sharpe_ratio = getattr(metrics, "sharpe_ratio", 0.0)
                result.max_drawdown = getattr(metrics, "max_drawdown", 0.0)

            console.print(
                f"   ✅ Validation complete. Score: {result.overall_score:.1f}/100, Grade: {result.validation_grade}"
            )
            result.validation_success = True

        except Exception as e:
            result.errors.append(f"Strategy validation failed: {str(e)}")
            result.validation_success = False

    def _execute_strategy_persistence(
        self, strategy: Strategy, result: ValidationPipelineResult
    ) -> None:
        """Persist strategy and results"""

        try:
            console.print("   💾 Persisting strategy and results...")

            # Create strategy metadata
            metadata = {
                "strategy_name": result.strategy_name,
                "validation_score": result.overall_score,
                "validation_grade": result.validation_grade,
                "sharpe_ratio": result.sharpe_ratio,
                "max_drawdown": result.max_drawdown,
                "timestamp": result.timestamp.isoformat(),
                "data_quality_scores": result.data_quality_scores,
            }

            # Register strategy with persistence manager
            strategy_record = self.persistence_manager.register_strategy(
                strategy=strategy, metadata=metadata
            )

            result.persistence_results = strategy_record
            console.print(f"   ✅ Strategy persisted with ID: {strategy_record.strategy_id}")
            result.persistence_success = True

        except Exception as e:
            result.errors.append(f"Strategy persistence failed: {str(e)}")
            result.persistence_success = False

    def _save_pipeline_results(self, result: ValidationPipelineResult) -> None:
        """Save detailed pipeline results to files"""

        try:
            # Create results directory for this strategy
            strategy_results_dir = self.output_dir / "results" / result.strategy_name
            strategy_results_dir.mkdir(parents=True, exist_ok=True)

            # Save main result as JSON
            result_dict = {
                "strategy_name": result.strategy_name,
                "pipeline_success": result.pipeline_success,
                "timestamp": result.timestamp.isoformat(),
                "execution_time_seconds": result.execution_time_seconds,
                "overall_score": result.overall_score,
                "validation_grade": result.validation_grade,
                "sharpe_ratio": result.sharpe_ratio,
                "max_drawdown": result.max_drawdown,
                "data_quality_scores": result.data_quality_scores,
                "component_success": {
                    "data_preparation": result.data_preparation_success,
                    "strategy_loading": result.strategy_loading_success,
                    "training": result.training_success,
                    "validation": result.validation_success,
                    "persistence": result.persistence_success,
                },
                "errors": result.errors,
                "warnings": result.warnings,
            }

            result_file = strategy_results_dir / f"{result.strategy_name}_pipeline_result.json"
            with open(result_file, "w") as f:
                json.dump(result_dict, f, indent=2, default=str)

            console.print(f"   📄 Pipeline results saved: {result_file}")

        except Exception as e:
            logger.warning(f"Failed to save pipeline results: {str(e)}")

    def _display_pipeline_summary(self, result: ValidationPipelineResult) -> None:
        """Display comprehensive pipeline summary"""

        console.print("\n🎯 [bold]Validation Pipeline Summary[/bold]")

        # Create status table
        status_table = Table(title=f"📊 {result.strategy_name} Pipeline Results")
        status_table.add_column("Component", style="cyan", width=25)
        status_table.add_column("Status", style="white", width=12)
        status_table.add_column("Details", style="dim", width=30)

        components = [
            (
                "Data Preparation",
                result.data_preparation_success,
                (
                    f"Avg Quality: {np.mean(list(result.data_quality_scores.values())):.1f}/100"
                    if result.data_quality_scores
                    else "N/A"
                ),
            ),
            ("Strategy Loading", result.strategy_loading_success, "Strategy instantiated"),
            (
                "Strategy Training",
                result.training_success,
                "Parameters optimized" if result.training_results else "No parameters",
            ),
            (
                "Strategy Validation",
                result.validation_success,
                f"Score: {result.overall_score:.1f}/100",
            ),
            ("Result Persistence", result.persistence_success, "Strategy registered"),
        ]

        for component, success, details in components:
            status = "✅ Success" if success else "❌ Failed"
            status_table.add_row(component, status, details)

        console.print(status_table)

        # Overall result
        if result.pipeline_success:
            console.print("\n✅ [bold green]Pipeline Completed Successfully![/bold green]")
            console.print(f"   • Overall Score: {result.overall_score:.1f}/100")
            console.print(f"   • Grade: {result.validation_grade}")
            console.print(f"   • Execution Time: {result.execution_time_seconds:.1f}s")

            if result.validation_results and hasattr(result.validation_results, "is_validated"):
                if result.validation_results.is_validated:
                    console.print("   🚀 Strategy is ready for deployment!")
                else:
                    console.print("   ⚠️  Strategy needs improvement before deployment")
        else:
            console.print("\n❌ [bold red]Pipeline Failed[/bold red]")
            if result.errors:
                console.print("   Errors:")
                for error in result.errors[:3]:  # Show first 3 errors
                    console.print(f"   • {error}")

        if result.warnings:
            console.print("\n⚠️  [bold yellow]Warnings[/bold yellow]")
            for warning in result.warnings[:3]:  # Show first 3 warnings
                console.print(f"   • {warning}")


def create_validation_pipeline(
    output_dir: str = "data/validation_pipeline",
    data_quality_threshold: float = 75.0,
    min_sharpe_ratio: float = 0.3,
    max_drawdown: float = 0.20,
) -> ValidationPipeline:
    """Factory function to create validation pipeline"""
    return ValidationPipeline(
        output_dir=output_dir,
        data_quality_threshold=data_quality_threshold,
        min_sharpe_ratio=min_sharpe_ratio,
        max_drawdown=max_drawdown,
    )


if __name__ == "__main__":
    # Example usage
    pipeline = create_validation_pipeline()

    # Example strategy file path (would be provided by user)
    # strategy_file = "data/strategy_development/strategies/my_strategy/my_strategy.py"
    # result = pipeline.validate_strategy_file(strategy_file)

    print("Validation Pipeline Integration created successfully!")
