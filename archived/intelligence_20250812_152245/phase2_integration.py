"""
Phase 2 Integration and Testing Framework for GPT-Trader.

This module provides comprehensive integration testing and validation for all Phase 2
Advanced Learning Systems components:

- Online Learning Framework integration
- Reinforcement Learning System validation
- Transfer Learning Framework testing
- Adaptive Model Selection verification
- Meta-Learning System validation
- Continual Learning Framework testing
- End-to-end pipeline integration
- Performance benchmarking and analysis

Ensures all Phase 2 components work together seamlessly and deliver the expected
advanced learning capabilities for trading system enhancement.
"""

from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

# Import all Phase 2 components
try:
    from .online_learning import (
        OnlineLearningConfig,
        OnlineStrategyLearner,
        create_default_online_learner,
    )

    HAS_ONLINE_LEARNING = True
except ImportError as e:
    HAS_ONLINE_LEARNING = False
    warnings.warn(f"Online Learning not available: {e}")

try:
    from .reinforcement_learning import (
        RLConfig,
        create_rl_trading_system,
        train_rl_trading_strategy,
    )

    HAS_REINFORCEMENT_LEARNING = True
except ImportError as e:
    HAS_REINFORCEMENT_LEARNING = False
    warnings.warn(f"Reinforcement Learning not available: {e}")

try:
    from .transfer_learning import (
        DomainData,
        TransferLearningConfig,
        create_transfer_learning_system,
        demonstrate_cross_asset_transfer,
    )

    HAS_TRANSFER_LEARNING = True
except ImportError as e:
    HAS_TRANSFER_LEARNING = False
    warnings.warn(f"Transfer Learning not available: {e}")

try:
    from .adaptive_model_selection import (
        AdaptiveModelConfig,
        create_adaptive_model_selection,
        demonstrate_adaptive_selection,
    )

    HAS_ADAPTIVE_SELECTION = True
except ImportError as e:
    HAS_ADAPTIVE_SELECTION = False
    warnings.warn(f"Adaptive Model Selection not available: {e}")

try:
    from .meta_learning import (
        MetaLearningConfig,
        create_meta_learning_system,
        demonstrate_meta_learning,
    )

    HAS_META_LEARNING = True
except ImportError as e:
    HAS_META_LEARNING = False
    warnings.warn(f"Meta-Learning not available: {e}")

try:
    from .continual_learning import (
        ContinualLearningConfig,
        create_continual_learning_system,
        demonstrate_continual_learning,
    )

    HAS_CONTINUAL_LEARNING = True
except ImportError as e:
    HAS_CONTINUAL_LEARNING = False
    warnings.warn(f"Continual Learning not available: {e}")

from bot.utils.base import BaseConfig

logger = logging.getLogger(__name__)


@dataclass
class Phase2TestConfig(BaseConfig):
    """Configuration for Phase 2 integration testing."""

    # Test data parameters
    n_samples: int = 1000
    n_features: int = 15
    n_time_periods: int = 5
    noise_level: float = 0.1

    # Component testing
    test_online_learning: bool = True
    test_reinforcement_learning: bool = True
    test_transfer_learning: bool = True
    test_adaptive_selection: bool = True
    test_meta_learning: bool = True
    test_continual_learning: bool = True
    test_integration_pipeline: bool = True

    # Performance thresholds
    min_online_learning_r2: float = 0.3
    min_transfer_improvement: float = 0.05
    min_adaptation_confidence: float = 0.4
    min_meta_learning_r2: float = 0.2
    min_continual_performance: float = 0.25

    # Test execution
    timeout_seconds: int = 300  # 5 minutes per test
    verbose_output: bool = True

    # Random state
    random_state: int = 42


@dataclass
class ComponentTestResult:
    """Result from individual component test."""

    component_name: str
    test_passed: bool
    execution_time: float
    performance_metrics: dict[str, float]
    error_message: str | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class IntegrationTestResult:
    """Result from integration testing."""

    total_tests: int
    passed_tests: int
    failed_tests: int
    success_rate: float
    total_execution_time: float
    component_results: list[ComponentTestResult]
    integration_pipeline_result: ComponentTestResult | None = None


class DataGenerator:
    """Generate synthetic data for testing Phase 2 components."""

    @staticmethod
    def create_financial_time_series(
        n_samples: int = 1000, n_features: int = 15, random_state: int = 42
    ) -> pd.DataFrame:
        """Create realistic financial time series data."""
        np.random.seed(random_state)

        # Generate dates
        dates = pd.date_range(start="2020-01-01", periods=n_samples, freq="D")

        # Generate price data with realistic patterns
        base_price = 100.0
        prices = [base_price]

        for _i in range(1, n_samples):
            # Random walk with mean reversion and volatility clustering
            prev_return = (prices[-1] - prices[-2]) / prices[-2] if len(prices) > 1 else 0
            volatility = 0.02 * (1 + 0.5 * abs(prev_return))

            # Mean reversion
            mean_reversion = -0.001 * (prices[-1] - base_price) / base_price

            # Random shock
            shock = np.random.normal(0, volatility)

            change = mean_reversion + shock
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 1.0))  # Prevent negative prices

        # Create OHLCV data
        df = pd.DataFrame(index=dates)
        df["Close"] = prices

        # Generate OHLC from close prices
        df["Open"] = df["Close"].shift(1) * (1 + np.random.normal(0, 0.001, n_samples))
        df["High"] = df[["Open", "Close"]].max(axis=1) * (
            1 + np.abs(np.random.normal(0, 0.005, n_samples))
        )
        df["Low"] = df[["Open", "Close"]].min(axis=1) * (
            1 - np.abs(np.random.normal(0, 0.005, n_samples))
        )
        df["Volume"] = np.random.lognormal(10, 1, n_samples)

        # Add technical features
        df["Returns"] = df["Close"].pct_change()
        df["MA_5"] = df["Close"].rolling(5).mean()
        df["MA_20"] = df["Close"].rolling(20).mean()
        df["Volatility"] = df["Returns"].rolling(20).std()
        df["RSI"] = DataGenerator._calculate_rsi(df["Close"])

        # Add regime indicators
        df["Volatility_Regime"] = (df["Volatility"] > df["Volatility"].rolling(50).mean()).astype(
            int
        )
        df["Trend_Regime"] = (df["MA_5"] > df["MA_20"]).astype(int)

        # Fill missing values
        df = df.fillna(method="ffill").fillna(0)

        return df

    @staticmethod
    def _calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def create_multi_asset_data(
        assets: list[str], n_samples: int = 1000, random_state: int = 42
    ) -> dict[str, pd.DataFrame]:
        """Create multi-asset dataset for cross-asset testing."""

        data_dict = {}

        for i, asset in enumerate(assets):
            # Use different random seed for each asset
            asset_data = DataGenerator.create_financial_time_series(
                n_samples, random_state=random_state + i
            )
            data_dict[asset] = asset_data

        return data_dict

    @staticmethod
    def create_regime_change_data(
        n_samples: int = 1000, n_regimes: int = 3, random_state: int = 42
    ) -> pd.DataFrame:
        """Create data with multiple regime changes."""
        np.random.seed(random_state)

        # Create regime periods
        regime_length = n_samples // n_regimes

        all_data = []

        for regime in range(n_regimes):
            start_idx = regime * regime_length
            end_idx = start_idx + regime_length if regime < n_regimes - 1 else n_samples
            regime_samples = end_idx - start_idx

            # Different characteristics for each regime
            if regime == 0:  # Low volatility bull market
                base_trend = 0.0005
            elif regime == 1:  # High volatility bear market
                base_trend = -0.001
            else:  # Sideways market
                base_trend = 0.0

            # Generate regime-specific data
            regime_data = DataGenerator.create_financial_time_series(
                regime_samples, random_state=random_state + regime
            )

            # Adjust for regime characteristics
            regime_data["Close"] *= 1 + base_trend * np.arange(regime_samples)
            regime_data["Returns"] = regime_data["Close"].pct_change()
            regime_data["Regime_ID"] = regime

            all_data.append(regime_data)

        # Concatenate all regimes
        combined_data = pd.concat(all_data, ignore_index=False)

        # Reset index to be continuous
        dates = pd.date_range(start="2020-01-01", periods=len(combined_data), freq="D")
        combined_data.index = dates

        return combined_data


class Phase2ComponentTester:
    """Test individual Phase 2 components."""

    def __init__(self, config: Phase2TestConfig) -> None:
        self.config = config

    def test_online_learning(self) -> ComponentTestResult:
        """Test Online Learning Framework."""
        start_time = time.time()
        component_name = "Online Learning"

        try:
            if not HAS_ONLINE_LEARNING:
                return ComponentTestResult(
                    component_name=component_name,
                    test_passed=False,
                    execution_time=0.0,
                    performance_metrics={},
                    error_message="Online Learning module not available",
                )

            logger.info("Testing Online Learning Framework...")

            # Generate test data
            data = DataGenerator.create_financial_time_series(
                self.config.n_samples, random_state=self.config.random_state
            )

            # Create online learner
            OnlineLearningConfig(learning_rate=0.01, batch_update_size=20, performance_window=30)
            online_learner = create_default_online_learner()

            # Initialize with first half of data
            split_idx = len(data) // 2
            init_data = data.iloc[:split_idx]
            target = init_data["Close"].pct_change().fillna(0)
            features = init_data[["Returns", "MA_5", "MA_20", "Volatility", "RSI"]].fillna(0)

            online_learner.initialize(features, target)

            # Update with streaming data
            stream_data = data.iloc[split_idx:]
            stream_target = stream_data["Close"].pct_change().fillna(0)
            stream_features = stream_data[["Returns", "MA_5", "MA_20", "Volatility", "RSI"]].fillna(
                0
            )

            update_results = []
            for i in range(0, len(stream_data), 10):  # Process in batches of 10
                batch_features = stream_features.iloc[i : i + 10]
                batch_target = stream_target.iloc[i : i + 10]

                if len(batch_features) > 0:
                    result = online_learner.update(batch_features, batch_target)
                    update_results.append(result)

            # Test predictions
            test_features = stream_features.tail(50)
            predictions = online_learner.predict(test_features)
            test_target = stream_target.tail(50)

            # Calculate performance
            r2 = r2_score(test_target, predictions)
            mse = mean_squared_error(test_target, predictions)

            execution_time = time.time() - start_time
            test_passed = r2 >= self.config.min_online_learning_r2

            return ComponentTestResult(
                component_name=component_name,
                test_passed=test_passed,
                execution_time=execution_time,
                performance_metrics={
                    "r2_score": r2,
                    "mse": mse,
                    "updates_processed": len(update_results),
                    "samples_processed": online_learner.samples_processed,
                },
            )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Online Learning test failed: {e}")
            return ComponentTestResult(
                component_name=component_name,
                test_passed=False,
                execution_time=execution_time,
                performance_metrics={},
                error_message=str(e),
            )

    def test_reinforcement_learning(self) -> ComponentTestResult:
        """Test Reinforcement Learning System."""
        start_time = time.time()
        component_name = "Reinforcement Learning"

        try:
            if not HAS_REINFORCEMENT_LEARNING:
                return ComponentTestResult(
                    component_name=component_name,
                    test_passed=False,
                    execution_time=0.0,
                    performance_metrics={},
                    error_message="Reinforcement Learning module not available",
                )

            logger.info("Testing Reinforcement Learning System...")

            # Generate test data
            data = DataGenerator.create_financial_time_series(
                500, random_state=self.config.random_state  # Smaller dataset for RL
            )

            # Train RL strategy (simplified)
            result = train_rl_trading_strategy(data, episodes=50)  # Fewer episodes for testing

            execution_time = time.time() - start_time
            test_passed = result.get("training_completed", False)

            performance_metrics = {
                "training_completed": result.get("training_completed", False),
                "episodes_trained": result.get("episodes", 0),
                "final_performance": result.get("final_performance", 0.0),
            }

            if "final_metrics" in result:
                performance_metrics.update(result["final_metrics"])

            return ComponentTestResult(
                component_name=component_name,
                test_passed=test_passed,
                execution_time=execution_time,
                performance_metrics=performance_metrics,
            )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Reinforcement Learning test failed: {e}")
            return ComponentTestResult(
                component_name=component_name,
                test_passed=False,
                execution_time=execution_time,
                performance_metrics={},
                error_message=str(e),
            )

    def test_transfer_learning(self) -> ComponentTestResult:
        """Test Transfer Learning Framework."""
        start_time = time.time()
        component_name = "Transfer Learning"

        try:
            if not HAS_TRANSFER_LEARNING:
                return ComponentTestResult(
                    component_name=component_name,
                    test_passed=False,
                    execution_time=0.0,
                    performance_metrics={},
                    error_message="Transfer Learning module not available",
                )

            logger.info("Testing Transfer Learning Framework...")

            # Generate multi-asset data for transfer learning
            assets_data = DataGenerator.create_multi_asset_data(
                ["STOCK_A", "STOCK_B"], n_samples=500, random_state=self.config.random_state
            )

            # Test cross-asset transfer
            result = demonstrate_cross_asset_transfer(
                assets_data["STOCK_A"], assets_data["STOCK_B"]
            )

            execution_time = time.time() - start_time

            # Evaluate results
            if result.get("transfer_completed", False):
                improvement = result.get("performance_improvement", 0.0)
                similarity = result.get("domain_similarity", 0.0)

                test_passed = improvement >= self.config.min_transfer_improvement

                performance_metrics = {
                    "transfer_completed": True,
                    "performance_improvement": improvement,
                    "domain_similarity": similarity,
                    "transfer_method": result.get("transfer_result", {}).get(
                        "method_used", "unknown"
                    ),
                }
            else:
                test_passed = False
                performance_metrics = {"error": result.get("error", "Unknown error")}

            return ComponentTestResult(
                component_name=component_name,
                test_passed=test_passed,
                execution_time=execution_time,
                performance_metrics=performance_metrics,
            )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Transfer Learning test failed: {e}")
            return ComponentTestResult(
                component_name=component_name,
                test_passed=False,
                execution_time=execution_time,
                performance_metrics={},
                error_message=str(e),
            )

    def test_adaptive_model_selection(self) -> ComponentTestResult:
        """Test Adaptive Model Selection."""
        start_time = time.time()
        component_name = "Adaptive Model Selection"

        try:
            if not HAS_ADAPTIVE_SELECTION:
                return ComponentTestResult(
                    component_name=component_name,
                    test_passed=False,
                    execution_time=0.0,
                    performance_metrics={},
                    error_message="Adaptive Model Selection module not available",
                )

            logger.info("Testing Adaptive Model Selection...")

            # Generate test data
            data = DataGenerator.create_financial_time_series(
                self.config.n_samples, random_state=self.config.random_state
            )

            # Create target
            target = data["Close"].pct_change().shift(-1).fillna(0)
            features = data[["Returns", "MA_5", "MA_20", "Volatility", "RSI"]].fillna(0)

            # Test adaptive selection
            result = demonstrate_adaptive_selection(features, target)

            execution_time = time.time() - start_time

            if result.get("demo_completed", False):
                confidence = result.get("selection_confidence", 0.0)
                selected_r2 = result.get("selected_model_r2", 0.0)
                ensemble_r2 = result.get("ensemble_r2", 0.0)

                test_passed = confidence >= self.config.min_adaptation_confidence

                performance_metrics = {
                    "demo_completed": True,
                    "selected_model": result.get("selected_model", "unknown"),
                    "selection_confidence": confidence,
                    "selected_model_r2": selected_r2,
                    "ensemble_r2": ensemble_r2,
                    "ensemble_size": result.get("ensemble_info", {}).get("ensemble_size", 0),
                }
            else:
                test_passed = False
                performance_metrics = {"error": result.get("error", "Unknown error")}

            return ComponentTestResult(
                component_name=component_name,
                test_passed=test_passed,
                execution_time=execution_time,
                performance_metrics=performance_metrics,
            )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Adaptive Model Selection test failed: {e}")
            return ComponentTestResult(
                component_name=component_name,
                test_passed=False,
                execution_time=execution_time,
                performance_metrics={},
                error_message=str(e),
            )

    def test_meta_learning(self) -> ComponentTestResult:
        """Test Meta-Learning System."""
        start_time = time.time()
        component_name = "Meta-Learning"

        try:
            if not HAS_META_LEARNING:
                return ComponentTestResult(
                    component_name=component_name,
                    test_passed=False,
                    execution_time=0.0,
                    performance_metrics={},
                    error_message="Meta-Learning module not available",
                )

            logger.info("Testing Meta-Learning System...")

            # Generate historical and new market data
            historical_data = DataGenerator.create_financial_time_series(
                800, random_state=self.config.random_state
            )
            new_market_data = DataGenerator.create_financial_time_series(
                200, random_state=self.config.random_state + 1
            )

            # Test meta-learning
            result = demonstrate_meta_learning(historical_data, new_market_data)

            execution_time = time.time() - start_time

            if result.get("demo_completed", False):
                success = result.get("meta_learning_success", False)
                adaptation_r2 = (
                    result.get("adaptation_result", {})
                    .get("performance_metrics", {})
                    .get("r2_score", 0.0)
                )

                test_passed = success and adaptation_r2 >= self.config.min_meta_learning_r2

                training_result = result.get("training_result", {})
                adaptation_result = result.get("adaptation_result", {})

                performance_metrics = {
                    "demo_completed": True,
                    "meta_learning_success": success,
                    "training_meta_loss": training_result.get("meta_loss", 0.0),
                    "adaptation_r2": adaptation_r2,
                    "adaptation_time": adaptation_result.get("adaptation_time", 0.0),
                    "tasks_trained": training_result.get("n_tasks_trained", 0),
                }
            else:
                test_passed = False
                performance_metrics = {"error": result.get("error", "Unknown error")}

            return ComponentTestResult(
                component_name=component_name,
                test_passed=test_passed,
                execution_time=execution_time,
                performance_metrics=performance_metrics,
            )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Meta-Learning test failed: {e}")
            return ComponentTestResult(
                component_name=component_name,
                test_passed=False,
                execution_time=execution_time,
                performance_metrics={},
                error_message=str(e),
            )

    def test_continual_learning(self) -> ComponentTestResult:
        """Test Continual Learning Framework."""
        start_time = time.time()
        component_name = "Continual Learning"

        try:
            if not HAS_CONTINUAL_LEARNING:
                return ComponentTestResult(
                    component_name=component_name,
                    test_passed=False,
                    execution_time=0.0,
                    performance_metrics={},
                    error_message="Continual Learning module not available",
                )

            logger.info("Testing Continual Learning Framework...")

            # Generate sequence of market data with regime changes
            regime_data = DataGenerator.create_regime_change_data(
                n_samples=900, n_regimes=3, random_state=self.config.random_state
            )

            # Split into sequence for continual learning
            seq_length = len(regime_data) // 3
            data_sequence = [
                regime_data.iloc[i * seq_length : (i + 1) * seq_length] for i in range(3)
            ]

            # Test continual learning
            result = demonstrate_continual_learning(data_sequence)

            execution_time = time.time() - start_time

            if result.get("demo_completed", False):
                evaluation = result.get("final_evaluation", {})
                avg_performance = evaluation.get("average_performance", 0.0)
                forgetting = evaluation.get("average_forgetting", 1.0)

                test_passed = avg_performance >= self.config.min_continual_performance

                performance_metrics = {
                    "demo_completed": True,
                    "tasks_learned": result.get("tasks_learned", 0),
                    "data_periods": result.get("total_data_periods", 0),
                    "average_performance": avg_performance,
                    "average_forgetting": forgetting,
                    "backward_transfer": evaluation.get("average_backward_transfer", 0.0),
                    "continual_strategy": evaluation.get("continual_learning_strategy", "unknown"),
                }
            else:
                test_passed = False
                performance_metrics = {"error": result.get("error", "Unknown error")}

            return ComponentTestResult(
                component_name=component_name,
                test_passed=test_passed,
                execution_time=execution_time,
                performance_metrics=performance_metrics,
            )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Continual Learning test failed: {e}")
            return ComponentTestResult(
                component_name=component_name,
                test_passed=False,
                execution_time=execution_time,
                performance_metrics={},
                error_message=str(e),
            )


class Phase2IntegrationPipeline:
    """Integration pipeline combining multiple Phase 2 components."""

    def __init__(self, config: Phase2TestConfig) -> None:
        self.config = config

    def test_integration_pipeline(self) -> ComponentTestResult:
        """Test end-to-end integration of Phase 2 components."""
        start_time = time.time()
        component_name = "Integration Pipeline"

        try:
            logger.info("Testing Phase 2 Integration Pipeline...")

            # Generate comprehensive test dataset
            base_data = DataGenerator.create_financial_time_series(
                self.config.n_samples, random_state=self.config.random_state
            )

            pipeline_results = {}

            # Step 1: Online Learning for real-time adaptation
            if HAS_ONLINE_LEARNING:
                try:
                    online_learner = create_default_online_learner()

                    # Initialize
                    split_idx = len(base_data) // 3
                    init_features = (
                        base_data[["Returns", "MA_5", "MA_20", "Volatility"]]
                        .iloc[:split_idx]
                        .fillna(0)
                    )
                    init_target = base_data["Close"].pct_change().iloc[:split_idx].fillna(0)

                    online_learner.initialize(init_features, init_target)

                    # Get framework state
                    state = online_learner.get_framework_state()
                    pipeline_results["online_learning"] = {
                        "initialized": state["is_fitted"],
                        "samples_processed": state["samples_processed"],
                    }
                except Exception as e:
                    pipeline_results["online_learning"] = {"error": str(e)}

            # Step 2: Adaptive Model Selection for regime-aware modeling
            if HAS_ADAPTIVE_SELECTION:
                try:
                    features = base_data[["Returns", "MA_5", "MA_20", "Volatility", "RSI"]].fillna(
                        0
                    )
                    target = base_data["Close"].pct_change().shift(-1).fillna(0)

                    framework = create_adaptive_model_selection()
                    selection_result = framework.select_model(features, target)
                    ensemble_info = framework.create_ensemble(features, target, top_k=3)

                    pipeline_results["adaptive_selection"] = {
                        "selected_model": selection_result.selected_model,
                        "confidence": selection_result.selection_confidence,
                        "ensemble_size": ensemble_info["ensemble_size"],
                    }
                except Exception as e:
                    pipeline_results["adaptive_selection"] = {"error": str(e)}

            # Step 3: Transfer Learning for cross-market knowledge transfer
            if HAS_TRANSFER_LEARNING:
                try:
                    # Create source and target domains
                    source_data = base_data.iloc[:400]
                    target_data = base_data.iloc[400:]

                    framework = create_transfer_learning_system()

                    # Prepare domain data
                    source_features = source_data[
                        ["Returns", "MA_5", "MA_20", "Volatility"]
                    ].fillna(0)
                    source_target = source_data["Close"].pct_change().fillna(0)

                    target_features = target_data[
                        ["Returns", "MA_5", "MA_20", "Volatility"]
                    ].fillna(0)
                    target_target = target_data["Close"].pct_change().fillna(0)

                    source_domain = DomainData("source", source_features, source_target)
                    target_domain = DomainData("target", target_features, target_target)

                    framework.train_source_domains([source_domain])
                    transfer_result = framework.transfer_to_target(target_domain)

                    pipeline_results["transfer_learning"] = {
                        "transfer_improvement": transfer_result.transfer_improvement,
                        "domain_similarity": transfer_result.domain_similarity,
                        "method_used": transfer_result.method_used,
                    }
                except Exception as e:
                    pipeline_results["transfer_learning"] = {"error": str(e)}

            # Step 4: Meta-Learning for rapid adaptation
            if HAS_META_LEARNING:
                try:
                    framework = create_meta_learning_system()

                    # Train on historical data
                    historical_portion = base_data.iloc[:600]
                    training_result = framework.train_on_historical_data(
                        historical_portion, target_col="Returns"
                    )

                    # Test adaptation
                    adaptation_data = base_data.iloc[600:650]
                    adaptation_features = adaptation_data[
                        ["Returns", "MA_5", "MA_20", "Volatility"]
                    ].fillna(0)
                    adaptation_target = adaptation_data["Close"].pct_change().fillna(0)

                    adaptation_result = framework.adapt_to_new_scenario(
                        adaptation_features.head(20),  # Support set
                        adaptation_target.head(20),
                        adaptation_features.tail(20),  # Query set
                        adaptation_target.tail(20),
                    )

                    pipeline_results["meta_learning"] = {
                        "training_completed": training_result.n_tasks_trained > 0,
                        "adaptation_r2": adaptation_result["performance_metrics"]["r2_score"],
                        "adaptation_time": adaptation_result["adaptation_time"],
                    }
                except Exception as e:
                    pipeline_results["meta_learning"] = {"error": str(e)}

            # Step 5: Continual Learning for lifelong learning
            if HAS_CONTINUAL_LEARNING:
                try:
                    framework = create_continual_learning_system()

                    # Create time series with regime changes
                    continual_data = base_data.copy()
                    continual_data["target"] = (
                        continual_data["Close"].pct_change().shift(-1).fillna(0)
                    )

                    # Process as streaming data
                    stream_result = framework.process_data_stream(continual_data, "target")

                    pipeline_results["continual_learning"] = {
                        "samples_processed": stream_result["samples_processed"],
                        "tasks_detected": stream_result["tasks_detected"],
                        "tasks_learned": stream_result["tasks_learned"],
                        "change_points": len(stream_result["change_points"]),
                    }
                except Exception as e:
                    pipeline_results["continual_learning"] = {"error": str(e)}

            # Calculate overall pipeline performance
            execution_time = time.time() - start_time

            # Check if critical components worked
            components_working = 0
            total_components = 0

            for _component, result in pipeline_results.items():
                total_components += 1
                if "error" not in result:
                    components_working += 1

            success_rate = components_working / total_components if total_components > 0 else 0.0
            test_passed = success_rate >= 0.6  # At least 60% of components working

            performance_metrics = {
                "pipeline_success_rate": success_rate,
                "components_working": components_working,
                "total_components": total_components,
                "execution_time": execution_time,
                **pipeline_results,
            }

            return ComponentTestResult(
                component_name=component_name,
                test_passed=test_passed,
                execution_time=execution_time,
                performance_metrics=performance_metrics,
            )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Integration Pipeline test failed: {e}")
            return ComponentTestResult(
                component_name=component_name,
                test_passed=False,
                execution_time=execution_time,
                performance_metrics={},
                error_message=str(e),
            )


class Phase2IntegrationTester:
    """
    Comprehensive Phase 2 integration testing framework.

    Tests all Phase 2 components individually and as an integrated system.
    """

    def __init__(self, config: Phase2TestConfig | None = None) -> None:
        self.config = config or Phase2TestConfig()
        self.component_tester = Phase2ComponentTester(self.config)
        self.pipeline_tester = Phase2IntegrationPipeline(self.config)

    def run_all_tests(self, verbose: bool = True) -> IntegrationTestResult:
        """Run comprehensive Phase 2 integration tests."""

        logger.info("Starting Phase 2 Integration Tests...")
        logger.info("=" * 60)

        start_time = time.time()
        component_results = []

        # Test individual components
        if self.config.test_online_learning:
            result = self.component_tester.test_online_learning()
            component_results.append(result)
            if verbose:
                self._log_component_result(result)

        if self.config.test_reinforcement_learning:
            result = self.component_tester.test_reinforcement_learning()
            component_results.append(result)
            if verbose:
                self._log_component_result(result)

        if self.config.test_transfer_learning:
            result = self.component_tester.test_transfer_learning()
            component_results.append(result)
            if verbose:
                self._log_component_result(result)

        if self.config.test_adaptive_selection:
            result = self.component_tester.test_adaptive_model_selection()
            component_results.append(result)
            if verbose:
                self._log_component_result(result)

        if self.config.test_meta_learning:
            result = self.component_tester.test_meta_learning()
            component_results.append(result)
            if verbose:
                self._log_component_result(result)

        if self.config.test_continual_learning:
            result = self.component_tester.test_continual_learning()
            component_results.append(result)
            if verbose:
                self._log_component_result(result)

        # Test integration pipeline
        integration_result = None
        if self.config.test_integration_pipeline:
            integration_result = self.pipeline_tester.test_integration_pipeline()
            if verbose:
                self._log_component_result(integration_result)

        # Calculate overall results
        total_tests = len(component_results) + (1 if integration_result else 0)
        passed_tests = sum(1 for r in component_results if r.test_passed)
        if integration_result and integration_result.test_passed:
            passed_tests += 1

        failed_tests = total_tests - passed_tests
        success_rate = passed_tests / total_tests if total_tests > 0 else 0.0
        total_execution_time = time.time() - start_time

        result = IntegrationTestResult(
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            success_rate=success_rate,
            total_execution_time=total_execution_time,
            component_results=component_results,
            integration_pipeline_result=integration_result,
        )

        if verbose:
            self._print_final_summary(result)

        logger.info("Phase 2 Integration Tests Completed")

        return result

    def _log_component_result(self, result: ComponentTestResult) -> None:
        """Log individual component test result."""

        status = "✓ PASSED" if result.test_passed else "✗ FAILED"
        logger.info(f"{result.component_name:<25} {status:>10} ({result.execution_time:.2f}s)")

        if result.error_message:
            logger.error(f"  Error: {result.error_message}")

        # Log key metrics
        if result.performance_metrics:
            key_metrics = {}
            for key, value in result.performance_metrics.items():
                if key in [
                    "r2_score",
                    "performance_improvement",
                    "confidence",
                    "success_rate",
                    "adaptation_r2",
                ]:
                    if isinstance(value, int | float):
                        key_metrics[key] = value

            if key_metrics:
                metrics_str = ", ".join([f"{k}: {v:.3f}" for k, v in key_metrics.items()])
                logger.info(f"  Key metrics: {metrics_str}")

    def _print_final_summary(self, result: IntegrationTestResult) -> None:
        """Print final test summary."""

        print("\n" + "=" * 60)
        print("PHASE 2 INTEGRATION TEST SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {result.total_tests}")
        print(f"Passed: {result.passed_tests}")
        print(f"Failed: {result.failed_tests}")
        print(f"Success Rate: {result.success_rate:.1%}")
        print(f"Total Execution Time: {result.total_execution_time:.2f}s")
        print()

        # Component breakdown
        print("COMPONENT TEST RESULTS:")
        print("-" * 40)
        for comp_result in result.component_results:
            status = "✓ PASSED" if comp_result.test_passed else "✗ FAILED"
            print(f"{comp_result.component_name:<25} {status:>10}")

        if result.integration_pipeline_result:
            status = "✓ PASSED" if result.integration_pipeline_result.test_passed else "✗ FAILED"
            print(f"{'Integration Pipeline':<25} {status:>10}")

        # Overall assessment
        print("\n" + "-" * 60)
        if result.success_rate >= 0.8:
            print("🎉 EXCELLENT: Phase 2 implementation is highly successful!")
            print("   All major components are working correctly.")
        elif result.success_rate >= 0.6:
            print("✅ GOOD: Phase 2 implementation is mostly successful.")
            print("   Most components are working with minor issues.")
        elif result.success_rate >= 0.4:
            print("⚠️  PARTIAL: Phase 2 implementation has some issues.")
            print("   Several components need attention.")
        else:
            print("❌ NEEDS WORK: Phase 2 implementation requires significant fixes.")
            print("   Many components are not working correctly.")


def run_phase2_integration_tests(
    config: Phase2TestConfig | None = None, verbose: bool = True
) -> IntegrationTestResult:
    """
    Run comprehensive Phase 2 integration tests.

    Args:
        config: Test configuration (uses defaults if None)
        verbose: Whether to print detailed output

    Returns:
        Integration test results
    """

    tester = Phase2IntegrationTester(config)
    return tester.run_all_tests(verbose=verbose)


if __name__ == "__main__":
    # Run tests when module is executed directly

    print("GPT-Trader Phase 2 Integration Testing")
    print("=====================================")

    config = Phase2TestConfig(n_samples=800, verbose_output=True, timeout_seconds=600)  # 10 minutes

    results = run_phase2_integration_tests(config, verbose=True)

    print(f"\nFinal Result: {results.passed_tests}/{results.total_tests} tests passed")
    print(f"Success Rate: {results.success_rate:.1%}")

    if results.success_rate >= 0.8:
        print("\n🚀 Phase 2 Advanced Learning Systems are ready for production!")
    else:
        print("\n⚙️  Some components need attention before production deployment.")
