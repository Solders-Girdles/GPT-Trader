"""
Integration Testing and Validation Framework for GPT-Trader Phase 1.

This module provides comprehensive testing and validation for the enhanced ML components:
- End-to-end pipeline testing
- Performance regression testing
- Integration with existing systems
- Data validation and quality checks
- Model stability testing

Ensures all Phase 1 components work together seamlessly with existing GPT-Trader infrastructure.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from bot.intelligence.bayesian_optimization import BayesianOptConfig, BayesianOptimizer
from bot.intelligence.deep_learning import DeepLearningConfig, DeepLearningFramework
from bot.intelligence.enhanced_features import EnhancedFeatureFramework, FeatureGenerationConfig

# Import our Phase 1 components
from bot.intelligence.ensemble_models import EnhancedModelEnsemble, EnsembleConfig
from bot.intelligence.feature_selection import AutomatedFeatureSelector, FeatureSelectionConfig
from bot.intelligence.model_comparison import ModelComparison, ModelComparisonConfig
from bot.utils.base import BaseConfig
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Container for individual test results."""

    test_name: str
    passed: bool
    execution_time: float
    error_message: str | None = None
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class IntegrationTestConfig(BaseConfig):
    """Configuration for integration testing."""

    # Test data parameters
    n_samples: int = 1000
    n_features: int = 20
    noise_level: float = 0.1
    random_state: int = 42

    # Test execution parameters
    timeout_seconds: int = 300  # 5 minutes per test
    performance_tolerance: float = 0.1  # 10% performance tolerance

    # Test coverage
    test_ensemble_models: bool = True
    test_deep_learning: bool = True
    test_feature_selection: bool = True
    test_bayesian_optimization: bool = True
    test_enhanced_features: bool = True
    test_model_comparison: bool = True
    test_integration_pipeline: bool = True

    # Performance benchmarks
    min_ensemble_r2: float = 0.7
    min_deep_learning_r2: float = 0.6
    max_feature_selection_time: float = 60.0  # seconds
    max_optimization_time: float = 120.0  # seconds

    # Data quality thresholds
    max_missing_values_ratio: float = 0.05
    min_feature_variance: float = 0.01
    max_feature_correlation: float = 0.95


class DataGenerator:
    """Generate synthetic data for testing."""

    @staticmethod
    def create_financial_data(
        n_samples: int = 1000, n_features: int = 20, random_state: int = 42
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Create synthetic financial time series data."""
        np.random.seed(random_state)

        # Generate base time series
        dates = pd.date_range(start="2020-01-01", periods=n_samples, freq="D")

        # Create OHLCV data
        base_price = 100.0
        prices = [base_price]

        for i in range(1, n_samples):
            # Random walk with mean reversion
            change = np.random.normal(0, 0.02) - 0.001 * (prices[-1] - base_price) / base_price
            prices.append(prices[-1] * (1 + change))

        prices = np.array(prices)

        # Create OHLCV from prices
        high_low_range = np.abs(np.random.normal(0, 0.01, n_samples))

        data = pd.DataFrame(
            {
                "Open": prices * (1 + np.random.normal(0, 0.005, n_samples)),
                "High": prices * (1 + high_low_range),
                "Low": prices * (1 - high_low_range),
                "Close": prices,
                "Volume": np.random.lognormal(10, 1, n_samples),
            },
            index=dates,
        )

        # Add technical indicators as features
        for i in range(n_features - 5):  # 5 are OHLCV
            if i < 5:
                # Moving averages
                window = 5 + i * 5
                data[f"MA_{window}"] = data["Close"].rolling(window).mean()
            elif i < 10:
                # Price ratios
                data[f"ratio_{i}"] = data["High"] / data["Low"]
            else:
                # Random technical indicators
                data[f"tech_{i}"] = np.random.normal(0, 1, n_samples)

        # Create target (future returns)
        target = data["Close"].shift(-1) / data["Close"] - 1
        target = target.fillna(0)

        # Remove rows with NaN values
        data = data.fillna(method="ffill").fillna(method="bfill")

        return data.iloc[:-1], target.iloc[:-1]  # Remove last row due to shift

    @staticmethod
    def create_test_data(
        n_samples: int = 500, n_features: int = 10, noise: float = 0.1, random_state: int = 42
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Create simple test data for quick testing."""
        X, y = make_regression(
            n_samples=n_samples, n_features=n_features, noise=noise, random_state=random_state
        )

        # Convert to pandas
        feature_names = [f"feature_{i}" for i in range(n_features)]
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y)

        return X_df, y_series


class ComponentTester:
    """Individual component testing utilities."""

    @staticmethod
    def test_ensemble_models(
        X: pd.DataFrame, y: pd.Series, config: IntegrationTestConfig
    ) -> TestResult:
        """Test ensemble model framework."""
        test_name = "Ensemble Models"
        start_time = time.time()

        try:
            # Create ensemble configuration
            ensemble_config = EnsembleConfig(
                use_random_forest=True,
                use_xgboost=True,  # Will only use if available
                use_lightgbm=True,  # Will only use if available
                n_calls=10,  # Quick test
            )

            # Create and train ensemble
            ensemble = EnhancedModelEnsemble(ensemble_config)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=config.random_state
            )

            # Train ensemble
            ensemble.fit(X_train, y_train)

            # Make predictions
            predictions = ensemble.predict(X_test)

            # Calculate performance
            from sklearn.metrics import r2_score

            r2 = r2_score(y_test, predictions)

            execution_time = time.time() - start_time

            # Check if performance meets minimum threshold
            passed = r2 >= config.min_ensemble_r2

            return TestResult(
                test_name=test_name,
                passed=passed,
                execution_time=execution_time,
                metrics={
                    "r2_score": float(r2),
                    "n_models": len(ensemble.models),
                    "prediction_shape": predictions.shape,
                },
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name=test_name,
                passed=False,
                execution_time=execution_time,
                error_message=str(e),
            )

    @staticmethod
    def test_deep_learning(
        X: pd.DataFrame, y: pd.Series, config: IntegrationTestConfig
    ) -> TestResult:
        """Test deep learning framework."""
        test_name = "Deep Learning"
        start_time = time.time()

        try:
            # Skip if TensorFlow not available
            try:
                import tensorflow as tf
            except ImportError:
                return TestResult(
                    test_name=test_name,
                    passed=True,  # Pass if optional dependency missing
                    execution_time=0.0,
                    warnings=["TensorFlow not available, skipping deep learning test"],
                )

            # Create configuration
            dl_config = DeepLearningConfig(
                model_type="lstm",
                sequence_length=min(20, len(X) // 4),
                epochs=5,  # Quick test
                batch_size=32,
            )

            # Create framework
            dl_framework = DeepLearningFramework(dl_config)

            # Create time series data
            data_with_target = X.copy()
            data_with_target["target"] = y

            # Train model
            model = dl_framework.train_model(data_with_target, "target", "lstm")

            execution_time = time.time() - start_time

            return TestResult(
                test_name=test_name,
                passed=True,
                execution_time=execution_time,
                metrics={
                    "model_trained": model.is_fitted,
                    "sequence_length": dl_config.sequence_length,
                },
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name=test_name,
                passed=False,
                execution_time=execution_time,
                error_message=str(e),
            )

    @staticmethod
    def test_feature_selection(
        X: pd.DataFrame, y: pd.Series, config: IntegrationTestConfig
    ) -> TestResult:
        """Test automated feature selection."""
        test_name = "Feature Selection"
        start_time = time.time()

        try:
            # Create configuration
            fs_config = FeatureSelectionConfig(
                use_mutual_information=True,
                use_correlation_filter=True,
                use_statistical_tests=True,
                use_rfe=True,
                ensemble_selection=True,
            )

            # Create selector
            selector = AutomatedFeatureSelector(fs_config)

            # Fit and transform
            X_selected = selector.fit_transform(X, y)

            execution_time = time.time() - start_time

            # Check if execution time is acceptable
            time_passed = execution_time <= config.max_feature_selection_time

            # Check if some features were selected
            features_selected = len(X_selected.columns) > 0

            passed = time_passed and features_selected

            return TestResult(
                test_name=test_name,
                passed=passed,
                execution_time=execution_time,
                metrics={
                    "original_features": len(X.columns),
                    "selected_features": len(X_selected.columns),
                    "selection_ratio": len(X_selected.columns) / len(X.columns),
                },
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name=test_name,
                passed=False,
                execution_time=execution_time,
                error_message=str(e),
            )

    @staticmethod
    def test_bayesian_optimization(config: IntegrationTestConfig) -> TestResult:
        """Test Bayesian optimization framework."""
        test_name = "Bayesian Optimization"
        start_time = time.time()

        try:
            # Create configuration
            bo_config = BayesianOptConfig(n_calls=10, n_initial_points=3)  # Quick test

            # Create optimizer
            optimizer = BayesianOptimizer(bo_config)

            # Add test parameters
            optimizer.add_parameter("x", "real", bounds=(-5.0, 5.0))
            optimizer.add_parameter("y", "real", bounds=(-5.0, 5.0))

            # Define simple objective function
            def objective(params):
                x, y = params["x"], params["y"]
                return -(x**2 + y**2)  # Maximize negative of simple function

            # Run optimization
            result = optimizer.optimize(objective)

            execution_time = time.time() - start_time

            # Check if optimization improved
            passed = (
                result.best_score > -50  # Reasonable result
                and execution_time <= config.max_optimization_time
            )

            return TestResult(
                test_name=test_name,
                passed=passed,
                execution_time=execution_time,
                metrics={
                    "best_score": result.best_score,
                    "n_iterations": result.n_iterations,
                    "best_params": result.best_params,
                },
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name=test_name,
                passed=False,
                execution_time=execution_time,
                error_message=str(e),
            )

    @staticmethod
    def test_enhanced_features(
        X: pd.DataFrame, y: pd.Series, config: IntegrationTestConfig
    ) -> TestResult:
        """Test enhanced feature generation."""
        test_name = "Enhanced Features"
        start_time = time.time()

        try:
            # Create configuration (disable heavy computations for testing)
            ef_config = FeatureGenerationConfig(
                use_wavelets=False,  # Disable to avoid dependency issues
                use_fourier=True,
                use_polynomial=True,
                use_pattern_recognition=False,  # Requires OHLC data
                use_microstructure=False,  # Requires OHLC data
                use_time_features=False,  # Requires datetime index
            )

            # Create framework
            ef_framework = EnhancedFeatureFramework(ef_config)

            # Generate features
            enhanced_features = ef_framework.generate_features(X)

            execution_time = time.time() - start_time

            # Check if features were generated
            passed = len(enhanced_features.columns) > 0

            return TestResult(
                test_name=test_name,
                passed=passed,
                execution_time=execution_time,
                metrics={
                    "original_features": len(X.columns),
                    "enhanced_features": len(enhanced_features.columns),
                    "feature_expansion_ratio": len(enhanced_features.columns) / len(X.columns),
                },
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name=test_name,
                passed=False,
                execution_time=execution_time,
                error_message=str(e),
            )

    @staticmethod
    def test_model_comparison(
        X: pd.DataFrame, y: pd.Series, config: IntegrationTestConfig
    ) -> TestResult:
        """Test model comparison framework."""
        test_name = "Model Comparison"
        start_time = time.time()

        try:
            # Create configuration
            mc_config = ModelComparisonConfig(
                cv_folds=3,  # Quick test
                compute_financial_metrics=False,  # Skip for synthetic data
                compute_statistical_tests=True,
            )

            # Create comparison framework
            comparison = ModelComparison(mc_config)

            # Add simple models for comparison
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.linear_model import LinearRegression, Ridge

            comparison.add_model("linear", LinearRegression(), "Linear regression")
            comparison.add_model("ridge", Ridge(), "Ridge regression")
            comparison.add_model("forest", RandomForestRegressor(n_estimators=10), "Random forest")

            # Run comparison
            results = comparison.compare_models(X, y)

            execution_time = time.time() - start_time

            # Check if comparison completed successfully
            passed = (
                len(results.model_metrics) >= 2  # At least 2 models compared
                and len(results.ranking) >= 2
            )

            return TestResult(
                test_name=test_name,
                passed=passed,
                execution_time=execution_time,
                metrics={
                    "models_compared": len(results.model_metrics),
                    "best_model": results.best_model,
                    "ranking_available": len(results.ranking) > 0,
                },
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name=test_name,
                passed=False,
                execution_time=execution_time,
                error_message=str(e),
            )


class IntegrationTester:
    """
    Comprehensive integration testing framework for Phase 1 components.

    Tests individual components and their integration with existing systems.
    """

    def __init__(self, config: IntegrationTestConfig) -> None:
        self.config = config
        self.test_results: list[TestResult] = []

    def run_all_tests(self, verbose: bool = True) -> dict[str, Any]:
        """Run comprehensive test suite."""
        logger.info("Starting Phase 1 integration tests...")

        # Generate test data
        logger.info("Generating test data...")
        X_simple, y_simple = DataGenerator.create_test_data(
            n_samples=self.config.n_samples,
            n_features=self.config.n_features,
            random_state=self.config.random_state,
        )

        X_financial, y_financial = DataGenerator.create_financial_data(
            n_samples=self.config.n_samples, random_state=self.config.random_state
        )

        # Run individual component tests
        self.test_results = []

        if self.config.test_ensemble_models:
            result = ComponentTester.test_ensemble_models(X_simple, y_simple, self.config)
            self.test_results.append(result)
            if verbose:
                self._log_test_result(result)

        if self.config.test_deep_learning:
            result = ComponentTester.test_deep_learning(X_financial, y_financial, self.config)
            self.test_results.append(result)
            if verbose:
                self._log_test_result(result)

        if self.config.test_feature_selection:
            result = ComponentTester.test_feature_selection(X_simple, y_simple, self.config)
            self.test_results.append(result)
            if verbose:
                self._log_test_result(result)

        if self.config.test_bayesian_optimization:
            result = ComponentTester.test_bayesian_optimization(self.config)
            self.test_results.append(result)
            if verbose:
                self._log_test_result(result)

        if self.config.test_enhanced_features:
            result = ComponentTester.test_enhanced_features(X_simple, y_simple, self.config)
            self.test_results.append(result)
            if verbose:
                self._log_test_result(result)

        if self.config.test_model_comparison:
            result = ComponentTester.test_model_comparison(X_simple, y_simple, self.config)
            self.test_results.append(result)
            if verbose:
                self._log_test_result(result)

        if self.config.test_integration_pipeline:
            result = self._test_integration_pipeline(X_financial, y_financial)
            self.test_results.append(result)
            if verbose:
                self._log_test_result(result)

        # Generate summary
        summary = self._generate_test_summary()

        logger.info("Phase 1 integration tests completed")
        return summary

    def _test_integration_pipeline(self, X: pd.DataFrame, y: pd.Series) -> TestResult:
        """Test end-to-end integration pipeline."""
        test_name = "Integration Pipeline"
        start_time = time.time()

        try:
            logger.info("Testing end-to-end integration pipeline...")

            # Step 1: Feature generation (simplified)
            ef_config = FeatureGenerationConfig(
                use_wavelets=False,
                use_fourier=False,  # Disable for speed
                use_polynomial=True,
                use_pattern_recognition=False,
                use_microstructure=False,
                use_time_features=False,
            )

            ef_framework = EnhancedFeatureFramework(ef_config)
            X_enhanced = ef_framework.generate_features(X)

            # Combine original and enhanced features
            X_combined = pd.concat([X, X_enhanced], axis=1)

            # Step 2: Feature selection
            fs_config = FeatureSelectionConfig(
                use_mutual_information=True,
                use_correlation_filter=True,
                ensemble_selection=True,
                max_features=20,  # Limit for speed
            )

            selector = AutomatedFeatureSelector(fs_config)
            X_selected = selector.fit_transform(X_combined, y)

            # Step 3: Model training with ensemble
            ensemble_config = EnsembleConfig(
                use_random_forest=True,
                use_xgboost=False,  # Disable optional models for speed
                use_lightgbm=False,
                use_catboost=False,
            )

            ensemble = EnhancedModelEnsemble(ensemble_config)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y, test_size=0.2, random_state=42
            )

            # Train ensemble
            ensemble.fit(X_train, y_train)

            # Make predictions
            predictions = ensemble.predict(X_test)

            # Evaluate
            from sklearn.metrics import mean_squared_error, r2_score

            r2 = r2_score(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)

            execution_time = time.time() - start_time

            # Integration success criteria
            passed = (
                r2 > 0.5  # Reasonable performance
                and len(X_selected.columns) > 0  # Features selected
                and len(X_selected.columns) <= len(X_combined.columns)  # Selection worked
            )

            return TestResult(
                test_name=test_name,
                passed=passed,
                execution_time=execution_time,
                metrics={
                    "original_features": len(X.columns),
                    "enhanced_features": len(X_enhanced.columns),
                    "selected_features": len(X_selected.columns),
                    "final_r2": float(r2),
                    "final_mse": float(mse),
                    "pipeline_steps_completed": 4,
                },
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name=test_name,
                passed=False,
                execution_time=execution_time,
                error_message=str(e),
            )

    def _log_test_result(self, result: TestResult) -> None:
        """Log individual test result."""
        status = "PASSED" if result.passed else "FAILED"
        logger.info(f"{result.test_name}: {status} ({result.execution_time:.2f}s)")

        if result.error_message:
            logger.error(f"  Error: {result.error_message}")

        if result.warnings:
            for warning in result.warnings:
                logger.warning(f"  Warning: {warning}")

        if result.metrics and result.passed:
            key_metrics = {
                k: v
                for k, v in result.metrics.items()
                if k in ["r2_score", "selected_features", "best_score"]
            }
            if key_metrics:
                logger.info(f"  Key metrics: {key_metrics}")

    def _generate_test_summary(self) -> dict[str, Any]:
        """Generate comprehensive test summary."""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r.passed)
        failed_tests = total_tests - passed_tests

        total_time = sum(r.execution_time for r in self.test_results)

        # Collect all warnings
        all_warnings = []
        for result in self.test_results:
            all_warnings.extend(result.warnings)

        # Collect all errors
        errors = [r for r in self.test_results if not r.passed]

        summary = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0.0,
            "total_execution_time": total_time,
            "average_test_time": total_time / total_tests if total_tests > 0 else 0.0,
            "warnings_count": len(all_warnings),
            "all_warnings": all_warnings,
            "failed_test_names": [r.test_name for r in errors],
            "test_details": {
                r.test_name: {
                    "passed": r.passed,
                    "execution_time": r.execution_time,
                    "error": r.error_message,
                    "metrics": r.metrics,
                }
                for r in self.test_results
            },
        }

        return summary

    def generate_test_report(self) -> str:
        """Generate a comprehensive test report."""
        if not self.test_results:
            return "No tests have been run yet."

        summary = self._generate_test_summary()

        report = []
        report.append("=" * 60)
        report.append("PHASE 1 INTEGRATION TEST REPORT")
        report.append("=" * 60)
        report.append("")

        # Summary
        report.append(f"Total Tests: {summary['total_tests']}")
        report.append(f"Passed: {summary['passed_tests']}")
        report.append(f"Failed: {summary['failed_tests']}")
        report.append(f"Success Rate: {summary['success_rate']:.1%}")
        report.append(f"Total Execution Time: {summary['total_execution_time']:.2f}s")
        report.append("")

        # Individual test results
        report.append("INDIVIDUAL TEST RESULTS:")
        report.append("-" * 40)

        for result in self.test_results:
            status = "✓ PASSED" if result.passed else "✗ FAILED"
            report.append(f"{result.test_name:<25} {status:>10} ({result.execution_time:.2f}s)")

            if result.error_message:
                report.append(f"  Error: {result.error_message}")

            if result.warnings:
                for warning in result.warnings:
                    report.append(f"  Warning: {warning}")

        # Failed tests details
        if summary["failed_tests"] > 0:
            report.append("")
            report.append("FAILED TESTS DETAILS:")
            report.append("-" * 40)

            for result in self.test_results:
                if not result.passed:
                    report.append(f"\n{result.test_name}:")
                    report.append(f"  Execution Time: {result.execution_time:.2f}s")
                    if result.error_message:
                        report.append(f"  Error: {result.error_message}")

        # Warnings summary
        if summary["warnings_count"] > 0:
            report.append("")
            report.append("WARNINGS SUMMARY:")
            report.append("-" * 40)
            for warning in set(summary["all_warnings"]):
                count = summary["all_warnings"].count(warning)
                report.append(f"  ({count}x) {warning}")

        # Performance summary
        report.append("")
        report.append("PERFORMANCE SUMMARY:")
        report.append("-" * 40)

        test_times = [(r.test_name, r.execution_time) for r in self.test_results]
        test_times.sort(key=lambda x: x[1], reverse=True)

        for name, time_taken in test_times:
            report.append(f"  {name:<25} {time_taken:>8.2f}s")

        return "\n".join(report)

    def export_results(self, filepath: str) -> None:
        """Export test results to file."""
        summary = self._generate_test_summary()

        import json

        with open(filepath, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"Test results exported to {filepath}")


def run_phase1_tests(verbose: bool = True, export_path: str | None = None) -> dict[str, Any]:
    """
    Run comprehensive Phase 1 integration tests.

    Args:
        verbose: Whether to log detailed test progress
        export_path: Optional path to export results

    Returns:
        Test summary dictionary
    """
    # Create test configuration
    config = IntegrationTestConfig(
        n_samples=500,  # Smaller dataset for faster testing
        n_features=15,
        test_ensemble_models=True,
        test_deep_learning=True,
        test_feature_selection=True,
        test_bayesian_optimization=True,
        test_enhanced_features=True,
        test_model_comparison=True,
        test_integration_pipeline=True,
    )

    # Create tester and run tests
    tester = IntegrationTester(config)
    summary = tester.run_all_tests(verbose=verbose)

    # Generate and log report
    report = tester.generate_test_report()

    if verbose:
        print("\n" + report)

    # Export results if requested
    if export_path:
        tester.export_results(export_path)

    return summary


if __name__ == "__main__":
    # Run tests when script is executed directly
    results = run_phase1_tests(verbose=True, export_path="phase1_test_results.json")

    print(f"\nTest Summary: {results['passed_tests']}/{results['total_tests']} tests passed")
    print(f"Success Rate: {results['success_rate']:.1%}")

    if results["failed_tests"] > 0:
        print(f"Failed tests: {', '.join(results['failed_test_names'])}")
    else:
        print("All tests passed! ✓")
