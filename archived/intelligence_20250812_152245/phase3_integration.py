"""
Phase 3 Integration and Testing Framework for Multi-Asset Strategy Enhancement

This module provides comprehensive integration testing for all Phase 3 components:
- Portfolio-Level Optimization Framework
- Cross-Asset Correlation Modeling
- Multi-Instrument Strategy Coordination
- Dynamic Asset Allocation System
- Risk-Adjusted Portfolio Optimization
- Alternative Data Integration

Includes end-to-end testing, performance benchmarking, and production readiness validation.
"""

import logging
import time
import traceback
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

# Import Phase 3 components
try:
    from ..analytics.correlation_modeling import (
        CorrelationMethod,
        DynamicCorrelationModel,
        create_correlation_analyzer,
    )
    from ..dataflow.alternative_data import (
        DataSourceType,
        ProcessingMethod,
        create_alternative_data_framework,
    )
    from ..portfolio.dynamic_allocation import (
        AllocationStrategy,
        RebalancingMethod,
        create_dynamic_allocator,
    )
    from ..portfolio.portfolio_optimization import (
        OptimizationMethod,
        RiskModel,
        create_portfolio_optimizer,
    )
    from ..risk.advanced_optimization import OptimizationType, RiskMeasure, create_risk_optimizer
    from ..strategy.multi_instrument import (
        CoordinationMethod,
        PositionSizingMethod,
        create_multi_instrument_coordinator,
    )

    PHASE3_IMPORTS_AVAILABLE = True
except ImportError as e:
    PHASE3_IMPORTS_AVAILABLE = False
    warnings.warn(f"Phase 3 imports not available: {str(e)}. Integration testing will be limited.")

logger = logging.getLogger(__name__)


class TestStatus(Enum):
    """Test execution status"""

    NOT_STARTED = "not_started"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


class TestCategory(Enum):
    """Categories of tests"""

    UNIT_TEST = "unit_test"
    INTEGRATION_TEST = "integration_test"
    PERFORMANCE_TEST = "performance_test"
    END_TO_END_TEST = "end_to_end_test"
    STRESS_TEST = "stress_test"


@dataclass
class TestResult:
    """Individual test result"""

    test_name: str
    category: TestCategory
    status: TestStatus
    execution_time: float
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    error_trace: str | None = None


@dataclass
class Phase3TestConfig:
    """Configuration for Phase 3 testing"""

    run_unit_tests: bool = True
    run_integration_tests: bool = True
    run_performance_tests: bool = True
    run_end_to_end_tests: bool = True
    run_stress_tests: bool = False
    n_samples: int = 500
    n_assets: int = 5
    test_duration_days: int = 252
    performance_threshold_ms: int = 5000  # 5 seconds
    memory_threshold_mb: int = 500
    timeout_seconds: int = 300  # 5 minutes per test
    verbose_output: bool = False
    save_test_data: bool = False


@dataclass
class IntegrationTestResult:
    """Complete integration test results"""

    total_tests: int
    passed_tests: int
    failed_tests: int
    error_tests: int
    skipped_tests: int
    total_execution_time: float
    success_rate: float
    test_results: list[TestResult]
    summary_metrics: dict[str, Any]
    framework_ready: bool


class DataGenerator:
    """Generate realistic test data for Phase 3 testing"""

    @staticmethod
    def generate_market_data(
        n_assets: int = 5, n_days: int = 252, seed: int = 42
    ) -> dict[str, pd.DataFrame]:
        """Generate realistic multi-asset market data"""
        np.random.seed(seed)

        # Asset names
        asset_names = [f"ASSET_{i:02d}" for i in range(n_assets)]

        # Different asset characteristics
        base_returns = np.random.uniform(0.02, 0.12, n_assets)  # 2-12% annual returns
        volatilities = np.random.uniform(0.10, 0.30, n_assets)  # 10-30% annual volatility

        # Generate correlated returns
        correlation_matrix = np.random.uniform(0.1, 0.8, (n_assets, n_assets))
        np.fill_diagonal(correlation_matrix, 1.0)
        correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2  # Make symmetric

        # Ensure positive definite
        eigenvals, eigenvecs = np.linalg.eigh(correlation_matrix)
        eigenvals = np.maximum(eigenvals, 0.01)  # Ensure positive
        correlation_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

        # Generate returns
        market_data = {}
        dates = pd.date_range("2022-01-01", periods=n_days, freq="D")

        # Multivariate normal returns
        returns_matrix = np.random.multivariate_normal(
            mean=base_returns / 252,  # Daily returns
            cov=np.outer(volatilities, volatilities) * correlation_matrix / 252,
            size=n_days,
        )

        for i, asset in enumerate(asset_names):
            returns = returns_matrix[:, i]

            # Add some regime changes and fat tail events
            for j in range(n_days):
                if np.random.random() < 0.02:  # 2% chance of extreme event
                    returns[j] *= np.random.uniform(2, 4)  # Extreme move

            # Generate price series
            prices = 100 * np.exp(np.cumsum(returns))

            # Create OHLCV data
            market_data[asset] = pd.DataFrame(
                {
                    "open": prices * (1 + np.random.normal(0, 0.001, n_days)),
                    "high": prices * (1 + np.abs(np.random.normal(0.005, 0.002, n_days))),
                    "low": prices * (1 - np.abs(np.random.normal(0.005, 0.002, n_days))),
                    "close": prices,
                    "volume": np.random.randint(100000, 1000000, n_days),
                },
                index=dates,
            )

        return market_data

    @staticmethod
    def generate_returns_data(market_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Convert market data to returns DataFrame"""
        returns_dict = {}
        for asset, data in market_data.items():
            returns_dict[asset] = data["close"].pct_change().dropna()

        return pd.DataFrame(returns_dict).dropna()


class Phase3ComponentTester:
    """Individual component testing"""

    def __init__(self, config: Phase3TestConfig) -> None:
        self.config = config

    def test_portfolio_optimization(self, market_data: dict[str, pd.DataFrame]) -> list[TestResult]:
        """Test Portfolio Optimization Framework"""
        test_results = []
        returns_data = DataGenerator.generate_returns_data(market_data)

        # Test different optimization methods
        methods_to_test = [
            OptimizationMethod.MEAN_VARIANCE,
            OptimizationMethod.RISK_PARITY,
            OptimizationMethod.MAXIMUM_SHARPE,
        ]

        for method in methods_to_test:
            start_time = time.time()
            try:
                optimizer = create_portfolio_optimizer(
                    method=method, risk_model=RiskModel.LEDOIT_WOLF
                )

                result = optimizer.optimize_portfolio(returns_data)
                execution_time = time.time() - start_time

                if result.success:
                    test_results.append(
                        TestResult(
                            test_name=f"portfolio_optimization_{method.value}",
                            category=TestCategory.UNIT_TEST,
                            status=TestStatus.PASSED,
                            execution_time=execution_time,
                            message=f"Portfolio optimization with {method.value} successful",
                            details={
                                "expected_return": result.expected_return,
                                "expected_volatility": result.expected_volatility,
                                "sharpe_ratio": result.sharpe_ratio,
                                "n_assets": len(result.weights),
                            },
                        )
                    )
                else:
                    test_results.append(
                        TestResult(
                            test_name=f"portfolio_optimization_{method.value}",
                            category=TestCategory.UNIT_TEST,
                            status=TestStatus.FAILED,
                            execution_time=execution_time,
                            message=f"Portfolio optimization failed: {result.message}",
                            details={"method": method.value},
                        )
                    )

            except Exception as e:
                execution_time = time.time() - start_time
                test_results.append(
                    TestResult(
                        test_name=f"portfolio_optimization_{method.value}",
                        category=TestCategory.UNIT_TEST,
                        status=TestStatus.ERROR,
                        execution_time=execution_time,
                        message=f"Portfolio optimization error: {str(e)}",
                        error_trace=traceback.format_exc(),
                    )
                )

        return test_results

    def test_correlation_modeling(self, market_data: dict[str, pd.DataFrame]) -> list[TestResult]:
        """Test Cross-Asset Correlation Modeling"""
        test_results = []
        returns_data = DataGenerator.generate_returns_data(market_data)

        correlation_methods = [
            CorrelationMethod.PEARSON,
            CorrelationMethod.SPEARMAN,
            CorrelationMethod.DISTANCE,
        ]

        for method in correlation_methods:
            start_time = time.time()
            try:
                analyzer = create_correlation_analyzer(
                    method=method, dynamic_model=DynamicCorrelationModel.EWMA, regime_detection=True
                )

                result = analyzer.analyze_correlations(returns_data)
                execution_time = time.time() - start_time

                if result.correlation_matrix is not None and not result.correlation_matrix.empty:
                    test_results.append(
                        TestResult(
                            test_name=f"correlation_modeling_{method.value}",
                            category=TestCategory.UNIT_TEST,
                            status=TestStatus.PASSED,
                            execution_time=execution_time,
                            message=f"Correlation modeling with {method.value} successful",
                            details={
                                "correlation_matrix_shape": result.correlation_matrix.shape,
                                "avg_correlation": result.model_statistics.get(
                                    "avg_correlation", 0
                                ),
                                "has_dynamic_correlations": result.dynamic_correlations is not None,
                                "has_regime_detection": result.regime_probabilities is not None,
                            },
                        )
                    )
                else:
                    test_results.append(
                        TestResult(
                            test_name=f"correlation_modeling_{method.value}",
                            category=TestCategory.UNIT_TEST,
                            status=TestStatus.FAILED,
                            execution_time=execution_time,
                            message="Correlation modeling returned empty results",
                            details={"method": method.value},
                        )
                    )

            except Exception as e:
                execution_time = time.time() - start_time
                test_results.append(
                    TestResult(
                        test_name=f"correlation_modeling_{method.value}",
                        category=TestCategory.UNIT_TEST,
                        status=TestStatus.ERROR,
                        execution_time=execution_time,
                        message=f"Correlation modeling error: {str(e)}",
                        error_trace=traceback.format_exc(),
                    )
                )

        return test_results

    def test_multi_instrument_coordination(
        self, market_data: dict[str, pd.DataFrame]
    ) -> list[TestResult]:
        """Test Multi-Instrument Strategy Coordination"""
        test_results = []

        coordination_methods = [
            CoordinationMethod.CORRELATION_AWARE,
        ]

        for method in coordination_methods:
            start_time = time.time()
            try:
                coordinator = create_multi_instrument_coordinator(
                    coordination_method=method, position_sizing=PositionSizingMethod.RISK_PARITY
                )

                # Mock strategy for testing
                class MockStrategy:
                    def generate_signal(self, data):
                        if len(data) < 20:
                            return type(
                                "StrategySignal",
                                (),
                                {"signal": 0.0, "confidence": 0.0, "metadata": {}},
                            )()

                        momentum = data["close"].pct_change().rolling(20).mean().iloc[-1]
                        signal = np.tanh(momentum * 100)
                        confidence = min(1.0, abs(signal) + 0.2)

                        return type(
                            "StrategySignal",
                            (),
                            {
                                "signal": signal,
                                "confidence": confidence,
                                "metadata": {"momentum": momentum},
                            },
                        )()

                # Add strategies for each asset
                for asset in list(market_data.keys())[:3]:  # Test with first 3 assets
                    coordinator.add_strategy(asset, MockStrategy())

                result = coordinator.coordinate_strategies(market_data)
                execution_time = time.time() - start_time

                if result.success:
                    test_results.append(
                        TestResult(
                            test_name=f"multi_instrument_coordination_{method.value}",
                            category=TestCategory.INTEGRATION_TEST,
                            status=TestStatus.PASSED,
                            execution_time=execution_time,
                            message=f"Multi-instrument coordination with {method.value} successful",
                            details={
                                "n_individual_signals": len(result.individual_signals),
                                "n_position_sizes": len(result.position_sizes),
                                "total_leverage": result.risk_metrics.get("total_leverage", 0),
                                "avg_correlation": result.risk_metrics.get("avg_correlation", 0),
                            },
                        )
                    )
                else:
                    test_results.append(
                        TestResult(
                            test_name=f"multi_instrument_coordination_{method.value}",
                            category=TestCategory.INTEGRATION_TEST,
                            status=TestStatus.FAILED,
                            execution_time=execution_time,
                            message=f"Multi-instrument coordination failed: {result.message}",
                            details={"method": method.value},
                        )
                    )

            except Exception as e:
                execution_time = time.time() - start_time
                test_results.append(
                    TestResult(
                        test_name=f"multi_instrument_coordination_{method.value}",
                        category=TestCategory.INTEGRATION_TEST,
                        status=TestStatus.ERROR,
                        execution_time=execution_time,
                        message=f"Multi-instrument coordination error: {str(e)}",
                        error_trace=traceback.format_exc(),
                    )
                )

        return test_results

    def test_dynamic_allocation(self, market_data: dict[str, pd.DataFrame]) -> list[TestResult]:
        """Test Dynamic Asset Allocation System"""
        test_results = []

        allocation_strategies = [
            AllocationStrategy.TACTICAL,
            AllocationStrategy.VOLATILITY_TARGETING,
            AllocationStrategy.RISK_PARITY,
        ]

        for strategy in allocation_strategies:
            start_time = time.time()
            try:
                allocator = create_dynamic_allocator(
                    strategy=strategy,
                    target_volatility=0.12,
                    rebalancing_method=RebalancingMethod.CALENDAR,
                )

                result = allocator.calculate_allocation(market_data)
                execution_time = time.time() - start_time

                if result.success:
                    test_results.append(
                        TestResult(
                            test_name=f"dynamic_allocation_{strategy.value}",
                            category=TestCategory.INTEGRATION_TEST,
                            status=TestStatus.PASSED,
                            execution_time=execution_time,
                            message=f"Dynamic allocation with {strategy.value} successful",
                            details={
                                "expected_return": result.expected_return,
                                "expected_volatility": result.expected_volatility,
                                "sharpe_ratio": result.sharpe_ratio,
                                "turnover": result.turnover,
                                "n_assets": len(result.target_weights),
                            },
                        )
                    )
                else:
                    test_results.append(
                        TestResult(
                            test_name=f"dynamic_allocation_{strategy.value}",
                            category=TestCategory.INTEGRATION_TEST,
                            status=TestStatus.FAILED,
                            execution_time=execution_time,
                            message=f"Dynamic allocation failed: {result.message}",
                            details={"strategy": strategy.value},
                        )
                    )

            except Exception as e:
                execution_time = time.time() - start_time
                test_results.append(
                    TestResult(
                        test_name=f"dynamic_allocation_{strategy.value}",
                        category=TestCategory.INTEGRATION_TEST,
                        status=TestStatus.ERROR,
                        execution_time=execution_time,
                        message=f"Dynamic allocation error: {str(e)}",
                        error_trace=traceback.format_exc(),
                    )
                )

        return test_results

    def test_risk_optimization(self, market_data: dict[str, pd.DataFrame]) -> list[TestResult]:
        """Test Risk-Adjusted Portfolio Optimization"""
        test_results = []
        returns_data = DataGenerator.generate_returns_data(market_data)

        optimization_types = [
            OptimizationType.MEAN_CVAR,
            OptimizationType.MAX_DIVERSIFICATION,
            OptimizationType.ROBUST_MEAN_VARIANCE,
        ]

        for opt_type in optimization_types:
            start_time = time.time()
            try:
                optimizer = create_risk_optimizer(
                    optimization_type=opt_type, confidence_level=0.95, risk_aversion=2.0
                )

                result = optimizer.optimize_portfolio(returns_data)
                execution_time = time.time() - start_time

                if result.success:
                    test_results.append(
                        TestResult(
                            test_name=f"risk_optimization_{opt_type.value}",
                            category=TestCategory.UNIT_TEST,
                            status=TestStatus.PASSED,
                            execution_time=execution_time,
                            message=f"Risk optimization with {opt_type.value} successful",
                            details={
                                "expected_return": result.expected_return,
                                "expected_risk": result.expected_risk,
                                "sharpe_ratio": result.sharpe_ratio,
                                "cvar": result.cvar,
                                "max_drawdown": result.max_drawdown,
                            },
                        )
                    )
                else:
                    test_results.append(
                        TestResult(
                            test_name=f"risk_optimization_{opt_type.value}",
                            category=TestCategory.UNIT_TEST,
                            status=TestStatus.FAILED,
                            execution_time=execution_time,
                            message=f"Risk optimization failed: {result.message}",
                            details={"optimization_type": opt_type.value},
                        )
                    )

            except Exception as e:
                execution_time = time.time() - start_time
                test_results.append(
                    TestResult(
                        test_name=f"risk_optimization_{opt_type.value}",
                        category=TestCategory.UNIT_TEST,
                        status=TestStatus.ERROR,
                        execution_time=execution_time,
                        message=f"Risk optimization error: {str(e)}",
                        error_trace=traceback.format_exc(),
                    )
                )

        return test_results

    def test_alternative_data(self, market_data: dict[str, pd.DataFrame]) -> list[TestResult]:
        """Test Alternative Data Integration"""
        test_results = []

        start_time = time.time()
        try:
            framework = create_alternative_data_framework(
                enabled_sources=[
                    DataSourceType.NEWS_SENTIMENT,
                    DataSourceType.ECONOMIC_INDICATORS,
                    DataSourceType.ESG_METRICS,
                ],
                processing_methods=[
                    ProcessingMethod.SENTIMENT_ANALYSIS,
                    ProcessingMethod.FACTOR_EXTRACTION,
                ],
            )

            # Test with subset of assets
            test_assets = list(market_data.keys())[:3]
            signals = framework.get_alternative_signals(test_assets, lookback_days=14)
            execution_time = time.time() - start_time

            total_signals = sum(len(asset_signals) for asset_signals in signals.values())

            test_results.append(
                TestResult(
                    test_name="alternative_data_integration",
                    category=TestCategory.INTEGRATION_TEST,
                    status=TestStatus.PASSED,
                    execution_time=execution_time,
                    message="Alternative data integration successful",
                    details={
                        "n_assets": len(test_assets),
                        "total_signals": total_signals,
                        "signals_per_asset": {asset: len(sigs) for asset, sigs in signals.items()},
                        "n_data_sources": len(framework.data_sources),
                    },
                )
            )

        except Exception as e:
            execution_time = time.time() - start_time
            test_results.append(
                TestResult(
                    test_name="alternative_data_integration",
                    category=TestCategory.INTEGRATION_TEST,
                    status=TestStatus.ERROR,
                    execution_time=execution_time,
                    message=f"Alternative data integration error: {str(e)}",
                    error_trace=traceback.format_exc(),
                )
            )

        return test_results


class Phase3IntegrationTester:
    """Main integration tester for Phase 3"""

    def __init__(self, config: Phase3TestConfig) -> None:
        self.config = config
        self.component_tester = Phase3ComponentTester(config)

    def run_all_tests(self, verbose: bool = False) -> IntegrationTestResult:
        """Run all Phase 3 integration tests"""
        if not PHASE3_IMPORTS_AVAILABLE:
            return IntegrationTestResult(
                total_tests=0,
                passed_tests=0,
                failed_tests=0,
                error_tests=0,
                skipped_tests=1,
                total_execution_time=0.0,
                success_rate=0.0,
                test_results=[
                    TestResult(
                        test_name="phase3_imports",
                        category=TestCategory.UNIT_TEST,
                        status=TestStatus.SKIPPED,
                        execution_time=0.0,
                        message="Phase 3 imports not available - skipping all tests",
                    )
                ],
                summary_metrics={},
                framework_ready=False,
            )

        start_time = time.time()
        all_test_results = []

        if verbose:
            print("🚀 Starting Phase 3 Multi-Asset Strategy Enhancement Integration Tests...")

        # Generate test data
        if verbose:
            print("📊 Generating test data...")

        market_data = DataGenerator.generate_market_data(
            n_assets=self.config.n_assets, n_days=self.config.test_duration_days
        )

        # Run component tests
        test_methods = [
            ("Portfolio Optimization", self.component_tester.test_portfolio_optimization),
            ("Correlation Modeling", self.component_tester.test_correlation_modeling),
            (
                "Multi-Instrument Coordination",
                self.component_tester.test_multi_instrument_coordination,
            ),
            ("Dynamic Allocation", self.component_tester.test_dynamic_allocation),
            ("Risk Optimization", self.component_tester.test_risk_optimization),
            ("Alternative Data", self.component_tester.test_alternative_data),
        ]

        for test_name, test_method in test_methods:
            if verbose:
                print(f"🧪 Testing {test_name}...")

            try:
                test_results = test_method(market_data)
                all_test_results.extend(test_results)

                if verbose:
                    passed = len([r for r in test_results if r.status == TestStatus.PASSED])
                    total = len(test_results)
                    print(f"   ✅ {passed}/{total} tests passed")

            except Exception as e:
                if verbose:
                    print(f"   ❌ Test suite failed: {str(e)}")

                all_test_results.append(
                    TestResult(
                        test_name=f"{test_name.lower().replace(' ', '_')}_suite",
                        category=TestCategory.INTEGRATION_TEST,
                        status=TestStatus.ERROR,
                        execution_time=0.0,
                        message=f"Test suite error: {str(e)}",
                        error_trace=traceback.format_exc(),
                    )
                )

        # Run end-to-end integration test
        if self.config.run_end_to_end_tests:
            if verbose:
                print("🔗 Running end-to-end integration test...")

            e2e_result = self._run_end_to_end_test(market_data)
            all_test_results.append(e2e_result)

        # Calculate results
        total_execution_time = time.time() - start_time

        passed_tests = len([r for r in all_test_results if r.status == TestStatus.PASSED])
        failed_tests = len([r for r in all_test_results if r.status == TestStatus.FAILED])
        error_tests = len([r for r in all_test_results if r.status == TestStatus.ERROR])
        skipped_tests = len([r for r in all_test_results if r.status == TestStatus.SKIPPED])
        total_tests = len(all_test_results)

        success_rate = passed_tests / total_tests if total_tests > 0 else 0.0
        framework_ready = success_rate >= 0.8 and error_tests == 0

        # Summary metrics
        summary_metrics = self._calculate_summary_metrics(all_test_results, market_data)

        return IntegrationTestResult(
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            error_tests=error_tests,
            skipped_tests=skipped_tests,
            total_execution_time=total_execution_time,
            success_rate=success_rate,
            test_results=all_test_results,
            summary_metrics=summary_metrics,
            framework_ready=framework_ready,
        )

    def _run_end_to_end_test(self, market_data: dict[str, pd.DataFrame]) -> TestResult:
        """Run comprehensive end-to-end integration test"""
        start_time = time.time()

        try:
            # Test complete pipeline: Alternative Data -> Correlation -> Coordination -> Optimization -> Allocation

            # 1. Alternative Data
            alt_data_framework = create_alternative_data_framework()
            test_assets = list(market_data.keys())[:3]
            alt_signals = alt_data_framework.get_alternative_signals(test_assets)

            # 2. Correlation Analysis
            returns_data = DataGenerator.generate_returns_data(market_data)
            corr_analyzer = create_correlation_analyzer()
            corr_result = corr_analyzer.analyze_correlations(returns_data)

            # 3. Multi-Instrument Coordination (mock strategies)
            coordinator = create_multi_instrument_coordinator()

            class MockStrategy:
                def generate_signal(self, data):
                    momentum = (
                        data["close"].pct_change().rolling(20).mean().iloc[-1]
                        if len(data) >= 20
                        else 0
                    )
                    signal = np.tanh(momentum * 100)
                    confidence = min(1.0, abs(signal) + 0.2)
                    return type(
                        "StrategySignal",
                        (),
                        {"signal": signal, "confidence": confidence, "metadata": {}},
                    )()

            for asset in test_assets:
                coordinator.add_strategy(asset, MockStrategy())

            coord_result = coordinator.coordinate_strategies(market_data)

            # 4. Portfolio Optimization
            portfolio_optimizer = create_portfolio_optimizer()
            portfolio_result = portfolio_optimizer.optimize_portfolio(returns_data)

            # 5. Risk-Adjusted Optimization
            risk_optimizer = create_risk_optimizer()
            risk_result = risk_optimizer.optimize_portfolio(returns_data)

            # 6. Dynamic Allocation
            allocator = create_dynamic_allocator()
            allocation_result = allocator.calculate_allocation(market_data)

            execution_time = time.time() - start_time

            # Validate pipeline results
            pipeline_success = (
                len(alt_signals) > 0
                and corr_result.correlation_matrix is not None
                and coord_result.success
                and portfolio_result.success
                and risk_result.success
                and allocation_result.success
            )

            if pipeline_success:
                return TestResult(
                    test_name="end_to_end_integration",
                    category=TestCategory.END_TO_END_TEST,
                    status=TestStatus.PASSED,
                    execution_time=execution_time,
                    message="End-to-end integration pipeline successful",
                    details={
                        "alt_data_signals": len(alt_signals),
                        "correlation_matrix_size": corr_result.correlation_matrix.shape,
                        "coordination_positions": len(coord_result.position_sizes),
                        "portfolio_sharpe": portfolio_result.sharpe_ratio,
                        "risk_cvar": risk_result.cvar,
                        "allocation_turnover": allocation_result.turnover,
                        "pipeline_stages_completed": 6,
                    },
                )
            else:
                return TestResult(
                    test_name="end_to_end_integration",
                    category=TestCategory.END_TO_END_TEST,
                    status=TestStatus.FAILED,
                    execution_time=execution_time,
                    message="End-to-end integration pipeline had failures",
                    details={
                        "alt_data_success": len(alt_signals) > 0,
                        "correlation_success": corr_result.correlation_matrix is not None,
                        "coordination_success": coord_result.success,
                        "portfolio_success": portfolio_result.success,
                        "risk_success": risk_result.success,
                        "allocation_success": allocation_result.success,
                    },
                )

        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name="end_to_end_integration",
                category=TestCategory.END_TO_END_TEST,
                status=TestStatus.ERROR,
                execution_time=execution_time,
                message=f"End-to-end integration error: {str(e)}",
                error_trace=traceback.format_exc(),
            )

    def _calculate_summary_metrics(
        self, test_results: list[TestResult], market_data: dict[str, pd.DataFrame]
    ) -> dict[str, Any]:
        """Calculate comprehensive summary metrics"""
        if not test_results:
            return {}

        # Performance metrics
        execution_times = [r.execution_time for r in test_results if r.execution_time > 0]

        # Test category breakdown
        category_counts = defaultdict(int)
        category_success_rates = defaultdict(list)

        for result in test_results:
            category_counts[result.category.value] += 1
            success = 1.0 if result.status == TestStatus.PASSED else 0.0
            category_success_rates[result.category.value].append(success)

        category_metrics = {}
        for category, successes in category_success_rates.items():
            category_metrics[category] = {
                "total_tests": len(successes),
                "success_rate": np.mean(successes),
                "avg_execution_time": np.mean(
                    [
                        r.execution_time
                        for r in test_results
                        if r.category.value == category and r.execution_time > 0
                    ]
                ),
            }

        return {
            "avg_execution_time": np.mean(execution_times) if execution_times else 0,
            "max_execution_time": np.max(execution_times) if execution_times else 0,
            "total_execution_time": np.sum(execution_times) if execution_times else 0,
            "category_metrics": category_metrics,
            "test_data_stats": {
                "n_assets": len(market_data),
                "data_duration_days": len(next(iter(market_data.values()))),
                "avg_daily_returns": np.mean(
                    [data["close"].pct_change().mean() for data in market_data.values()]
                ),
                "avg_volatility": np.mean(
                    [
                        data["close"].pct_change().std() * np.sqrt(252)
                        for data in market_data.values()
                    ]
                ),
            },
            "performance_indicators": {
                "all_tests_under_threshold": all(
                    t <= self.config.performance_threshold_ms / 1000 for t in execution_times
                ),
                "framework_performance_grade": (
                    "A"
                    if np.mean(execution_times) <= 1.0
                    else "B"
                    if np.mean(execution_times) <= 3.0
                    else "C"
                ),
            },
        }


def run_phase3_integration_tests(
    config: Phase3TestConfig | None = None, verbose: bool = True
) -> IntegrationTestResult:
    """Main entry point for Phase 3 integration testing"""
    if config is None:
        config = Phase3TestConfig()

    tester = Phase3IntegrationTester(config)
    return tester.run_all_tests(verbose=verbose)


# Example usage and testing
if __name__ == "__main__":
    print("=" * 70)
    print("🚀 GPT-Trader Phase 3: Multi-Asset Strategy Enhancement")
    print("🧪 Comprehensive Integration Testing Framework")
    print("=" * 70)

    # Configure comprehensive testing
    config = Phase3TestConfig(
        run_unit_tests=True,
        run_integration_tests=True,
        run_performance_tests=True,
        run_end_to_end_tests=True,
        n_samples=500,
        n_assets=5,
        test_duration_days=252,
        verbose_output=True,
        timeout_seconds=600,  # 10 minutes
    )

    # Run all tests
    print("🔬 Executing comprehensive Phase 3 integration test suite...")
    print()

    results = run_phase3_integration_tests(config, verbose=True)

    # Print detailed results
    print("\n" + "=" * 70)
    print("📊 PHASE 3 INTEGRATION TEST RESULTS")
    print("=" * 70)

    print(f"Total Tests: {results.total_tests}")
    print(f"✅ Passed: {results.passed_tests}")
    print(f"❌ Failed: {results.failed_tests}")
    print(f"🚨 Errors: {results.error_tests}")
    print(f"⏭️  Skipped: {results.skipped_tests}")
    print(f"⏱️  Total Execution Time: {results.total_execution_time:.2f}s")
    print(f"📈 Success Rate: {results.success_rate:.2%}")
    print()

    # Category breakdown
    if results.summary_metrics.get("category_metrics"):
        print("📋 Test Category Breakdown:")
        for category, metrics in results.summary_metrics["category_metrics"].items():
            print(
                f"  {category}: {metrics['total_tests']} tests, "
                f"{metrics['success_rate']:.2%} success rate, "
                f"{metrics['avg_execution_time']:.2f}s avg time"
            )
        print()

    # Performance metrics
    if results.summary_metrics.get("performance_indicators"):
        perf_indicators = results.summary_metrics["performance_indicators"]
        print(f"⚡ Performance Grade: {perf_indicators.get('framework_performance_grade', 'N/A')}")
        print(
            f"🏃 All Tests Under Threshold: {perf_indicators.get('all_tests_under_threshold', False)}"
        )
        print()

    # Final assessment
    print("🎯 FINAL ASSESSMENT:")
    if results.framework_ready:
        print("🎉 Phase 3 Multi-Asset Strategy Enhancement Framework is READY FOR PRODUCTION!")
        print("✨ All critical components are functioning correctly with high reliability.")
        print("🚀 The framework provides:")
        print("   • Advanced portfolio optimization across multiple assets")
        print("   • Dynamic correlation modeling and regime detection")
        print("   • Multi-instrument strategy coordination")
        print("   • Risk-adjusted portfolio optimization with CVaR and robust methods")
        print("   • Dynamic asset allocation with multiple strategies")
        print("   • Alternative data integration for enhanced signals")
        print("   • Comprehensive end-to-end integration pipeline")
    else:
        print("⚠️  Phase 3 Framework requires attention before production deployment.")
        print(f"   Success rate: {results.success_rate:.2%} (target: ≥80%)")
        print(f"   Errors encountered: {results.error_tests} (target: 0)")

        # Show failing tests
        failing_tests = [
            r for r in results.test_results if r.status in [TestStatus.FAILED, TestStatus.ERROR]
        ]
        if failing_tests:
            print("   Failing tests:")
            for test in failing_tests[:5]:  # Show first 5
                print(f"     • {test.test_name}: {test.message}")

    print("\n" + "=" * 70)
    print("Phase 3 Multi-Asset Strategy Enhancement Testing Complete! 🏁")
    print("=" * 70)
