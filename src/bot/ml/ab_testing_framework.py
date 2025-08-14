"""
A/B Testing Framework for Model Comparison
Phase 3, Week 1: MON-009
Statistical framework for comparing model performance
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class TestStatus(Enum):
    """A/B test status"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    STOPPED = "stopped"
    FAILED = "failed"


class AllocationStrategy(Enum):
    """Traffic allocation strategies"""

    FIXED = "fixed"  # Fixed 50/50 split
    EPSILON_GREEDY = "epsilon_greedy"  # Explore vs exploit
    THOMPSON_SAMPLING = "thompson_sampling"  # Bayesian approach
    UCB = "upper_confidence_bound"  # Upper confidence bound


@dataclass
class ABTestConfig:
    """Configuration for A/B test"""

    test_id: str
    model_a_id: str
    model_b_id: str

    # Test parameters
    metric: str = "accuracy"  # Primary metric to optimize
    secondary_metrics: list[str] = field(default_factory=list)

    # Statistical parameters
    confidence_level: float = 0.95  # Statistical confidence
    power: float = 0.80  # Statistical power
    minimum_detectable_effect: float = 0.02  # 2% improvement

    # Sample size
    min_samples_per_variant: int = 100
    max_samples_per_variant: int = 10000

    # Allocation
    allocation_strategy: AllocationStrategy = AllocationStrategy.FIXED
    initial_split: tuple[float, float] = (0.5, 0.5)

    # Duration
    min_duration_hours: int = 24
    max_duration_hours: int = 168  # 1 week

    # Early stopping
    enable_early_stopping: bool = True
    sequential_testing: bool = True  # Use sequential probability ratio test

    # Segmentation
    segment_by: list[str] | None = None  # Features to segment by

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "test_id": self.test_id,
            "model_a_id": self.model_a_id,
            "model_b_id": self.model_b_id,
            "metric": self.metric,
            "secondary_metrics": self.secondary_metrics,
            "confidence_level": self.confidence_level,
            "power": self.power,
            "minimum_detectable_effect": self.minimum_detectable_effect,
            "allocation_strategy": self.allocation_strategy.value,
            "enable_early_stopping": self.enable_early_stopping,
        }


@dataclass
class ABTestResult:
    """Results from an A/B test"""

    test_id: str
    status: TestStatus

    # Sample sizes
    n_samples_a: int
    n_samples_b: int

    # Performance metrics
    metric_a: float
    metric_b: float
    metric_difference: float
    relative_improvement: float  # Percentage improvement

    # Statistical results
    p_value: float
    confidence_interval: tuple[float, float]
    is_significant: bool
    statistical_power: float

    # Additional metrics
    secondary_metrics_a: dict[str, float]
    secondary_metrics_b: dict[str, float]

    # Test metadata
    start_time: datetime
    end_time: datetime | None
    duration_hours: float

    # Winner
    winner: str | None  # 'A', 'B', or None
    recommendation: str

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "test_id": self.test_id,
            "status": self.status.value,
            "n_samples": {"A": self.n_samples_a, "B": self.n_samples_b},
            "metrics": {
                "A": self.metric_a,
                "B": self.metric_b,
                "difference": self.metric_difference,
                "relative_improvement": self.relative_improvement,
            },
            "statistics": {
                "p_value": self.p_value,
                "confidence_interval": self.confidence_interval,
                "is_significant": self.is_significant,
                "power": self.statistical_power,
            },
            "winner": self.winner,
            "recommendation": self.recommendation,
            "duration_hours": self.duration_hours,
        }


class ABTestingFramework:
    """
    Framework for conducting A/B tests between models.

    Features:
    - Statistical significance testing
    - Sample size calculation
    - Early stopping
    - Multiple allocation strategies
    - Segmented analysis
    """

    def __init__(self):
        """Initialize A/B testing framework"""
        self.active_tests: dict[str, dict] = {}
        self.completed_tests: list[ABTestResult] = []
        self.test_data: dict[str, dict] = {}

        logger.info("A/B Testing Framework initialized")

    def create_test(self, config: ABTestConfig) -> str:
        """
        Create a new A/B test.

        Args:
            config: Test configuration

        Returns:
            Test ID
        """
        # Calculate required sample size
        required_samples = self._calculate_sample_size(
            config.minimum_detectable_effect, config.confidence_level, config.power
        )

        # Initialize test
        test_info = {
            "config": config,
            "status": TestStatus.PENDING,
            "start_time": None,
            "samples_a": [],
            "samples_b": [],
            "metrics_a": defaultdict(list),
            "metrics_b": defaultdict(list),
            "allocation_counts": {"A": 0, "B": 0},
            "required_samples": required_samples,
            "current_split": list(config.initial_split),
        }

        self.active_tests[config.test_id] = test_info
        self.test_data[config.test_id] = {
            "predictions_a": [],
            "predictions_b": [],
            "actuals_a": [],
            "actuals_b": [],
        }

        logger.info(
            f"Created A/B test {config.test_id}: {config.model_a_id} vs {config.model_b_id}"
        )
        logger.info(f"Required samples per variant: {required_samples}")

        return config.test_id

    def start_test(self, test_id: str) -> None:
        """Start an A/B test"""
        if test_id not in self.active_tests:
            raise ValueError(f"Test {test_id} not found")

        test = self.active_tests[test_id]
        test["status"] = TestStatus.RUNNING
        test["start_time"] = datetime.now()

        logger.info(f"Started A/B test {test_id}")

    def assign_variant(self, test_id: str, features: pd.DataFrame | None = None) -> str:
        """
        Assign a variant (A or B) for a new sample.

        Args:
            test_id: Test identifier
            features: Optional features for contextual assignment

        Returns:
            'A' or 'B'
        """
        if test_id not in self.active_tests:
            raise ValueError(f"Test {test_id} not found")

        test = self.active_tests[test_id]
        config = test["config"]

        if test["status"] != TestStatus.RUNNING:
            raise ValueError(f"Test {test_id} is not running")

        # Get allocation strategy
        strategy = config.allocation_strategy

        if strategy == AllocationStrategy.FIXED:
            # Fixed split
            variant = "A" if np.random.random() < test["current_split"][0] else "B"

        elif strategy == AllocationStrategy.EPSILON_GREEDY:
            # Epsilon-greedy: explore vs exploit
            epsilon = 0.1  # 10% exploration
            if np.random.random() < epsilon:
                # Explore: random assignment
                variant = np.random.choice(["A", "B"])
            else:
                # Exploit: choose better performing
                if len(test["metrics_a"][config.metric]) > 10:
                    avg_a = np.mean(test["metrics_a"][config.metric][-50:])
                    avg_b = np.mean(test["metrics_b"][config.metric][-50:])
                    variant = "A" if avg_a >= avg_b else "B"
                else:
                    variant = np.random.choice(["A", "B"])

        elif strategy == AllocationStrategy.THOMPSON_SAMPLING:
            # Thompson sampling: Bayesian approach
            variant = self._thompson_sampling_assignment(test, config)

        elif strategy == AllocationStrategy.UCB:
            # Upper confidence bound
            variant = self._ucb_assignment(test, config)

        else:
            # Default to fixed split
            variant = "A" if np.random.random() < 0.5 else "B"

        # Update allocation count
        test["allocation_counts"][variant] += 1

        return variant

    def record_outcome(
        self,
        test_id: str,
        variant: str,
        prediction: float | np.ndarray,
        actual: float | np.ndarray,
        confidence: float | None = None,
        features: pd.DataFrame | None = None,
    ) -> None:
        """
        Record the outcome of a prediction.

        Args:
            test_id: Test identifier
            variant: 'A' or 'B'
            prediction: Model prediction
            actual: Actual value
            confidence: Prediction confidence
            features: Optional features for segmentation
        """
        if test_id not in self.active_tests:
            raise ValueError(f"Test {test_id} not found")

        test = self.active_tests[test_id]
        config = test["config"]

        # Calculate metrics
        if config.metric == "accuracy":
            metric_value = float(prediction == actual)
        elif config.metric == "mse":
            metric_value = float((prediction - actual) ** 2)
        elif config.metric == "mae":
            metric_value = float(abs(prediction - actual))
        else:
            # Custom metric
            metric_value = float(prediction == actual)

        # Store data
        if variant == "A":
            test["samples_a"].append({"prediction": prediction, "actual": actual})
            test["metrics_a"][config.metric].append(metric_value)
            self.test_data[test_id]["predictions_a"].append(prediction)
            self.test_data[test_id]["actuals_a"].append(actual)
        else:
            test["samples_b"].append({"prediction": prediction, "actual": actual})
            test["metrics_b"][config.metric].append(metric_value)
            self.test_data[test_id]["predictions_b"].append(prediction)
            self.test_data[test_id]["actuals_b"].append(actual)

        # Check for early stopping
        if config.enable_early_stopping:
            self._check_early_stopping(test_id)

    def get_current_results(self, test_id: str) -> ABTestResult | None:
        """
        Get current results of an A/B test.

        Args:
            test_id: Test identifier

        Returns:
            Current test results or None if insufficient data
        """
        if test_id not in self.active_tests:
            raise ValueError(f"Test {test_id} not found")

        test = self.active_tests[test_id]
        config = test["config"]

        # Check if we have enough samples
        n_a = len(test["metrics_a"][config.metric])
        n_b = len(test["metrics_b"][config.metric])

        if n_a < 10 or n_b < 10:
            return None

        # Calculate metrics
        metric_a = np.mean(test["metrics_a"][config.metric])
        metric_b = np.mean(test["metrics_b"][config.metric])

        # Perform statistical test
        statistic, p_value = stats.ttest_ind(
            test["metrics_a"][config.metric], test["metrics_b"][config.metric]
        )

        # Calculate confidence interval
        difference = metric_b - metric_a
        se_diff = np.sqrt(
            np.var(test["metrics_a"][config.metric]) / n_a
            + np.var(test["metrics_b"][config.metric]) / n_b
        )
        z_score = stats.norm.ppf((1 + config.confidence_level) / 2)
        ci_lower = difference - z_score * se_diff
        ci_upper = difference + z_score * se_diff

        # Determine significance
        is_significant = p_value < (1 - config.confidence_level)

        # Calculate relative improvement
        if metric_a != 0:
            relative_improvement = (metric_b - metric_a) / metric_a
        else:
            relative_improvement = 0

        # Calculate power
        effect_size = abs(difference) / np.sqrt(
            (np.var(test["metrics_a"][config.metric]) + np.var(test["metrics_b"][config.metric]))
            / 2
        )
        power = self._calculate_power(effect_size, n_a, n_b, config.confidence_level)

        # Determine winner
        if is_significant:
            winner = "B" if metric_b > metric_a else "A"
            recommendation = f"Model {winner} performs significantly better"
        else:
            winner = None
            recommendation = "No significant difference detected yet"

        # Calculate secondary metrics
        secondary_a = {}
        secondary_b = {}
        for metric in config.secondary_metrics:
            if metric in test["metrics_a"]:
                secondary_a[metric] = np.mean(test["metrics_a"][metric])
                secondary_b[metric] = np.mean(test["metrics_b"][metric])

        # Create result
        result = ABTestResult(
            test_id=test_id,
            status=test["status"],
            n_samples_a=n_a,
            n_samples_b=n_b,
            metric_a=metric_a,
            metric_b=metric_b,
            metric_difference=difference,
            relative_improvement=relative_improvement,
            p_value=p_value,
            confidence_interval=(ci_lower, ci_upper),
            is_significant=is_significant,
            statistical_power=power,
            secondary_metrics_a=secondary_a,
            secondary_metrics_b=secondary_b,
            start_time=test["start_time"],
            end_time=None,
            duration_hours=(
                (datetime.now() - test["start_time"]).total_seconds() / 3600
                if test["start_time"]
                else 0
            ),
            winner=winner,
            recommendation=recommendation,
        )

        return result

    def stop_test(self, test_id: str) -> ABTestResult:
        """
        Stop an A/B test and get final results.

        Args:
            test_id: Test identifier

        Returns:
            Final test results
        """
        if test_id not in self.active_tests:
            raise ValueError(f"Test {test_id} not found")

        # Get final results
        result = self.get_current_results(test_id)
        if result:
            result.status = TestStatus.COMPLETED
            result.end_time = datetime.now()

        # Move to completed
        test = self.active_tests[test_id]
        test["status"] = TestStatus.COMPLETED

        if result:
            self.completed_tests.append(result)

        # Clean up
        del self.active_tests[test_id]

        logger.info(f"Stopped A/B test {test_id}. Winner: {result.winner if result else 'N/A'}")

        return result

    def _calculate_sample_size(
        self, effect_size: float, confidence_level: float, power: float
    ) -> int:
        """
        Calculate required sample size for statistical significance.

        Args:
            effect_size: Minimum detectable effect
            confidence_level: Statistical confidence level
            power: Statistical power

        Returns:
            Required sample size per variant
        """
        # Standard sample size calculation for two-sample t-test
        alpha = 1 - confidence_level
        beta = 1 - power

        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(1 - beta)

        # Assume standard deviation of 0.5 for proportions
        std_dev = 0.5

        n = 2 * ((z_alpha + z_beta) ** 2) * (std_dev**2) / (effect_size**2)

        return max(100, int(np.ceil(n)))

    def _calculate_power(
        self, effect_size: float, n_a: int, n_b: int, confidence_level: float
    ) -> float:
        """Calculate statistical power of test"""
        alpha = 1 - confidence_level
        z_alpha = stats.norm.ppf(1 - alpha / 2)

        # Harmonic mean of sample sizes
        n_harm = 2 * n_a * n_b / (n_a + n_b)

        # Calculate power
        z = effect_size * np.sqrt(n_harm / 2)
        power = stats.norm.cdf(z - z_alpha) + stats.norm.cdf(-z - z_alpha)

        return min(0.999, max(0.001, power))

    def _check_early_stopping(self, test_id: str) -> bool:
        """
        Check if test should be stopped early.

        Returns:
            True if test should stop
        """
        test = self.active_tests[test_id]
        config = test["config"]

        n_a = len(test["metrics_a"][config.metric])
        n_b = len(test["metrics_b"][config.metric])

        # Need minimum samples
        if n_a < 50 or n_b < 50:
            return False

        # Sequential probability ratio test (SPRT)
        if config.sequential_testing:
            # Calculate log likelihood ratio
            metric_a = test["metrics_a"][config.metric]
            metric_b = test["metrics_b"][config.metric]

            # Simple SPRT for normal distributions
            mean_a = np.mean(metric_a)
            mean_b = np.mean(metric_b)
            var_a = np.var(metric_a)
            var_b = np.var(metric_b)

            if var_a > 0 and var_b > 0:
                # Log likelihood ratio
                llr = sum(
                    np.log(
                        stats.norm.pdf(x, mean_b, np.sqrt(var_b))
                        / stats.norm.pdf(x, mean_a, np.sqrt(var_a))
                    )
                    for x in metric_a[-10:]
                )

                # Thresholds
                alpha = 1 - config.confidence_level
                beta = 1 - config.power
                upper_threshold = np.log((1 - beta) / alpha)
                lower_threshold = np.log(beta / (1 - alpha))

                if llr >= upper_threshold or llr <= lower_threshold:
                    logger.info(f"Early stopping triggered for test {test_id}")
                    self.stop_test(test_id)
                    return True

        # Check if we've reached max samples
        if n_a >= config.max_samples_per_variant or n_b >= config.max_samples_per_variant:
            logger.info(f"Max samples reached for test {test_id}")
            self.stop_test(test_id)
            return True

        return False

    def _thompson_sampling_assignment(self, test: dict, config: ABTestConfig) -> str:
        """Thompson sampling for variant assignment"""
        # Use Beta distribution for binary outcomes
        # Alpha = successes + 1, Beta = failures + 1

        metrics_a = test["metrics_a"][config.metric]
        metrics_b = test["metrics_b"][config.metric]

        if len(metrics_a) > 0:
            alpha_a = sum(metrics_a) + 1
            beta_a = len(metrics_a) - sum(metrics_a) + 1
        else:
            alpha_a = 1
            beta_a = 1

        if len(metrics_b) > 0:
            alpha_b = sum(metrics_b) + 1
            beta_b = len(metrics_b) - sum(metrics_b) + 1
        else:
            alpha_b = 1
            beta_b = 1

        # Sample from Beta distributions
        sample_a = np.random.beta(alpha_a, beta_a)
        sample_b = np.random.beta(alpha_b, beta_b)

        return "A" if sample_a > sample_b else "B"

    def _ucb_assignment(self, test: dict, config: ABTestConfig) -> str:
        """Upper confidence bound for variant assignment"""
        metrics_a = test["metrics_a"][config.metric]
        metrics_b = test["metrics_b"][config.metric]

        n_a = len(metrics_a)
        n_b = len(metrics_b)

        if n_a == 0:
            return "A"
        if n_b == 0:
            return "B"

        # Calculate UCB
        mean_a = np.mean(metrics_a)
        mean_b = np.mean(metrics_b)

        total_n = n_a + n_b
        ucb_a = mean_a + np.sqrt(2 * np.log(total_n) / n_a)
        ucb_b = mean_b + np.sqrt(2 * np.log(total_n) / n_b)

        return "A" if ucb_a > ucb_b else "B"

    def get_test_summary(self, test_id: str) -> dict[str, Any]:
        """Get summary of test status and metrics"""
        if test_id in self.active_tests:
            test = self.active_tests[test_id]
            result = self.get_current_results(test_id)

            summary = {
                "test_id": test_id,
                "status": test["status"].value,
                "models": {"A": test["config"].model_a_id, "B": test["config"].model_b_id},
                "samples": {
                    "A": len(test["metrics_a"][test["config"].metric]),
                    "B": len(test["metrics_b"][test["config"].metric]),
                },
                "required_samples": test["required_samples"],
            }

            if result:
                summary["current_results"] = {
                    "p_value": result.p_value,
                    "is_significant": result.is_significant,
                    "winner": result.winner,
                    "improvement": f"{result.relative_improvement:.2%}",
                }

            return summary

        return {"error": f"Test {test_id} not found"}

    def save_results(self, filepath: str) -> None:
        """Save test results to file"""
        results = {
            "active_tests": [self.get_test_summary(test_id) for test_id in self.active_tests],
            "completed_tests": [result.to_dict() for result in self.completed_tests],
            "timestamp": datetime.now().isoformat(),
        }

        with open(filepath, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Saved A/B test results to {filepath}")


def create_ab_test(
    model_a_id: str, model_b_id: str, metric: str = "accuracy", **kwargs
) -> tuple[ABTestingFramework, str]:
    """
    Convenience function to create and start an A/B test.

    Args:
        model_a_id: ID of model A
        model_b_id: ID of model B
        metric: Primary metric to optimize
        **kwargs: Additional configuration parameters

    Returns:
        Tuple of (framework, test_id)
    """
    framework = ABTestingFramework()

    config = ABTestConfig(
        test_id=f"test_{model_a_id}_vs_{model_b_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        model_a_id=model_a_id,
        model_b_id=model_b_id,
        metric=metric,
        **kwargs,
    )

    test_id = framework.create_test(config)
    framework.start_test(test_id)

    return framework, test_id


if __name__ == "__main__":
    # Example usage
    framework = ABTestingFramework()

    # Create test configuration
    config = ABTestConfig(
        test_id="test_001",
        model_a_id="model_v1",
        model_b_id="model_v2",
        metric="accuracy",
        allocation_strategy=AllocationStrategy.THOMPSON_SAMPLING,
        enable_early_stopping=True,
    )

    # Create and start test
    test_id = framework.create_test(config)
    framework.start_test(test_id)

    # Simulate some predictions
    np.random.seed(42)
    for i in range(200):
        # Assign variant
        variant = framework.assign_variant(test_id)

        # Simulate prediction (Model B slightly better)
        if variant == "A":
            accuracy = 0.70
        else:
            accuracy = 0.73

        prediction = 1 if np.random.random() < accuracy else 0
        actual = 1 if np.random.random() < 0.5 else 0

        # Record outcome
        framework.record_outcome(test_id, variant, prediction, actual)

        # Check results periodically
        if i % 50 == 49:
            result = framework.get_current_results(test_id)
            if result:
                print(f"\nAfter {i+1} samples:")
                print(f"  Model A: {result.metric_a:.3f} (n={result.n_samples_a})")
                print(f"  Model B: {result.metric_b:.3f} (n={result.n_samples_b})")
                print(f"  P-value: {result.p_value:.4f}")
                print(f"  Significant: {result.is_significant}")

    # Get final results
    final_result = framework.stop_test(test_id)
    print("\nFinal Results:")
    print(f"  Winner: {final_result.winner}")
    print(f"  Recommendation: {final_result.recommendation}")
    print(f"  Relative improvement: {final_result.relative_improvement:.2%}")
