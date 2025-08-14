"""
Enhanced Statistical Analysis Framework
Phase 3, Week 2: MON-011
Comprehensive statistical testing for model comparison and validation
"""

import logging
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class TestType(Enum):
    """Types of statistical tests"""

    T_TEST = "t_test"
    MANN_WHITNEY = "mann_whitney"
    WILCOXON = "wilcoxon"
    CHI_SQUARE = "chi_square"
    ANOVA = "anova"
    KRUSKAL_WALLIS = "kruskal_wallis"
    KOLMOGOROV_SMIRNOV = "kolmogorov_smirnov"
    ANDERSON_DARLING = "anderson_darling"
    SHAPIRO_WILK = "shapiro_wilk"
    LEVENE = "levene"
    BOOTSTRAP = "bootstrap"
    PERMUTATION = "permutation"


class CorrectionMethod(Enum):
    """Multiple comparison correction methods"""

    NONE = "none"
    BONFERRONI = "bonferroni"
    HOLM = "holm"
    BENJAMINI_HOCHBERG = "benjamini_hochberg"
    BENJAMINI_YEKUTIELI = "benjamini_yekutieli"


@dataclass
class StatisticalTestResult:
    """Result from a statistical test"""

    test_type: TestType
    statistic: float
    p_value: float
    is_significant: bool
    confidence_level: float

    # Effect size metrics
    effect_size: float | None = None
    effect_size_interpretation: str | None = None

    # Confidence intervals
    confidence_interval: tuple[float, float] | None = None

    # Power analysis
    statistical_power: float | None = None
    required_sample_size: int | None = None

    # Additional metrics
    degrees_of_freedom: float | None = None
    sample_sizes: dict[str, int] | None = None

    # Interpretation
    interpretation: str = ""
    recommendation: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "test_type": self.test_type.value,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "is_significant": self.is_significant,
            "confidence_level": self.confidence_level,
            "effect_size": self.effect_size,
            "effect_size_interpretation": self.effect_size_interpretation,
            "confidence_interval": self.confidence_interval,
            "statistical_power": self.statistical_power,
            "interpretation": self.interpretation,
            "recommendation": self.recommendation,
        }


@dataclass
class MultipleComparisonResult:
    """Result from multiple comparison tests"""

    original_p_values: list[float]
    corrected_p_values: list[float]
    correction_method: CorrectionMethod
    significant_tests: list[int]  # Indices of significant tests
    family_wise_error_rate: float
    false_discovery_rate: float | None = None


class StatisticalAnalyzer:
    """
    Enhanced statistical analysis framework for rigorous model comparison.

    Features:
    - Multiple test types (parametric and non-parametric)
    - Effect size calculations
    - Power analysis
    - Multiple comparison corrections
    - Bootstrap confidence intervals
    - Permutation testing
    """

    def __init__(self, confidence_level: float = 0.95, min_sample_size: int = 30):
        """
        Initialize statistical analyzer.

        Args:
            confidence_level: Confidence level for tests
            min_sample_size: Minimum sample size for valid tests
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self.min_sample_size = min_sample_size

        # Test history
        self.test_history: list[StatisticalTestResult] = []

        logger.info(f"StatisticalAnalyzer initialized with {confidence_level:.0%} confidence")

    def compare_two_samples(
        self,
        sample_a: np.ndarray,
        sample_b: np.ndarray,
        test_type: TestType | None = None,
        paired: bool = False,
    ) -> StatisticalTestResult:
        """
        Compare two samples using appropriate statistical test.

        Args:
            sample_a: First sample
            sample_b: Second sample
            test_type: Specific test to use (auto-select if None)
            paired: Whether samples are paired

        Returns:
            Statistical test result
        """
        # Validate samples
        if len(sample_a) < self.min_sample_size or len(sample_b) < self.min_sample_size:
            logger.warning(f"Sample sizes too small: {len(sample_a)}, {len(sample_b)}")

        # Auto-select test if not specified
        if test_type is None:
            test_type = self._select_appropriate_test(sample_a, sample_b, paired)

        # Perform test
        if test_type == TestType.T_TEST:
            result = self._t_test(sample_a, sample_b, paired)
        elif test_type == TestType.MANN_WHITNEY:
            result = self._mann_whitney_test(sample_a, sample_b)
        elif test_type == TestType.WILCOXON:
            result = self._wilcoxon_test(sample_a, sample_b)
        elif test_type == TestType.CHI_SQUARE:
            # Determine if data is categorical
            unique_vals = np.unique(np.concatenate([sample_a, sample_b]))
            is_categorical = not set(unique_vals).issubset({0, 1}) or len(unique_vals) > 2
            result = self._chi_square_test(sample_a, sample_b, categorical=is_categorical)
        elif test_type == TestType.BOOTSTRAP:
            result = self._bootstrap_test(sample_a, sample_b)
        elif test_type == TestType.PERMUTATION:
            result = self._permutation_test(sample_a, sample_b)
        else:
            # Default to t-test
            result = self._t_test(sample_a, sample_b, paired)

        # Calculate effect size
        result.effect_size = self._calculate_effect_size(sample_a, sample_b, test_type)
        result.effect_size_interpretation = self._interpret_effect_size(result.effect_size)

        # Calculate statistical power
        result.statistical_power = self._calculate_power(
            result.effect_size, len(sample_a), len(sample_b)
        )

        # Add interpretation
        result.interpretation = self._interpret_result(result)
        result.recommendation = self._make_recommendation(result)

        # Store in history
        self.test_history.append(result)

        return result

    def _select_appropriate_test(
        self, sample_a: np.ndarray, sample_b: np.ndarray, paired: bool
    ) -> TestType:
        """Select appropriate test based on data characteristics"""
        # Check if data is binary or categorical
        unique_vals = np.unique(np.concatenate([sample_a, sample_b]))

        # If data is binary (0/1) or has few unique values, use chi-square
        if set(unique_vals).issubset({0, 1}):
            return TestType.CHI_SQUARE  # Binary data
        elif len(unique_vals) <= 10:  # Arbitrary threshold for categorical
            return TestType.CHI_SQUARE  # Categorical data

        # For continuous data, proceed with normality tests
        # Test for normality
        _, p_a = stats.shapiro(sample_a[: min(5000, len(sample_a))])
        _, p_b = stats.shapiro(sample_b[: min(5000, len(sample_b))])

        both_normal = p_a > 0.05 and p_b > 0.05

        # Test for equal variances
        _, p_var = stats.levene(sample_a, sample_b)
        equal_var = p_var > 0.05

        # Select test
        if both_normal and equal_var:
            return TestType.T_TEST
        elif both_normal and not equal_var:
            return TestType.T_TEST  # Welch's t-test
        elif paired:
            return TestType.WILCOXON
        else:
            return TestType.MANN_WHITNEY

    def _t_test(
        self, sample_a: np.ndarray, sample_b: np.ndarray, paired: bool
    ) -> StatisticalTestResult:
        """Perform t-test"""
        if paired:
            statistic, p_value = stats.ttest_rel(sample_a, sample_b)
        else:
            # Check for equal variances
            _, p_var = stats.levene(sample_a, sample_b)
            equal_var = p_var > 0.05
            statistic, p_value = stats.ttest_ind(sample_a, sample_b, equal_var=equal_var)

        # Calculate confidence interval
        mean_diff = np.mean(sample_b) - np.mean(sample_a)
        se_diff = np.sqrt(np.var(sample_a) / len(sample_a) + np.var(sample_b) / len(sample_b))
        ci_margin = stats.t.ppf(1 - self.alpha / 2, len(sample_a) + len(sample_b) - 2) * se_diff
        confidence_interval = (mean_diff - ci_margin, mean_diff + ci_margin)

        return StatisticalTestResult(
            test_type=TestType.T_TEST,
            statistic=statistic,
            p_value=p_value,
            is_significant=p_value < self.alpha,
            confidence_level=self.confidence_level,
            confidence_interval=confidence_interval,
            degrees_of_freedom=len(sample_a) + len(sample_b) - 2,
            sample_sizes={"A": len(sample_a), "B": len(sample_b)},
        )

    def _mann_whitney_test(
        self, sample_a: np.ndarray, sample_b: np.ndarray
    ) -> StatisticalTestResult:
        """Perform Mann-Whitney U test"""
        statistic, p_value = stats.mannwhitneyu(sample_a, sample_b, alternative="two-sided")

        # Calculate confidence interval using Hodges-Lehmann estimator
        differences = []
        for a in sample_a:
            for b in sample_b:
                differences.append(b - a)
        differences = np.array(differences)
        ci_lower = np.percentile(differences, (1 - self.confidence_level) / 2 * 100)
        ci_upper = np.percentile(differences, (1 + self.confidence_level) / 2 * 100)

        return StatisticalTestResult(
            test_type=TestType.MANN_WHITNEY,
            statistic=statistic,
            p_value=p_value,
            is_significant=p_value < self.alpha,
            confidence_level=self.confidence_level,
            confidence_interval=(ci_lower, ci_upper),
            sample_sizes={"A": len(sample_a), "B": len(sample_b)},
        )

    def _wilcoxon_test(self, sample_a: np.ndarray, sample_b: np.ndarray) -> StatisticalTestResult:
        """Perform Wilcoxon signed-rank test"""
        if len(sample_a) != len(sample_b):
            raise ValueError("Samples must have same length for Wilcoxon test")

        statistic, p_value = stats.wilcoxon(sample_a, sample_b)

        # Calculate confidence interval
        differences = sample_b - sample_a
        ci_lower = np.percentile(differences, (1 - self.confidence_level) / 2 * 100)
        ci_upper = np.percentile(differences, (1 + self.confidence_level) / 2 * 100)

        return StatisticalTestResult(
            test_type=TestType.WILCOXON,
            statistic=statistic,
            p_value=p_value,
            is_significant=p_value < self.alpha,
            confidence_level=self.confidence_level,
            confidence_interval=(ci_lower, ci_upper),
            sample_sizes={"pairs": len(sample_a)},
        )

    def _bootstrap_test(
        self, sample_a: np.ndarray, sample_b: np.ndarray, n_bootstrap: int = 10000
    ) -> StatisticalTestResult:
        """Perform bootstrap hypothesis test"""
        observed_diff = np.mean(sample_b) - np.mean(sample_a)

        # Combine samples for permutation under null hypothesis
        combined = np.concatenate([sample_a, sample_b])
        n_a = len(sample_a)

        # Bootstrap resampling
        bootstrap_diffs = []
        for _ in range(n_bootstrap):
            # Resample with replacement
            resampled = np.random.choice(combined, size=len(combined), replace=True)
            sample_a_boot = resampled[:n_a]
            sample_b_boot = resampled[n_a:]
            bootstrap_diffs.append(np.mean(sample_b_boot) - np.mean(sample_a_boot))

        bootstrap_diffs = np.array(bootstrap_diffs)

        # Calculate p-value
        p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff))

        # Bootstrap confidence interval
        ci_lower = np.percentile(bootstrap_diffs, (1 - self.confidence_level) / 2 * 100)
        ci_upper = np.percentile(bootstrap_diffs, (1 + self.confidence_level) / 2 * 100)

        return StatisticalTestResult(
            test_type=TestType.BOOTSTRAP,
            statistic=observed_diff,
            p_value=p_value,
            is_significant=p_value < self.alpha,
            confidence_level=self.confidence_level,
            confidence_interval=(ci_lower, ci_upper),
            sample_sizes={"A": len(sample_a), "B": len(sample_b)},
        )

    def _permutation_test(
        self, sample_a: np.ndarray, sample_b: np.ndarray, n_permutations: int = 10000
    ) -> StatisticalTestResult:
        """Perform permutation test"""
        observed_diff = np.mean(sample_b) - np.mean(sample_a)

        # Combine samples
        combined = np.concatenate([sample_a, sample_b])
        n_a = len(sample_a)

        # Permutation testing
        permutation_diffs = []
        for _ in range(n_permutations):
            # Shuffle and split
            np.random.shuffle(combined)
            perm_a = combined[:n_a]
            perm_b = combined[n_a:]
            permutation_diffs.append(np.mean(perm_b) - np.mean(perm_a))

        permutation_diffs = np.array(permutation_diffs)

        # Calculate p-value
        p_value = np.mean(np.abs(permutation_diffs) >= np.abs(observed_diff))

        # Confidence interval from permutation distribution
        ci_lower = np.percentile(permutation_diffs, (1 - self.confidence_level) / 2 * 100)
        ci_upper = np.percentile(permutation_diffs, (1 + self.confidence_level) / 2 * 100)

        return StatisticalTestResult(
            test_type=TestType.PERMUTATION,
            statistic=observed_diff,
            p_value=p_value,
            is_significant=p_value < self.alpha,
            confidence_level=self.confidence_level,
            confidence_interval=(ci_lower, ci_upper),
            sample_sizes={"A": len(sample_a), "B": len(sample_b)},
        )

    def _chi_square_test(
        self,
        sample_a: np.ndarray | list,
        sample_b: np.ndarray | list,
        categorical: bool = False,
    ) -> StatisticalTestResult:
        """
        Perform chi-square test for categorical or binary outcomes.

        Args:
            sample_a: Data from variant A (binary or categorical)
            sample_b: Data from variant B (binary or categorical)
            categorical: If True, treat as categorical data; if False, treat as binary

        Returns:
            Statistical test result
        """
        sample_a = np.asarray(sample_a)
        sample_b = np.asarray(sample_b)

        if not categorical:
            # Binary outcomes (e.g., conversion/no conversion)
            # Ensure binary data
            unique_vals = np.unique(np.concatenate([sample_a, sample_b]))
            if not set(unique_vals).issubset({0, 1}):
                logger.warning("Data contains non-binary values, treating as categorical")
                categorical = True

        if categorical:
            # Categorical outcomes
            # Get all unique categories
            categories = np.unique(np.concatenate([sample_a, sample_b]))

            # Create contingency table
            contingency_table = np.zeros((2, len(categories)))
            for i, cat in enumerate(categories):
                contingency_table[0, i] = np.sum(sample_a == cat)
                contingency_table[1, i] = np.sum(sample_b == cat)
        else:
            # Binary outcomes - create 2x2 contingency table
            successes_a = np.sum(sample_a == 1)
            failures_a = len(sample_a) - successes_a
            successes_b = np.sum(sample_b == 1)
            failures_b = len(sample_b) - successes_b

            contingency_table = np.array([[successes_a, failures_a], [successes_b, failures_b]])

        # Perform chi-square test
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

        # Calculate Cramér's V as effect size
        n = np.sum(contingency_table)
        min_dim = min(contingency_table.shape[0] - 1, contingency_table.shape[1] - 1)
        cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0

        # Interpret effect size for Cramér's V
        if cramers_v < 0.1:
            effect_interpretation = "negligible"
        elif cramers_v < 0.3:
            effect_interpretation = "small"
        elif cramers_v < 0.5:
            effect_interpretation = "medium"
        else:
            effect_interpretation = "large"

        # For binary data, calculate confidence interval for proportion difference
        if not categorical and contingency_table.shape == (2, 2):
            prop_a = successes_a / len(sample_a)
            prop_b = successes_b / len(sample_b)
            prop_diff = prop_b - prop_a

            # Standard error for proportion difference
            se = np.sqrt(
                prop_a * (1 - prop_a) / len(sample_a) + prop_b * (1 - prop_b) / len(sample_b)
            )
            z_critical = stats.norm.ppf((1 + self.confidence_level) / 2)
            ci_lower = prop_diff - z_critical * se
            ci_upper = prop_diff + z_critical * se
        else:
            ci_lower, ci_upper = None, None

        result = StatisticalTestResult(
            test_type=TestType.CHI_SQUARE,
            statistic=chi2,
            p_value=p_value,
            is_significant=p_value < self.alpha,
            effect_size=cramers_v,
            effect_size_interpretation=effect_interpretation,
            confidence_level=self.confidence_level,
            confidence_interval=(ci_lower, ci_upper) if ci_lower is not None else None,
            degrees_of_freedom=dof,
            sample_sizes={"A": len(sample_a), "B": len(sample_b)},
        )

        # Add specific interpretation for chi-square test
        if not categorical and contingency_table.shape == (2, 2):
            result.interpretation = (
                f"Chi-square test for binary outcomes: "
                f"Variant A success rate: {prop_a:.2%}, "
                f"Variant B success rate: {prop_b:.2%}, "
                f"p-value: {p_value:.4f}"
            )
            if result.is_significant:
                better = "B" if prop_b > prop_a else "A"
                result.recommendation = f"Variant {better} has significantly better success rate"
            else:
                result.recommendation = "No significant difference in success rates"
        else:
            result.interpretation = (
                f"Chi-square test for categorical outcomes: "
                f"χ² = {chi2:.3f}, df = {dof}, p-value: {p_value:.4f}"
            )
            if result.is_significant:
                result.recommendation = "Significant difference in distribution across categories"
            else:
                result.recommendation = "No significant difference in distributions"

        return result

    def _calculate_effect_size(
        self, sample_a: np.ndarray, sample_b: np.ndarray, test_type: TestType
    ) -> float:
        """Calculate effect size for the test"""
        if test_type in [TestType.T_TEST, TestType.BOOTSTRAP, TestType.PERMUTATION]:
            # Cohen's d
            mean_diff = np.mean(sample_b) - np.mean(sample_a)
            pooled_std = np.sqrt((np.var(sample_a) + np.var(sample_b)) / 2)
            if pooled_std > 0:
                return mean_diff / pooled_std
            return 0

        elif test_type == TestType.MANN_WHITNEY:
            # Rank biserial correlation
            U, _ = stats.mannwhitneyu(sample_a, sample_b)
            n_a, n_b = len(sample_a), len(sample_b)
            return 1 - (2 * U) / (n_a * n_b)

        elif test_type == TestType.WILCOXON:
            # Matched pairs rank biserial correlation
            differences = sample_b - sample_a
            positive_ranks = sum(stats.rankdata(np.abs(differences))[differences > 0])
            negative_ranks = sum(stats.rankdata(np.abs(differences))[differences < 0])
            n = len(differences)
            return (positive_ranks - negative_ranks) / (n * (n + 1) / 2)

        return 0

    def _interpret_effect_size(self, effect_size: float) -> str:
        """Interpret effect size magnitude"""
        abs_effect = abs(effect_size)

        if abs_effect < 0.2:
            return "negligible"
        elif abs_effect < 0.5:
            return "small"
        elif abs_effect < 0.8:
            return "medium"
        else:
            return "large"

    def _calculate_power(self, effect_size: float, n_a: int, n_b: int) -> float:
        """Calculate statistical power"""
        from statsmodels.stats.power import ttest_power

        try:
            # Calculate power for two-sample t-test
            power = ttest_power(effect_size, n_a, self.alpha, alternative="two-sided")
            return min(0.999, max(0.001, power))
        except:
            # Fallback calculation
            z_alpha = stats.norm.ppf(1 - self.alpha / 2)
            n_harm = 2 * n_a * n_b / (n_a + n_b)
            z = effect_size * np.sqrt(n_harm / 2)
            power = stats.norm.cdf(z - z_alpha) + stats.norm.cdf(-z - z_alpha)
            return min(0.999, max(0.001, power))

    def _interpret_result(self, result: StatisticalTestResult) -> str:
        """Generate interpretation of test result"""
        if result.is_significant:
            interpretation = (
                f"The test shows a statistically significant difference "
                f"(p={result.p_value:.4f} < {self.alpha:.3f}) with "
                f"{result.effect_size_interpretation} effect size "
                f"(d={result.effect_size:.3f}). "
            )

            if result.statistical_power:
                interpretation += f"Statistical power: {result.statistical_power:.2f}. "

            if result.confidence_interval:
                interpretation += (
                    f"The {self.confidence_level:.0%} CI "
                    f"[{result.confidence_interval[0]:.3f}, "
                    f"{result.confidence_interval[1]:.3f}] "
                    f"does not include zero."
                )
        else:
            interpretation = (
                f"No statistically significant difference detected "
                f"(p={result.p_value:.4f} >= {self.alpha:.3f}). "
            )

            if result.effect_size:
                interpretation += (
                    f"Effect size is {result.effect_size_interpretation} "
                    f"(d={result.effect_size:.3f}). "
                )

            if result.statistical_power and result.statistical_power < 0.8:
                interpretation += (
                    f"Low statistical power ({result.statistical_power:.2f}) "
                    f"suggests more samples may be needed."
                )

        return interpretation

    def _make_recommendation(self, result: StatisticalTestResult) -> str:
        """Make recommendation based on test result"""
        if result.is_significant:
            if abs(result.effect_size) >= 0.5:  # Medium or large effect
                return (
                    "Strong evidence for difference. Consider adopting the better performing model."
                )
            else:  # Small effect
                return (
                    "Statistically significant but small effect. Consider practical significance."
                )
        else:
            if result.statistical_power < 0.8:
                required_n = self.calculate_required_sample_size(
                    effect_size=0.5,
                    power=0.8,  # Medium effect
                )
                return f"Insufficient evidence. Collect more data (n≥{required_n} per group)."
            else:
                return "No meaningful difference detected. Models perform equivalently."

    def compare_multiple_samples(
        self,
        samples: list[np.ndarray],
        labels: list[str] | None = None,
        correction_method: CorrectionMethod = CorrectionMethod.BENJAMINI_HOCHBERG,
    ) -> tuple[list[StatisticalTestResult], MultipleComparisonResult]:
        """
        Compare multiple samples with correction for multiple comparisons.

        Args:
            samples: List of samples to compare
            labels: Optional labels for samples
            correction_method: Method for multiple comparison correction

        Returns:
            Individual test results and multiple comparison result
        """
        if labels is None:
            labels = [f"Sample_{i}" for i in range(len(samples))]

        # Perform pairwise comparisons
        results = []
        p_values = []

        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                result = self.compare_two_samples(samples[i], samples[j])
                results.append((labels[i], labels[j], result))
                p_values.append(result.p_value)

        # Apply multiple comparison correction
        corrected_p_values = self._apply_correction(p_values, correction_method)

        # Identify significant tests after correction
        significant_tests = [i for i, p in enumerate(corrected_p_values) if p < self.alpha]

        # Calculate error rates
        family_wise_error_rate = 1 - (1 - self.alpha) ** len(p_values)

        # False discovery rate (for Benjamini methods)
        if correction_method in [
            CorrectionMethod.BENJAMINI_HOCHBERG,
            CorrectionMethod.BENJAMINI_YEKUTIELI,
        ]:
            if significant_tests:
                false_discovery_rate = self.alpha * len(significant_tests) / len(p_values)
            else:
                false_discovery_rate = 0
        else:
            false_discovery_rate = None

        mc_result = MultipleComparisonResult(
            original_p_values=p_values,
            corrected_p_values=corrected_p_values,
            correction_method=correction_method,
            significant_tests=significant_tests,
            family_wise_error_rate=family_wise_error_rate,
            false_discovery_rate=false_discovery_rate,
        )

        return results, mc_result

    def _apply_correction(self, p_values: list[float], method: CorrectionMethod) -> list[float]:
        """Apply multiple comparison correction"""
        p_values = np.array(p_values)

        if method == CorrectionMethod.NONE:
            return p_values.tolist()

        elif method == CorrectionMethod.BONFERRONI:
            return (p_values * len(p_values)).tolist()

        elif method == CorrectionMethod.HOLM:
            # Holm-Bonferroni method
            n = len(p_values)
            sorted_indices = np.argsort(p_values)
            sorted_p = p_values[sorted_indices]

            corrected = np.zeros_like(p_values)
            for i, idx in enumerate(sorted_indices):
                corrected[idx] = min(1, sorted_p[i] * (n - i))

            return corrected.tolist()

        elif method == CorrectionMethod.BENJAMINI_HOCHBERG:
            # Benjamini-Hochberg FDR control
            from statsmodels.stats.multitest import multipletests

            _, corrected, _, _ = multipletests(p_values, method="fdr_bh", alpha=self.alpha)
            return corrected.tolist()

        elif method == CorrectionMethod.BENJAMINI_YEKUTIELI:
            # Benjamini-Yekutieli FDR control
            from statsmodels.stats.multitest import multipletests

            _, corrected, _, _ = multipletests(p_values, method="fdr_by", alpha=self.alpha)
            return corrected.tolist()

        return p_values.tolist()

    def calculate_required_sample_size(
        self, effect_size: float = 0.5, power: float = 0.8, ratio: float = 1.0
    ) -> int:
        """
        Calculate required sample size for desired power.

        Args:
            effect_size: Expected effect size (Cohen's d)
            power: Desired statistical power
            ratio: Ratio of sample sizes (n_b/n_a)

        Returns:
            Required sample size per group
        """
        from statsmodels.stats.power import tt_solve_power

        try:
            n = tt_solve_power(
                effect_size=effect_size,
                alpha=self.alpha,
                power=power,
                ratio=ratio,
                alternative="two-sided",
            )
            return max(self.min_sample_size, int(np.ceil(n)))
        except:
            # Fallback calculation
            z_alpha = stats.norm.ppf(1 - self.alpha / 2)
            z_beta = stats.norm.ppf(power)

            n = 2 * ((z_alpha + z_beta) ** 2) / (effect_size**2)
            return max(self.min_sample_size, int(np.ceil(n)))

    def test_normality(self, sample: np.ndarray) -> dict[str, Any]:
        """
        Test if sample follows normal distribution.

        Args:
            sample: Data sample

        Returns:
            Dictionary with multiple normality test results
        """
        results = {}

        # Shapiro-Wilk test (best for small samples)
        if len(sample) <= 5000:
            stat_sw, p_sw = stats.shapiro(sample)
            results["shapiro_wilk"] = {
                "statistic": stat_sw,
                "p_value": p_sw,
                "is_normal": p_sw > self.alpha,
            }

        # Anderson-Darling test
        result_ad = stats.anderson(sample, dist="norm")
        results["anderson_darling"] = {
            "statistic": result_ad.statistic,
            "critical_values": result_ad.critical_values.tolist(),
            "significance_levels": result_ad.significance_level.tolist(),
            "is_normal": result_ad.statistic < result_ad.critical_values[2],  # 5% level
        }

        # Kolmogorov-Smirnov test
        stat_ks, p_ks = stats.kstest(sample, "norm", args=(np.mean(sample), np.std(sample)))
        results["kolmogorov_smirnov"] = {
            "statistic": stat_ks,
            "p_value": p_ks,
            "is_normal": p_ks > self.alpha,
        }

        # Overall assessment
        normal_count = sum(1 for test in results.values() if test.get("is_normal", False))
        results["overall_normal"] = normal_count >= 2  # At least 2 tests agree

        return results

    def test_variance_equality(self, *samples) -> StatisticalTestResult:
        """
        Test equality of variances across samples.

        Args:
            *samples: Variable number of samples

        Returns:
            Test result for variance equality
        """
        # Levene's test (robust to non-normality)
        statistic, p_value = stats.levene(*samples)

        return StatisticalTestResult(
            test_type=TestType.LEVENE,
            statistic=statistic,
            p_value=p_value,
            is_significant=p_value < self.alpha,
            confidence_level=self.confidence_level,
            interpretation=f"Variances are {'not ' if p_value < self.alpha else ''}equal",
            recommendation=(
                "Use Welch's t-test" if p_value < self.alpha else "Standard t-test appropriate"
            ),
        )

    def get_summary_statistics(self, sample: np.ndarray) -> dict[str, float]:
        """Get comprehensive summary statistics"""
        return {
            "mean": np.mean(sample),
            "median": np.median(sample),
            "std": np.std(sample),
            "var": np.var(sample),
            "min": np.min(sample),
            "max": np.max(sample),
            "q1": np.percentile(sample, 25),
            "q3": np.percentile(sample, 75),
            "iqr": np.percentile(sample, 75) - np.percentile(sample, 25),
            "skewness": stats.skew(sample),
            "kurtosis": stats.kurtosis(sample),
            "n": len(sample),
        }


def create_statistical_analyzer(confidence_level: float = 0.95) -> StatisticalAnalyzer:
    """Create and initialize a statistical analyzer"""
    return StatisticalAnalyzer(confidence_level=confidence_level)


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)

    # Create analyzer
    analyzer = create_statistical_analyzer(confidence_level=0.95)

    # Generate sample data
    # Model A: baseline performance
    model_a_accuracy = np.random.normal(0.70, 0.05, 100)

    # Model B: slightly better
    model_b_accuracy = np.random.normal(0.73, 0.05, 100)

    # Model C: same as A
    model_c_accuracy = np.random.normal(0.70, 0.05, 100)

    print("=" * 60)
    print("Statistical Analysis Example")
    print("=" * 60)

    # Test normality
    print("\n1. Normality Tests for Model A:")
    normality = analyzer.test_normality(model_a_accuracy)
    for test, result in normality.items():
        if isinstance(result, dict):
            print(f"   {test}: Normal={result.get('is_normal', 'N/A')}")

    # Compare two models
    print("\n2. Comparing Model A vs Model B:")
    result = analyzer.compare_two_samples(model_a_accuracy, model_b_accuracy)
    print(f"   Test: {result.test_type.value}")
    print(f"   P-value: {result.p_value:.4f}")
    print(f"   Significant: {result.is_significant}")
    print(f"   Effect size: {result.effect_size:.3f} ({result.effect_size_interpretation})")
    print(f"   Power: {result.statistical_power:.3f}")
    print(f"   Interpretation: {result.interpretation}")
    print(f"   Recommendation: {result.recommendation}")

    # Multiple comparisons
    print("\n3. Multiple Model Comparison:")
    samples = [model_a_accuracy, model_b_accuracy, model_c_accuracy]
    labels = ["Model A", "Model B", "Model C"]

    pairwise_results, mc_result = analyzer.compare_multiple_samples(
        samples, labels, CorrectionMethod.BENJAMINI_HOCHBERG
    )

    print(f"   Number of comparisons: {len(pairwise_results)}")
    print(f"   Correction method: {mc_result.correction_method.value}")
    print(f"   Family-wise error rate: {mc_result.family_wise_error_rate:.3f}")

    for i, (label1, label2, result) in enumerate(pairwise_results):
        print(f"   {label1} vs {label2}:")
        print(f"      Original p-value: {result.p_value:.4f}")
        print(f"      Corrected p-value: {mc_result.corrected_p_values[i]:.4f}")
        print(f"      Significant after correction: {i in mc_result.significant_tests}")

    # Sample size calculation
    print("\n4. Sample Size Calculation:")
    required_n = analyzer.calculate_required_sample_size(
        effect_size=0.5,
        power=0.8,  # Medium effect
    )
    print(f"   Required sample size for medium effect (d=0.5): {required_n} per group")

    # Bootstrap test
    print("\n5. Bootstrap Test (Model A vs B):")
    bootstrap_result = analyzer.compare_two_samples(
        model_a_accuracy, model_b_accuracy, test_type=TestType.BOOTSTRAP
    )
    print(f"   P-value: {bootstrap_result.p_value:.4f}")
    print(
        f"   CI: [{bootstrap_result.confidence_interval[0]:.4f}, "
        f"{bootstrap_result.confidence_interval[1]:.4f}]"
    )
