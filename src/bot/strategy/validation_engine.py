"""
Strategy Validation Engine for GPT-Trader

Provides comprehensive risk-adjusted performance evaluation, statistical significance testing,
and validation criteria for systematic strategy assessment and ranking.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class ValidationCriteria(Enum):
    """Validation criteria for strategy assessment"""

    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    CALMAR_RATIO = "calmar_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    TAIL_RATIO = "tail_ratio"
    STATISTICAL_SIGNIFICANCE = "statistical_significance"
    CONSISTENCY = "consistency"
    RISK_ADJUSTED_RETURN = "risk_adjusted_return"


class SignificanceTest(Enum):
    """Statistical significance tests"""

    T_TEST = "t_test"
    BOOTSTRAP = "bootstrap"
    PERMUTATION = "permutation"
    SHARPE_RATIO_TEST = "sharpe_ratio_test"


class RiskMetric(Enum):
    """Risk metrics for evaluation"""

    VALUE_AT_RISK = "value_at_risk"
    CONDITIONAL_VAR = "conditional_var"
    MAXIMUM_DRAWDOWN = "maximum_drawdown"
    ULCER_INDEX = "ulcer_index"
    DOWNSIDE_DEVIATION = "downside_deviation"
    BETA = "beta"
    TRACKING_ERROR = "tracking_error"


@dataclass
class ValidationThresholds:
    """Thresholds for strategy validation"""

    # Performance thresholds
    min_sharpe_ratio: float = 0.5
    min_sortino_ratio: float = 0.7
    min_calmar_ratio: float = 0.3
    max_drawdown: float = 0.15
    min_win_rate: float = 0.45
    min_profit_factor: float = 1.2

    # Risk thresholds
    max_var_95: float = 0.03
    max_cvar_95: float = 0.05
    max_ulcer_index: float = 0.10
    max_downside_deviation: float = 0.15

    # Statistical significance
    min_confidence_level: float = 0.95
    min_sample_size: int = 250
    max_p_value: float = 0.05

    # Consistency requirements
    min_monthly_win_rate: float = 0.4
    max_monthly_volatility: float = 0.25
    min_quarterly_consistency: float = 0.6

    # Trading activity
    min_trades_per_year: int = 12
    max_trades_per_year: int = 500
    min_avg_trade_duration: int = 1  # days
    max_avg_trade_duration: int = 90  # days

    # Implementation feasibility
    max_turnover_annual: float = 5.0
    min_capacity_millions: float = 1.0
    max_transaction_cost_impact: float = 0.02


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""

    # Basic metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Risk metrics
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    value_at_risk_95: float = 0.0
    conditional_var_95: float = 0.0
    ulcer_index: float = 0.0
    downside_deviation: float = 0.0

    # Trading metrics
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_trade_duration: float = 0.0

    # Consistency metrics
    monthly_win_rate: float = 0.0
    quarterly_win_rate: float = 0.0
    annual_win_rate: float = 0.0
    best_month: float = 0.0
    worst_month: float = 0.0
    positive_months: int = 0
    negative_months: int = 0

    # Statistical metrics
    skewness: float = 0.0
    kurtosis: float = 0.0
    tail_ratio: float = 0.0
    expectancy: float = 0.0
    kelly_criterion: float = 0.0

    # Implementation metrics
    annual_turnover: float = 0.0
    estimated_capacity: float = 0.0
    transaction_cost_impact: float = 0.0


@dataclass
class SignificanceResult:
    """Statistical significance test result"""

    test_type: SignificanceTest
    statistic: float
    p_value: float
    confidence_interval: tuple[float, float]
    is_significant: bool
    effect_size: float
    sample_size: int
    interpretation: str


@dataclass
class ValidationResult:
    """Comprehensive validation result"""

    strategy_id: str
    validation_timestamp: datetime
    performance_metrics: PerformanceMetrics
    risk_assessment: dict[str, float]
    significance_tests: list[SignificanceResult]

    # Validation outcomes
    meets_thresholds: dict[str, bool] = field(default_factory=dict)
    overall_score: float = 0.0
    risk_score: float = 0.0
    consistency_score: float = 0.0
    significance_score: float = 0.0

    # Final assessment
    is_validated: bool = False
    validation_grade: str = "F"  # A, B, C, D, F
    confidence_level: float = 0.0

    # Detailed analysis
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    # Metadata
    evaluation_period: tuple[datetime, datetime] = field(
        default_factory=lambda: (datetime.min, datetime.min)
    )
    benchmark_comparison: dict[str, float] = field(default_factory=dict)
    regime_analysis: dict[str, dict[str, float]] = field(default_factory=dict)


class PerformanceCalculator:
    """Calculates comprehensive performance metrics"""

    def __init__(self, risk_free_rate: float = 0.02) -> None:
        self.risk_free_rate = risk_free_rate

    def calculate_metrics(
        self,
        returns: pd.Series,
        trades: pd.DataFrame | None = None,
        prices: pd.Series | None = None,
    ) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""

        if len(returns) == 0:
            return PerformanceMetrics()

        # Ensure returns are properly formatted
        returns = returns.dropna()

        metrics = PerformanceMetrics()

        # Basic return metrics
        metrics.total_return = (1 + returns).prod() - 1
        metrics.annualized_return = (1 + returns.mean()) ** 252 - 1
        metrics.volatility = returns.std() * np.sqrt(252)

        # Risk-adjusted ratios
        excess_returns = returns - self.risk_free_rate / 252
        metrics.sharpe_ratio = self._calculate_sharpe(excess_returns)
        metrics.sortino_ratio = self._calculate_sortino(excess_returns)
        metrics.calmar_ratio = self._calculate_calmar(returns)

        # Risk metrics
        metrics.max_drawdown, metrics.max_drawdown_duration = self._calculate_max_drawdown(returns)
        metrics.value_at_risk_95 = self._calculate_var(returns, 0.05)
        metrics.conditional_var_95 = self._calculate_cvar(returns, 0.05)
        metrics.ulcer_index = self._calculate_ulcer_index(returns)
        metrics.downside_deviation = self._calculate_downside_deviation(returns)

        # Statistical metrics
        metrics.skewness = stats.skew(returns.dropna())
        metrics.kurtosis = stats.kurtosis(returns.dropna())
        metrics.tail_ratio = self._calculate_tail_ratio(returns)

        # Trading metrics (if trades data available)
        if trades is not None and not trades.empty:
            metrics = self._calculate_trading_metrics(metrics, trades)
        else:
            # Estimate from returns
            metrics.total_trades = self._estimate_trade_count(returns)
            metrics.win_rate = self._calculate_win_rate(returns)
            metrics.profit_factor = self._calculate_profit_factor(returns)

        # Consistency metrics
        metrics = self._calculate_consistency_metrics(metrics, returns)

        # Implementation metrics
        if prices is not None:
            metrics.annual_turnover = self._estimate_turnover(returns, prices)
            metrics.estimated_capacity = self._estimate_capacity(returns, prices)

        return metrics

    def _calculate_sharpe(self, excess_returns: pd.Series) -> float:
        """Calculate Sharpe ratio"""
        if excess_returns.std() == 0:
            return 0.0
        return excess_returns.mean() / excess_returns.std() * np.sqrt(252)

    def _calculate_sortino(self, excess_returns: pd.Series) -> float:
        """Calculate Sortino ratio"""
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return float("inf") if excess_returns.mean() > 0 else 0.0
        return excess_returns.mean() / downside_returns.std() * np.sqrt(252)

    def _calculate_calmar(self, returns: pd.Series) -> float:
        """Calculate Calmar ratio"""
        max_dd, _ = self._calculate_max_drawdown(returns)
        if max_dd == 0:
            return float("inf") if returns.mean() > 0 else 0.0
        annualized_return = (1 + returns.mean()) ** 252 - 1
        return annualized_return / max_dd

    def _calculate_max_drawdown(self, returns: pd.Series) -> tuple[float, int]:
        """Calculate maximum drawdown and duration"""
        cumulative = (1 + returns).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak

        max_drawdown = drawdown.min()

        # Calculate duration
        drawdown_start = None
        max_duration = 0
        current_duration = 0

        for i, dd in enumerate(drawdown):
            if dd < 0:
                if drawdown_start is None:
                    drawdown_start = i
                current_duration = i - drawdown_start + 1
            else:
                if drawdown_start is not None:
                    max_duration = max(max_duration, current_duration)
                    drawdown_start = None
                    current_duration = 0

        return abs(max_drawdown), max_duration

    def _calculate_var(self, returns: pd.Series, alpha: float) -> float:
        """Calculate Value at Risk"""
        return np.percentile(returns, alpha * 100)

    def _calculate_cvar(self, returns: pd.Series, alpha: float) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        var = self._calculate_var(returns, alpha)
        return returns[returns <= var].mean()

    def _calculate_ulcer_index(self, returns: pd.Series) -> float:
        """Calculate Ulcer Index"""
        cumulative = (1 + returns).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (peak - cumulative) / peak * 100

        return np.sqrt((drawdown**2).mean())

    def _calculate_downside_deviation(self, returns: pd.Series) -> float:
        """Calculate downside deviation"""
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return 0.0
        return downside_returns.std() * np.sqrt(252)

    def _calculate_tail_ratio(self, returns: pd.Series) -> float:
        """Calculate tail ratio (95th percentile / 5th percentile)"""
        p95 = np.percentile(returns, 95)
        p5 = np.percentile(returns, 5)
        if p5 == 0:
            return float("inf") if p95 > 0 else 0.0
        return abs(p95 / p5)

    def _calculate_trading_metrics(
        self, metrics: PerformanceMetrics, trades: pd.DataFrame
    ) -> PerformanceMetrics:
        """Calculate trading-specific metrics"""
        if trades.empty:
            return metrics

        # Assume trades DataFrame has columns: entry_date, exit_date, pnl, quantity
        metrics.total_trades = len(trades)

        if "pnl" in trades.columns:
            pnl = trades["pnl"]
            winning_trades = pnl[pnl > 0]
            losing_trades = pnl[pnl < 0]

            metrics.win_rate = len(winning_trades) / len(trades) if len(trades) > 0 else 0
            metrics.avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0
            metrics.avg_loss = losing_trades.mean() if len(losing_trades) > 0 else 0
            metrics.largest_win = winning_trades.max() if len(winning_trades) > 0 else 0
            metrics.largest_loss = losing_trades.min() if len(losing_trades) > 0 else 0

            # Profit factor
            gross_profit = winning_trades.sum() if len(winning_trades) > 0 else 0
            gross_loss = abs(losing_trades.sum()) if len(losing_trades) > 0 else 0
            metrics.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Trade duration
        if "entry_date" in trades.columns and "exit_date" in trades.columns:
            durations = (trades["exit_date"] - trades["entry_date"]).dt.days
            metrics.avg_trade_duration = durations.mean()

        return metrics

    def _estimate_trade_count(self, returns: pd.Series) -> int:
        """Estimate number of trades from return patterns"""
        # Simple heuristic: count direction changes
        non_zero_returns = returns[returns != 0]
        if len(non_zero_returns) < 2:
            return 0

        signs = np.sign(non_zero_returns)
        changes = (signs != signs.shift()).sum()
        return int(changes / 2)  # Each trade involves entry and exit

    def _calculate_win_rate(self, returns: pd.Series) -> float:
        """Calculate win rate from returns"""
        non_zero_returns = returns[returns != 0]
        if len(non_zero_returns) == 0:
            return 0.0
        return (non_zero_returns > 0).mean()

    def _calculate_profit_factor(self, returns: pd.Series) -> float:
        """Calculate profit factor from returns"""
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]

        gross_profit = positive_returns.sum()
        gross_loss = abs(negative_returns.sum())

        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0

        return gross_profit / gross_loss

    def _calculate_consistency_metrics(
        self, metrics: PerformanceMetrics, returns: pd.Series
    ) -> PerformanceMetrics:
        """Calculate consistency metrics"""

        # Monthly aggregation
        if len(returns) > 30:  # Need sufficient data
            try:
                monthly_returns = returns.resample("M").apply(lambda x: (1 + x).prod() - 1)
                metrics.monthly_win_rate = (monthly_returns > 0).mean()
                metrics.best_month = monthly_returns.max()
                metrics.worst_month = monthly_returns.min()
                metrics.positive_months = (monthly_returns > 0).sum()
                metrics.negative_months = (monthly_returns < 0).sum()
            except Exception:
                pass

        # Quarterly aggregation
        if len(returns) > 90:  # Need sufficient data
            try:
                quarterly_returns = returns.resample("Q").apply(lambda x: (1 + x).prod() - 1)
                metrics.quarterly_win_rate = (quarterly_returns > 0).mean()
            except Exception:
                pass

        return metrics

    def _estimate_turnover(self, returns: pd.Series, prices: pd.Series) -> float:
        """Estimate annual turnover"""
        # Simple estimation based on return volatility and price changes
        if len(returns) == 0 or len(prices) == 0:
            return 0.0

        # Heuristic: turnover roughly proportional to absolute returns
        daily_turnover = abs(returns).mean()
        return daily_turnover * 252

    def _estimate_capacity(self, returns: pd.Series, prices: pd.Series) -> float:
        """Estimate strategy capacity in millions"""
        # Simple heuristic based on return consistency and volatility
        if len(returns) == 0:
            return 0.0

        # Strategies with lower volatility and higher consistency can handle more capital
        vol = returns.std()
        consistency = self._calculate_win_rate(returns)

        # Base capacity estimate (very rough)
        base_capacity = 100  # $100M base
        vol_factor = max(0.1, 1 - vol * 10)  # Penalty for high volatility
        consistency_factor = consistency * 2  # Bonus for consistency

        return base_capacity * vol_factor * consistency_factor


class StatisticalTester:
    """Performs statistical significance tests"""

    def __init__(self, confidence_level: float = 0.95) -> None:
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level

    def test_sharpe_ratio_significance(
        self, returns: pd.Series, benchmark_returns: pd.Series | None = None
    ) -> SignificanceResult:
        """Test statistical significance of Sharpe ratio"""

        if len(returns) < 30:  # Need minimum sample size
            return SignificanceResult(
                test_type=SignificanceTest.SHARPE_RATIO_TEST,
                statistic=0.0,
                p_value=1.0,
                confidence_interval=(0.0, 0.0),
                is_significant=False,
                effect_size=0.0,
                sample_size=len(returns),
                interpretation="Insufficient data for significance testing",
            )

        # Calculate Sharpe ratio
        excess_returns = returns - 0.02 / 252  # Assume 2% risk-free rate
        sharpe = (
            excess_returns.mean() / excess_returns.std() * np.sqrt(252)
            if excess_returns.std() > 0
            else 0
        )

        # Standard error of Sharpe ratio
        n = len(returns)
        se_sharpe = np.sqrt((1 + 0.5 * sharpe**2) / n)

        # t-statistic
        t_stat = sharpe / se_sharpe

        # Two-tailed p-value
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1))

        # Confidence interval
        t_critical = stats.t.ppf(1 - self.alpha / 2, df=n - 1)
        ci_lower = sharpe - t_critical * se_sharpe
        ci_upper = sharpe + t_critical * se_sharpe

        # Interpretation
        if p_value < self.alpha:
            interpretation = (
                f"Sharpe ratio {sharpe:.3f} is statistically significant (p={p_value:.4f})"
            )
        else:
            interpretation = (
                f"Sharpe ratio {sharpe:.3f} is not statistically significant (p={p_value:.4f})"
            )

        return SignificanceResult(
            test_type=SignificanceTest.SHARPE_RATIO_TEST,
            statistic=t_stat,
            p_value=p_value,
            confidence_interval=(ci_lower, ci_upper),
            is_significant=p_value < self.alpha,
            effect_size=sharpe,
            sample_size=n,
            interpretation=interpretation,
        )

    def bootstrap_test(
        self, returns: pd.Series, metric_func: Callable[[pd.Series], float], n_bootstrap: int = 1000
    ) -> SignificanceResult:
        """Bootstrap test for metric significance"""

        if len(returns) < 30:
            return SignificanceResult(
                test_type=SignificanceTest.BOOTSTRAP,
                statistic=0.0,
                p_value=1.0,
                confidence_interval=(0.0, 0.0),
                is_significant=False,
                effect_size=0.0,
                sample_size=len(returns),
                interpretation="Insufficient data for bootstrap testing",
            )

        # Original metric
        original_metric = metric_func(returns)

        # Bootstrap samples
        bootstrap_metrics = []
        for _ in range(n_bootstrap):
            bootstrap_sample = returns.sample(n=len(returns), replace=True)
            bootstrap_metric = metric_func(bootstrap_sample)
            bootstrap_metrics.append(bootstrap_metric)

        bootstrap_metrics = np.array(bootstrap_metrics)

        # P-value (assuming null hypothesis of zero metric)
        if original_metric >= 0:
            p_value = (bootstrap_metrics <= 0).mean()
        else:
            p_value = (bootstrap_metrics >= 0).mean()

        # Confidence interval
        ci_lower = np.percentile(bootstrap_metrics, self.alpha / 2 * 100)
        ci_upper = np.percentile(bootstrap_metrics, (1 - self.alpha / 2) * 100)

        # Statistical significance based on CI not containing zero
        is_significant = not (ci_lower <= 0 <= ci_upper)

        interpretation = (
            f"Bootstrap test: metric={original_metric:.3f}, CI=({ci_lower:.3f}, {ci_upper:.3f})"
        )

        return SignificanceResult(
            test_type=SignificanceTest.BOOTSTRAP,
            statistic=original_metric,
            p_value=p_value * 2,  # Two-tailed
            confidence_interval=(ci_lower, ci_upper),
            is_significant=is_significant,
            effect_size=original_metric,
            sample_size=len(returns),
            interpretation=interpretation,
        )

    def permutation_test(
        self, returns: pd.Series, benchmark_returns: pd.Series, n_permutations: int = 1000
    ) -> SignificanceResult:
        """Permutation test comparing strategy vs benchmark"""

        if len(returns) < 30 or len(benchmark_returns) < 30:
            return SignificanceResult(
                test_type=SignificanceTest.PERMUTATION,
                statistic=0.0,
                p_value=1.0,
                confidence_interval=(0.0, 0.0),
                is_significant=False,
                effect_size=0.0,
                sample_size=min(len(returns), len(benchmark_returns)),
                interpretation="Insufficient data for permutation testing",
            )

        # Align series
        common_index = returns.index.intersection(benchmark_returns.index)
        if len(common_index) < 30:
            return SignificanceResult(
                test_type=SignificanceTest.PERMUTATION,
                statistic=0.0,
                p_value=1.0,
                confidence_interval=(0.0, 0.0),
                is_significant=False,
                effect_size=0.0,
                sample_size=0,
                interpretation="Insufficient common data for comparison",
            )

        returns_aligned = returns.loc[common_index]
        benchmark_aligned = benchmark_returns.loc[common_index]

        # Original difference in means
        original_diff = returns_aligned.mean() - benchmark_aligned.mean()

        # Combined data
        combined = pd.concat([returns_aligned, benchmark_aligned])
        n1, _n2 = len(returns_aligned), len(benchmark_aligned)

        # Permutation test
        permuted_diffs = []
        for _ in range(n_permutations):
            permuted = combined.sample(n=len(combined), replace=False)
            perm_group1 = permuted.iloc[:n1]
            perm_group2 = permuted.iloc[n1:]
            permuted_diff = perm_group1.mean() - perm_group2.mean()
            permuted_diffs.append(permuted_diff)

        permuted_diffs = np.array(permuted_diffs)

        # P-value
        if original_diff >= 0:
            p_value = (permuted_diffs >= original_diff).mean()
        else:
            p_value = (permuted_diffs <= original_diff).mean()

        p_value = p_value * 2  # Two-tailed

        # Confidence interval from permutation distribution
        ci_lower = np.percentile(permuted_diffs, self.alpha / 2 * 100)
        ci_upper = np.percentile(permuted_diffs, (1 - self.alpha / 2) * 100)

        interpretation = f"Strategy outperforms benchmark by {original_diff:.5f} daily return"

        return SignificanceResult(
            test_type=SignificanceTest.PERMUTATION,
            statistic=original_diff,
            p_value=p_value,
            confidence_interval=(ci_lower, ci_upper),
            is_significant=p_value < self.alpha,
            effect_size=original_diff * 252,  # Annualized
            sample_size=len(common_index),
            interpretation=interpretation,
        )


class StrategyValidator:
    """Main strategy validation engine"""

    def __init__(
        self, thresholds: ValidationThresholds | None = None, risk_free_rate: float = 0.02
    ) -> None:
        self.thresholds = thresholds or ValidationThresholds()
        self.performance_calc = PerformanceCalculator(risk_free_rate)
        self.stats_tester = StatisticalTester()

    def validate_strategy(
        self,
        returns: pd.Series,
        strategy_id: str,
        trades: pd.DataFrame | None = None,
        prices: pd.Series | None = None,
        benchmark_returns: pd.Series | None = None,
    ) -> ValidationResult:
        """Comprehensive strategy validation"""

        if len(returns) == 0:
            return ValidationResult(
                strategy_id=strategy_id,
                validation_timestamp=datetime.now(),
                performance_metrics=PerformanceMetrics(),
                risk_assessment={},
                significance_tests=[],
                is_validated=False,
                validation_grade="F",
                confidence_level=0.0,
            )

        logger.info(f"Validating strategy: {strategy_id}")

        # Step 1: Calculate performance metrics
        performance_metrics = self.performance_calc.calculate_metrics(returns, trades, prices)

        # Step 2: Risk assessment
        risk_assessment = self._assess_risk(performance_metrics, returns)

        # Step 3: Statistical significance tests
        significance_tests = self._run_significance_tests(returns, benchmark_returns)

        # Step 4: Threshold validation
        meets_thresholds = self._check_thresholds(performance_metrics)

        # Step 5: Calculate composite scores
        overall_score = self._calculate_overall_score(performance_metrics, meets_thresholds)
        risk_score = self._calculate_risk_score(risk_assessment)
        consistency_score = self._calculate_consistency_score(performance_metrics)
        significance_score = self._calculate_significance_score(significance_tests)

        # Step 6: Final assessment
        is_validated, validation_grade, confidence_level = self._make_final_assessment(
            overall_score, risk_score, consistency_score, significance_score, meets_thresholds
        )

        # Step 7: Generate insights
        strengths, weaknesses, recommendations = self._generate_insights(
            performance_metrics, risk_assessment, meets_thresholds
        )

        # Step 8: Compile result
        result = ValidationResult(
            strategy_id=strategy_id,
            validation_timestamp=datetime.now(),
            performance_metrics=performance_metrics,
            risk_assessment=risk_assessment,
            significance_tests=significance_tests,
            meets_thresholds=meets_thresholds,
            overall_score=overall_score,
            risk_score=risk_score,
            consistency_score=consistency_score,
            significance_score=significance_score,
            is_validated=is_validated,
            validation_grade=validation_grade,
            confidence_level=confidence_level,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations,
            evaluation_period=(returns.index[0].to_pydatetime(), returns.index[-1].to_pydatetime()),
        )

        logger.info(
            f"Validation completed for {strategy_id}: Grade {validation_grade}, Score {overall_score:.2f}"
        )
        return result

    def _assess_risk(self, metrics: PerformanceMetrics, returns: pd.Series) -> dict[str, float]:
        """Assess risk characteristics"""

        return {
            "max_drawdown_severity": min(metrics.max_drawdown / self.thresholds.max_drawdown, 2.0),
            "volatility_level": metrics.volatility / 0.20,  # Normalized to 20% baseline
            "var_risk": abs(metrics.value_at_risk_95) / self.thresholds.max_var_95,
            "tail_risk": metrics.conditional_var_95 / self.thresholds.max_cvar_95,
            "downside_risk": metrics.downside_deviation / self.thresholds.max_downside_deviation,
            "skew_risk": abs(metrics.skewness),  # Negative skew is concerning
            "kurtosis_risk": max(0, metrics.kurtosis - 3) / 10,  # Excess kurtosis over normal
        }

    def _run_significance_tests(
        self, returns: pd.Series, benchmark_returns: pd.Series | None = None
    ) -> list[SignificanceResult]:
        """Run statistical significance tests"""

        tests = []

        # Sharpe ratio significance
        sharpe_test = self.stats_tester.test_sharpe_ratio_significance(returns, benchmark_returns)
        tests.append(sharpe_test)

        # Bootstrap test for return mean
        mean_test = self.stats_tester.bootstrap_test(returns, lambda x: x.mean() * 252)
        tests.append(mean_test)

        # Permutation test vs benchmark if available
        if benchmark_returns is not None and len(benchmark_returns) > 30:
            perm_test = self.stats_tester.permutation_test(returns, benchmark_returns)
            tests.append(perm_test)

        return tests

    def _check_thresholds(self, metrics: PerformanceMetrics) -> dict[str, bool]:
        """Check if metrics meet validation thresholds"""

        return {
            "sharpe_ratio": metrics.sharpe_ratio >= self.thresholds.min_sharpe_ratio,
            "sortino_ratio": metrics.sortino_ratio >= self.thresholds.min_sortino_ratio,
            "calmar_ratio": metrics.calmar_ratio >= self.thresholds.min_calmar_ratio,
            "max_drawdown": metrics.max_drawdown <= self.thresholds.max_drawdown,
            "win_rate": metrics.win_rate >= self.thresholds.min_win_rate,
            "profit_factor": metrics.profit_factor >= self.thresholds.min_profit_factor,
            "var_95": abs(metrics.value_at_risk_95) <= self.thresholds.max_var_95,
            "cvar_95": metrics.conditional_var_95 <= self.thresholds.max_cvar_95,
            "downside_deviation": metrics.downside_deviation
            <= self.thresholds.max_downside_deviation,
            "monthly_win_rate": metrics.monthly_win_rate >= self.thresholds.min_monthly_win_rate,
            "trade_frequency": (
                (
                    self.thresholds.min_trades_per_year
                    <= metrics.total_trades
                    <= self.thresholds.max_trades_per_year
                )
                if metrics.total_trades > 0
                else True
            ),
            "turnover": metrics.annual_turnover <= self.thresholds.max_turnover_annual,
        }

    def _calculate_overall_score(
        self, metrics: PerformanceMetrics, meets_thresholds: dict[str, bool]
    ) -> float:
        """Calculate overall validation score (0-100)"""

        # Base score from thresholds
        threshold_score = sum(meets_thresholds.values()) / len(meets_thresholds) * 50

        # Performance bonus
        performance_bonus = 0
        performance_bonus += min(metrics.sharpe_ratio / 2.0, 1.0) * 15  # Up to 15 points for Sharpe
        performance_bonus += min(metrics.calmar_ratio / 1.0, 1.0) * 10  # Up to 10 points for Calmar
        performance_bonus += (
            min(metrics.sortino_ratio / 2.0, 1.0) * 10
        )  # Up to 10 points for Sortino
        performance_bonus += max(0, min(metrics.win_rate, 0.8)) * 15  # Up to 15 points for win rate

        return min(100, threshold_score + performance_bonus)

    def _calculate_risk_score(self, risk_assessment: dict[str, float]) -> float:
        """Calculate risk score (0-100, higher is better)"""

        # Risk penalties (lower is better for risk)
        risk_penalties = []
        for _risk_metric, value in risk_assessment.items():
            penalty = max(0, value - 1.0)  # Penalty for exceeding threshold
            risk_penalties.append(penalty)

        # Average risk penalty
        avg_penalty = np.mean(risk_penalties) if risk_penalties else 0

        # Convert to score (100 - penalty)
        return max(0, 100 - avg_penalty * 50)

    def _calculate_consistency_score(self, metrics: PerformanceMetrics) -> float:
        """Calculate consistency score (0-100)"""

        score = 0

        # Monthly win rate contribution
        score += metrics.monthly_win_rate * 30

        # Quarterly win rate contribution
        score += metrics.quarterly_win_rate * 25

        # Profit factor contribution
        pf_score = min(metrics.profit_factor / 3.0, 1.0) * 20
        score += pf_score

        # Drawdown duration penalty
        if metrics.max_drawdown_duration > 30:  # More than 30 days
            score -= min((metrics.max_drawdown_duration - 30) / 100, 0.25) * 100

        # Tail ratio (prefer balanced)
        if 1.0 <= metrics.tail_ratio <= 3.0:
            score += 15
        elif metrics.tail_ratio > 3.0:
            score += max(0, 15 - (metrics.tail_ratio - 3.0) * 5)

        return max(0, min(100, score))

    def _calculate_significance_score(self, significance_tests: list[SignificanceResult]) -> float:
        """Calculate statistical significance score (0-100)"""

        if not significance_tests:
            return 0

        significant_tests = [test for test in significance_tests if test.is_significant]
        significance_ratio = len(significant_tests) / len(significance_tests)

        # Base score from significance ratio
        base_score = significance_ratio * 60

        # Bonus for strong effect sizes
        effect_bonus = 0
        for test in significant_tests:
            if test.test_type == SignificanceTest.SHARPE_RATIO_TEST and test.effect_size > 1.0:
                effect_bonus += 20
            elif test.effect_size > 0:
                effect_bonus += 10

        return min(100, base_score + effect_bonus)

    def _make_final_assessment(
        self,
        overall_score: float,
        risk_score: float,
        consistency_score: float,
        significance_score: float,
        meets_thresholds: dict[str, bool],
    ) -> tuple[bool, str, float]:
        """Make final validation assessment"""

        # Weighted composite score
        weights = {"overall": 0.4, "risk": 0.3, "consistency": 0.2, "significance": 0.1}
        composite_score = (
            weights["overall"] * overall_score
            + weights["risk"] * risk_score
            + weights["consistency"] * consistency_score
            + weights["significance"] * significance_score
        )

        # Critical requirements (must pass)
        critical_thresholds = ["sharpe_ratio", "max_drawdown", "var_95"]
        critical_passed = all(
            meets_thresholds.get(threshold, False) for threshold in critical_thresholds
        )

        # Grade assignment
        if composite_score >= 85 and critical_passed:
            grade = "A"
            validated = True
            confidence = 0.95
        elif composite_score >= 75 and critical_passed:
            grade = "B"
            validated = True
            confidence = 0.85
        elif composite_score >= 65 and critical_passed:
            grade = "C"
            validated = True
            confidence = 0.75
        elif composite_score >= 55:
            grade = "D"
            validated = False
            confidence = 0.60
        else:
            grade = "F"
            validated = False
            confidence = 0.40

        return validated, grade, confidence

    def _generate_insights(
        self,
        metrics: PerformanceMetrics,
        risk_assessment: dict[str, float],
        meets_thresholds: dict[str, bool],
    ) -> tuple[list[str], list[str], list[str]]:
        """Generate strengths, weaknesses, and recommendations"""

        strengths = []
        weaknesses = []
        recommendations = []

        # Analyze performance strengths
        if metrics.sharpe_ratio > 1.0:
            strengths.append(
                f"Excellent risk-adjusted returns (Sharpe: {metrics.sharpe_ratio:.2f})"
            )
        elif metrics.sharpe_ratio > 0.5:
            strengths.append(f"Good risk-adjusted returns (Sharpe: {metrics.sharpe_ratio:.2f})")

        if metrics.max_drawdown < 0.05:
            strengths.append(f"Low maximum drawdown ({metrics.max_drawdown:.1%})")

        if metrics.win_rate > 0.6:
            strengths.append(f"High win rate ({metrics.win_rate:.1%})")

        if metrics.profit_factor > 2.0:
            strengths.append(f"Strong profit factor ({metrics.profit_factor:.2f})")

        # Analyze weaknesses
        if not meets_thresholds.get("sharpe_ratio", False):
            weaknesses.append(
                f"Low Sharpe ratio ({metrics.sharpe_ratio:.2f} < {self.thresholds.min_sharpe_ratio})"
            )
            recommendations.append("Improve risk-adjusted returns through better entry/exit timing")

        if not meets_thresholds.get("max_drawdown", False):
            weaknesses.append(
                f"Excessive drawdown ({metrics.max_drawdown:.1%} > {self.thresholds.max_drawdown:.1%})"
            )
            recommendations.append("Implement position sizing or stop-loss mechanisms")

        if metrics.win_rate < 0.4:
            weaknesses.append(f"Low win rate ({metrics.win_rate:.1%})")
            recommendations.append("Review signal quality and reduce false positives")

        if risk_assessment.get("tail_risk", 0) > 1.5:
            weaknesses.append("High tail risk exposure")
            recommendations.append("Consider tail risk hedging or position limits")

        if metrics.annual_turnover > 3.0:
            weaknesses.append(f"High turnover ({metrics.annual_turnover:.1f}x annually)")
            recommendations.append("Reduce trading frequency to lower transaction costs")

        # General recommendations
        if len(strengths) < 2:
            recommendations.append("Strategy needs significant improvement before deployment")

        if metrics.skewness < -1.0:
            recommendations.append("Address negative return skewness with risk management")

        return strengths, weaknesses, recommendations


# Factory function for easy initialization
def create_strategy_validator(
    min_sharpe_ratio: float = 0.5,
    max_drawdown: float = 0.15,
    min_confidence_level: float = 0.95,
    **kwargs,
) -> StrategyValidator:
    """Factory function to create strategy validator"""

    thresholds = ValidationThresholds(
        min_sharpe_ratio=min_sharpe_ratio,
        max_drawdown=max_drawdown,
        min_confidence_level=min_confidence_level,
        **kwargs,
    )

    return StrategyValidator(thresholds)


# Example usage and testing
if __name__ == "__main__":

    def main() -> None:
        """Example usage of Strategy Validation Engine"""
        print("Strategy Validation Engine Testing")
        print("=" * 40)

        # Generate sample returns for testing
        np.random.seed(42)
        dates = pd.date_range(start="2022-01-01", end="2023-12-31", freq="D")
        returns = pd.Series(
            np.random.normal(0.0008, 0.02, len(dates)),  # ~20% annual vol, 20% annual return
            index=dates,
        )

        print(f"Generated sample returns: {len(returns)} observations")
        print(f"Date range: {returns.index[0].date()} to {returns.index[-1].date()}")

        # Create validator
        validator = create_strategy_validator(
            min_sharpe_ratio=0.5, max_drawdown=0.15, min_confidence_level=0.95
        )

        # Run validation
        print("\n📊 Running comprehensive validation...")
        result = validator.validate_strategy(returns, "TestStrategy_001")

        print("\n🎯 VALIDATION RESULTS")
        print(f"   Strategy ID: {result.strategy_id}")
        print(f"   Overall Score: {result.overall_score:.1f}/100")
        print(f"   Validation Grade: {result.validation_grade}")
        print(f"   Is Validated: {'✅ Yes' if result.is_validated else '❌ No'}")
        print(f"   Confidence Level: {result.confidence_level:.1%}")

        print("\n📈 Performance Metrics:")
        metrics = result.performance_metrics
        print(f"   Sharpe Ratio: {metrics.sharpe_ratio:.3f}")
        print(f"   Max Drawdown: {metrics.max_drawdown:.1%}")
        print(f"   Win Rate: {metrics.win_rate:.1%}")
        print(f"   Profit Factor: {metrics.profit_factor:.2f}")

        print("\n🔍 Statistical Tests:")
        for test in result.significance_tests:
            print(
                f"   {test.test_type.value}: p-value = {test.p_value:.4f} ({'✅ Significant' if test.is_significant else '❌ Not significant'})"
            )

        print("\n💪 Strengths:")
        for strength in result.strengths:
            print(f"   • {strength}")

        if result.weaknesses:
            print("\n⚠️  Weaknesses:")
            for weakness in result.weaknesses:
                print(f"   • {weakness}")

        if result.recommendations:
            print("\n💡 Recommendations:")
            for rec in result.recommendations:
                print(f"   • {rec}")

        print("\n🚀 Strategy Validation Engine ready for production!")

    # Run the example
    main()
