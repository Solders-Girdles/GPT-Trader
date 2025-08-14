"""
Correlation Monitoring System
Phase 3, Week 4: RISK-025 to RISK-032
Rolling correlation matrices, breakdown detection, and regime changes
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.cluster import hierarchy

logger = logging.getLogger(__name__)


class CorrelationMethod(Enum):
    """Methods for correlation calculation"""

    PEARSON = "pearson"
    SPEARMAN = "spearman"
    KENDALL = "kendall"
    DISTANCE = "distance"
    PARTIAL = "partial"
    ROLLING = "rolling"
    EWMA = "ewma"


class RegimeType(Enum):
    """Market regime types"""

    NORMAL = "normal"
    STRESSED = "stressed"
    CRISIS = "crisis"
    RECOVERY = "recovery"
    TRANSITION = "transition"


@dataclass
class CorrelationBreakdown:
    """Detected correlation breakdown event"""

    timestamp: datetime
    asset_pair: tuple[str, str]
    historical_correlation: float
    current_correlation: float
    correlation_change: float
    significance: float
    duration_days: int | None = None
    regime: RegimeType | None = None
    description: str = ""


@dataclass
class CorrelationRegime:
    """Market correlation regime"""

    regime_type: RegimeType
    start_date: datetime
    end_date: datetime | None
    avg_correlation: float
    correlation_dispersion: float
    key_changes: list[tuple[str, str, float]]
    characteristics: dict[str, Any] = field(default_factory=dict)


class RollingCorrelationCalculator:
    """Calculate rolling correlation matrices"""

    def __init__(
        self,
        window_size: int = 60,
        min_periods: int = 30,
        method: CorrelationMethod = CorrelationMethod.PEARSON,
    ):
        """
        Initialize rolling correlation calculator.

        Args:
            window_size: Rolling window size in periods
            min_periods: Minimum periods for calculation
            method: Correlation calculation method
        """
        self.window_size = window_size
        self.min_periods = min_periods
        self.method = method

        # Storage for rolling correlations
        self.correlation_history = []
        self.current_correlation = None

    def calculate_correlation_matrix(
        self, returns: pd.DataFrame, method: CorrelationMethod | None = None
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix.

        Args:
            returns: DataFrame of asset returns
            method: Correlation method (overrides default)

        Returns:
            Correlation matrix
        """
        method = method or self.method

        if method == CorrelationMethod.PEARSON:
            return returns.corr(method="pearson")
        elif method == CorrelationMethod.SPEARMAN:
            return returns.corr(method="spearman")
        elif method == CorrelationMethod.KENDALL:
            return returns.corr(method="kendall")
        else:
            return returns.corr()

    def calculate_rolling_correlation(self, returns: pd.DataFrame) -> list[pd.DataFrame]:
        """
        Calculate rolling correlation matrices.

        Args:
            returns: DataFrame of asset returns

        Returns:
            List of correlation matrices over time
        """
        rolling_correlations = []

        for i in range(self.window_size, len(returns) + 1):
            window_returns = returns.iloc[i - self.window_size : i]

            if len(window_returns) >= self.min_periods:
                corr_matrix = self.calculate_correlation_matrix(window_returns)
                rolling_correlations.append(corr_matrix)

        self.correlation_history = rolling_correlations
        if rolling_correlations:
            self.current_correlation = rolling_correlations[-1]

        return rolling_correlations

    def calculate_ewma_correlation(self, returns: pd.DataFrame, halflife: int = 30) -> pd.DataFrame:
        """
        Calculate exponentially weighted correlation.

        Args:
            returns: DataFrame of asset returns
            halflife: Halflife for exponential weighting

        Returns:
            EWMA correlation matrix
        """
        # Calculate exponentially weighted covariance
        ewma_cov = returns.ewm(halflife=halflife, min_periods=self.min_periods).cov()

        # Convert to correlation
        ewma_std = returns.ewm(halflife=halflife, min_periods=self.min_periods).std()

        # Get the last timestamp's correlation
        if isinstance(ewma_cov.index, pd.MultiIndex):
            last_date = ewma_cov.index.get_level_values(0)[-1]
            cov_matrix = ewma_cov.loc[last_date]
        else:
            cov_matrix = ewma_cov.iloc[-len(returns.columns) :]

        # Normalize to get correlation
        std_matrix = np.outer(ewma_std.iloc[-1], ewma_std.iloc[-1])
        corr_matrix = cov_matrix / std_matrix

        return pd.DataFrame(corr_matrix, index=returns.columns, columns=returns.columns)

    def calculate_partial_correlation(
        self, returns: pd.DataFrame, control_variables: list[str] | None = None
    ) -> pd.DataFrame:
        """
        Calculate partial correlation controlling for other variables.

        Args:
            returns: DataFrame of asset returns
            control_variables: Variables to control for

        Returns:
            Partial correlation matrix
        """
        from sklearn.linear_model import LinearRegression

        n_assets = len(returns.columns)
        partial_corr = np.zeros((n_assets, n_assets))

        for i in range(n_assets):
            for j in range(i, n_assets):
                if i == j:
                    partial_corr[i, j] = 1.0
                else:
                    # Residualize both variables
                    if control_variables:
                        X_control = returns[control_variables]

                        # Residuals for variable i
                        reg_i = LinearRegression()
                        reg_i.fit(X_control, returns.iloc[:, i])
                        resid_i = returns.iloc[:, i] - reg_i.predict(X_control)

                        # Residuals for variable j
                        reg_j = LinearRegression()
                        reg_j.fit(X_control, returns.iloc[:, j])
                        resid_j = returns.iloc[:, j] - reg_j.predict(X_control)

                        # Correlation of residuals
                        partial_corr[i, j] = np.corrcoef(resid_i, resid_j)[0, 1]
                    else:
                        partial_corr[i, j] = np.corrcoef(returns.iloc[:, i], returns.iloc[:, j])[
                            0, 1
                        ]

                    partial_corr[j, i] = partial_corr[i, j]

        return pd.DataFrame(partial_corr, index=returns.columns, columns=returns.columns)


class CorrelationBreakdownDetector:
    """Detect correlation breakdowns and regime changes"""

    def __init__(self, threshold: float = 0.3, significance_level: float = 0.05):
        """
        Initialize breakdown detector.

        Args:
            threshold: Threshold for significant correlation change
            significance_level: Statistical significance level
        """
        self.threshold = threshold
        self.significance_level = significance_level

        # Breakdown history
        self.breakdowns: list[CorrelationBreakdown] = []
        self.regimes: list[CorrelationRegime] = []

    def detect_correlation_breakdown(
        self,
        historical_corr: pd.DataFrame,
        current_corr: pd.DataFrame,
        min_historical_corr: float = 0.5,
    ) -> list[CorrelationBreakdown]:
        """
        Detect correlation breakdowns.

        Args:
            historical_corr: Historical correlation matrix
            current_corr: Current correlation matrix
            min_historical_corr: Minimum historical correlation to consider

        Returns:
            List of detected breakdowns
        """
        breakdowns = []

        for i in range(len(historical_corr.columns)):
            for j in range(i + 1, len(historical_corr.columns)):
                asset1 = historical_corr.columns[i]
                asset2 = historical_corr.columns[j]

                hist_corr = historical_corr.iloc[i, j]
                curr_corr = current_corr.iloc[i, j]

                # Check if historically correlated
                if abs(hist_corr) >= min_historical_corr:
                    corr_change = curr_corr - hist_corr

                    # Check if breakdown is significant
                    if abs(corr_change) >= self.threshold:
                        breakdown = CorrelationBreakdown(
                            timestamp=datetime.now(),
                            asset_pair=(asset1, asset2),
                            historical_correlation=hist_corr,
                            current_correlation=curr_corr,
                            correlation_change=corr_change,
                            significance=abs(corr_change) / abs(hist_corr),
                            description=f"Correlation breakdown between {asset1} and {asset2}",
                        )
                        breakdowns.append(breakdown)

        self.breakdowns.extend(breakdowns)
        return breakdowns

    def test_correlation_stability(
        self, corr1: float, corr2: float, n1: int, n2: int
    ) -> tuple[float, bool]:
        """
        Test if correlation change is statistically significant.

        Args:
            corr1: First correlation
            corr2: Second correlation
            n1: Sample size for first correlation
            n2: Sample size for second correlation

        Returns:
            Tuple of (p-value, is_significant)
        """
        # Fisher Z transformation
        z1 = 0.5 * np.log((1 + corr1) / (1 - corr1))
        z2 = 0.5 * np.log((1 + corr2) / (1 - corr2))

        # Standard error
        se = np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))

        # Z statistic
        z_stat = (z1 - z2) / se

        # P-value (two-tailed)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        return p_value, p_value < self.significance_level

    def detect_regime_change(
        self, correlation_series: list[pd.DataFrame], window_size: int = 20
    ) -> list[CorrelationRegime]:
        """
        Detect correlation regime changes.

        Args:
            correlation_series: Time series of correlation matrices
            window_size: Window for regime detection

        Returns:
            List of detected regimes
        """
        if len(correlation_series) < window_size * 2:
            return []

        regimes = []
        avg_correlations = []

        # Calculate average correlation over time
        for corr_matrix in correlation_series:
            # Get upper triangle (excluding diagonal)
            upper_triangle = np.triu(corr_matrix.values, k=1)
            avg_corr = upper_triangle[upper_triangle != 0].mean()
            avg_correlations.append(avg_corr)

        # Detect regime changes using rolling statistics
        avg_corr_series = pd.Series(avg_correlations)
        rolling_mean = avg_corr_series.rolling(window_size).mean()
        rolling_std = avg_corr_series.rolling(window_size).std()

        # Identify regime changes
        z_scores = (avg_corr_series - rolling_mean) / rolling_std
        regime_changes = abs(z_scores) > 2  # 2 standard deviations

        # Create regime objects
        current_regime_start = 0
        for i, is_change in enumerate(regime_changes):
            if is_change and i > current_regime_start + window_size:
                # Determine regime type
                avg_corr = avg_correlations[current_regime_start:i]
                mean_corr = np.mean(avg_corr)

                if mean_corr > 0.7:
                    regime_type = RegimeType.CRISIS
                elif mean_corr > 0.5:
                    regime_type = RegimeType.STRESSED
                elif mean_corr < 0.3:
                    regime_type = RegimeType.RECOVERY
                else:
                    regime_type = RegimeType.NORMAL

                regime = CorrelationRegime(
                    regime_type=regime_type,
                    start_date=datetime.now()
                    - timedelta(days=len(correlation_series) - current_regime_start),
                    end_date=datetime.now() - timedelta(days=len(correlation_series) - i),
                    avg_correlation=mean_corr,
                    correlation_dispersion=np.std(avg_corr),
                    key_changes=[],
                )
                regimes.append(regime)
                current_regime_start = i

        self.regimes = regimes
        return regimes


class CorrelationRiskManager:
    """Manage correlation-related risks"""

    def __init__(self):
        """Initialize correlation risk manager"""
        self.correlation_limits = {
            "max_avg_correlation": 0.8,
            "min_diversification": 0.3,
            "max_concentration": 0.6,
        }

        self.risk_metrics = {}

    def calculate_diversification_ratio(
        self, weights: np.ndarray, volatilities: np.ndarray, correlation_matrix: np.ndarray
    ) -> float:
        """
        Calculate portfolio diversification ratio.

        Args:
            weights: Portfolio weights
            volatilities: Asset volatilities
            correlation_matrix: Correlation matrix

        Returns:
            Diversification ratio
        """
        # Weighted average volatility
        weighted_avg_vol = np.sum(weights * volatilities)

        # Portfolio volatility
        cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix
        portfolio_vol = np.sqrt(weights @ cov_matrix @ weights.T)

        # Diversification ratio
        div_ratio = weighted_avg_vol / portfolio_vol

        return div_ratio

    def calculate_effective_correlation(
        self, weights: np.ndarray, correlation_matrix: np.ndarray
    ) -> float:
        """
        Calculate effective portfolio correlation.

        Args:
            weights: Portfolio weights
            correlation_matrix: Correlation matrix

        Returns:
            Effective correlation
        """
        n_assets = len(weights)
        total_weight = 0
        weighted_corr = 0

        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                pair_weight = weights[i] * weights[j]
                total_weight += pair_weight
                weighted_corr += pair_weight * correlation_matrix[i, j]

        if total_weight > 0:
            return weighted_corr / total_weight
        return 0

    def assess_concentration_risk(
        self, correlation_matrix: pd.DataFrame, threshold: float = 0.7
    ) -> dict[str, Any]:
        """
        Assess concentration risk from correlations.

        Args:
            correlation_matrix: Correlation matrix
            threshold: Correlation threshold for clustering

        Returns:
            Concentration risk assessment
        """
        # Hierarchical clustering
        distance_matrix = 1 - abs(correlation_matrix)
        linkage_matrix = hierarchy.linkage(
            distance_matrix.values[np.triu_indices(len(correlation_matrix), k=1)], method="ward"
        )

        # Get clusters
        clusters = hierarchy.fcluster(linkage_matrix, threshold, criterion="distance")

        # Analyze clusters
        cluster_sizes = pd.Series(clusters).value_counts()
        max_cluster_size = cluster_sizes.max()
        n_clusters = len(cluster_sizes)

        # Calculate concentration
        concentration = max_cluster_size / len(correlation_matrix)

        return {
            "n_clusters": n_clusters,
            "max_cluster_size": max_cluster_size,
            "concentration": concentration,
            "cluster_assignments": dict(zip(correlation_matrix.columns, clusters, strict=False)),
            "risk_level": (
                "High" if concentration > self.correlation_limits["max_concentration"] else "Low"
            ),
        }

    def calculate_correlation_var(
        self, returns: pd.DataFrame, weights: np.ndarray, confidence_level: float = 0.95
    ) -> float:
        """
        Calculate VaR considering correlation.

        Args:
            returns: Asset returns
            weights: Portfolio weights
            confidence_level: VaR confidence level

        Returns:
            Correlation-adjusted VaR
        """
        # Portfolio returns
        portfolio_returns = returns @ weights

        # Calculate VaR
        var = np.percentile(portfolio_returns, (1 - confidence_level) * 100)

        return var


class CorrelationMonitoringSystem:
    """Complete correlation monitoring system"""

    def __init__(self):
        """Initialize correlation monitoring system"""
        self.calculator = RollingCorrelationCalculator()
        self.detector = CorrelationBreakdownDetector()
        self.risk_manager = CorrelationRiskManager()

        # Monitoring state
        self.is_monitoring = False
        self.alert_callbacks = []

        # Data storage
        self.correlation_history = deque(maxlen=1000)
        self.breakdown_history = []
        self.regime_history = []

    def update_correlations(self, returns: pd.DataFrame) -> dict[str, Any]:
        """
        Update correlation monitoring with new data.

        Args:
            returns: New return data

        Returns:
            Monitoring results
        """
        # Calculate current correlation
        current_corr = self.calculator.calculate_correlation_matrix(returns)

        # Store in history
        self.correlation_history.append({"timestamp": datetime.now(), "correlation": current_corr})

        # Calculate rolling correlations
        rolling_corr = self.calculator.calculate_rolling_correlation(returns)

        # Detect breakdowns if we have history
        breakdowns = []
        if len(rolling_corr) > 1:
            historical_corr = rolling_corr[-2]
            breakdowns = self.detector.detect_correlation_breakdown(historical_corr, current_corr)
            self.breakdown_history.extend(breakdowns)

        # Detect regime changes
        regimes = self.detector.detect_regime_change(rolling_corr)
        self.regime_history.extend(regimes)

        # Calculate risk metrics
        weights = np.ones(len(returns.columns)) / len(returns.columns)  # Equal weight
        volatilities = returns.std().values

        div_ratio = self.risk_manager.calculate_diversification_ratio(
            weights, volatilities, current_corr.values
        )

        eff_corr = self.risk_manager.calculate_effective_correlation(weights, current_corr.values)

        concentration = self.risk_manager.assess_concentration_risk(current_corr)

        # Prepare results
        results = {
            "current_correlation": current_corr,
            "breakdowns": breakdowns,
            "current_regime": regimes[-1] if regimes else None,
            "diversification_ratio": div_ratio,
            "effective_correlation": eff_corr,
            "concentration_risk": concentration,
            "alerts": [],
        }

        # Check for alerts
        if eff_corr > 0.7:
            results["alerts"].append("High correlation environment detected")

        if div_ratio < 1.2:
            results["alerts"].append("Low diversification benefit")

        if concentration["risk_level"] == "High":
            results["alerts"].append("High concentration risk detected")

        if breakdowns:
            results["alerts"].append(f"{len(breakdowns)} correlation breakdowns detected")

        # Trigger callbacks
        for callback in self.alert_callbacks:
            callback(results)

        return results

    def generate_correlation_report(self) -> str:
        """
        Generate correlation monitoring report.

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 60)
        report.append("CORRELATION MONITORING REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Current state
        if self.correlation_history:
            latest = self.correlation_history[-1]
            report.append("CURRENT CORRELATION STATE")
            report.append("-" * 40)
            corr_matrix = latest["correlation"]

            # Average correlation
            upper_triangle = np.triu(corr_matrix.values, k=1)
            avg_corr = upper_triangle[upper_triangle != 0].mean()
            report.append(f"  Average Correlation: {avg_corr:.3f}")

            # Highest correlations
            report.append("  Highest Correlations:")
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        report.append(
                            f"    {corr_matrix.columns[i]} - {corr_matrix.columns[j]}: {corr_val:.3f}"
                        )
            report.append("")

        # Recent breakdowns
        if self.breakdown_history:
            report.append("RECENT CORRELATION BREAKDOWNS")
            report.append("-" * 40)
            recent_breakdowns = self.breakdown_history[-5:]  # Last 5
            for breakdown in recent_breakdowns:
                report.append(f"  {breakdown.asset_pair[0]} - {breakdown.asset_pair[1]}")
                report.append(f"    Historical: {breakdown.historical_correlation:.3f}")
                report.append(f"    Current: {breakdown.current_correlation:.3f}")
                report.append(f"    Change: {breakdown.correlation_change:.3f}")
                report.append("")

        # Regime analysis
        if self.regime_history:
            report.append("CORRELATION REGIMES")
            report.append("-" * 40)
            current_regime = self.regime_history[-1]
            report.append(f"  Current Regime: {current_regime.regime_type.value}")
            report.append(f"  Average Correlation: {current_regime.avg_correlation:.3f}")
            report.append(f"  Dispersion: {current_regime.correlation_dispersion:.3f}")
            report.append("")

        report.append("=" * 60)

        return "\n".join(report)

    def visualize_correlation_matrix(
        self, correlation_matrix: pd.DataFrame, title: str = "Correlation Matrix"
    ) -> None:
        """
        Visualize correlation matrix as heatmap.

        Args:
            correlation_matrix: Correlation matrix to visualize
            title: Plot title
        """
        plt.figure(figsize=(12, 10))

        # Create heatmap
        sns.heatmap(
            correlation_matrix,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
        )

        plt.title(title, fontsize=16, pad=20)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Example usage
    monitoring = CorrelationMonitoringSystem()

    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=100)

    # Create correlated returns
    n_assets = 5
    returns_data = np.random.multivariate_normal(
        mean=[0] * n_assets, cov=np.eye(n_assets) * 0.01, size=100
    )

    returns = pd.DataFrame(
        returns_data, index=dates, columns=[f"Asset_{i+1}" for i in range(n_assets)]
    )

    # Update monitoring
    results = monitoring.update_correlations(returns)

    # Generate report
    report = monitoring.generate_correlation_report()
    print(report)

    # Print alerts
    if results["alerts"]:
        print("\nALERTS:")
        for alert in results["alerts"]:
            print(f"  ⚠️  {alert}")
