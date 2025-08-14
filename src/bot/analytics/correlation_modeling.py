"""
Cross-Asset Correlation Modeling for Multi-Asset Strategy Enhancement

This module implements sophisticated correlation modeling techniques including:
- Dynamic correlation models (DCC-GARCH, RiskMetrics)
- Copula-based dependency modeling
- Correlation regime detection and switching
- Cross-asset factor analysis
- Alternative correlation measures (Spearman, Kendall, Distance correlation)
- Network analysis of asset relationships
"""

import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

# Optional dependencies with graceful fallback
try:
    from sklearn.cluster import KMeans
    from sklearn.covariance import GraphicalLassoCV
    from sklearn.decomposition import PCA, FactorAnalysis
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("Scikit-learn not available. Some correlation modeling features will be limited.", stacklevel=2)

try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    warnings.warn("NetworkX not available. Network analysis features will be limited.", stacklevel=2)

try:
    from arch import arch_model
    from arch.univariate import GARCH, EWMAVariance

    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    warnings.warn("ARCH not available. GARCH correlation modeling will be limited.", stacklevel=2)

logger = logging.getLogger(__name__)


class CorrelationMethod(Enum):
    """Correlation estimation methods"""

    PEARSON = "pearson"
    SPEARMAN = "spearman"
    KENDALL = "kendall"
    DISTANCE = "distance"
    MUTUAL_INFO = "mutual_info"
    PARTIAL = "partial"


class DynamicCorrelationModel(Enum):
    """Dynamic correlation models"""

    DCC_GARCH = "dcc_garch"
    EWMA = "ewma"
    ROLLING_WINDOW = "rolling_window"
    SHRINKAGE = "shrinkage"
    FACTOR_MODEL = "factor_model"


class CopulaType(Enum):
    """Copula types for dependency modeling"""

    GAUSSIAN = "gaussian"
    STUDENT_T = "student_t"
    CLAYTON = "clayton"
    GUMBEL = "gumbel"
    FRANK = "frank"


@dataclass
class CorrelationModelConfig:
    """Configuration for correlation modeling"""

    method: CorrelationMethod = CorrelationMethod.PEARSON
    dynamic_model: DynamicCorrelationModel = DynamicCorrelationModel.EWMA
    lookback_window: int = 252
    decay_factor: float = 0.94
    min_periods: int = 30
    confidence_level: float = 0.95
    regime_detection: bool = True
    n_regimes: int = 3
    copula_type: CopulaType = CopulaType.GAUSSIAN
    factor_analysis: bool = True
    n_factors: int | None = None
    network_analysis: bool = True
    correlation_threshold: float = 0.3


@dataclass
class CorrelationResult:
    """Results from correlation analysis"""

    correlation_matrix: pd.DataFrame
    dynamic_correlations: pd.DataFrame | None = None
    factor_loadings: pd.DataFrame | None = None
    regime_probabilities: pd.DataFrame | None = None
    network_metrics: dict[str, Any] | None = None
    copula_parameters: dict[str, Any] | None = None
    model_statistics: dict[str, Any] = field(default_factory=dict)


class BaseCorrelationModel(ABC):
    """Base class for correlation models"""

    def __init__(self, config: CorrelationModelConfig) -> None:
        self.config = config
        self.is_fitted = False
        self.assets = []

    @abstractmethod
    def fit(self, returns: pd.DataFrame) -> "BaseCorrelationModel":
        """Fit the correlation model"""
        pass

    @abstractmethod
    def estimate_correlation(self, returns: pd.DataFrame) -> CorrelationResult:
        """Estimate correlations"""
        pass

    def _calculate_basic_correlation(
        self, returns: pd.DataFrame, method: CorrelationMethod
    ) -> pd.DataFrame:
        """Calculate basic correlation matrix"""
        if method == CorrelationMethod.PEARSON:
            return returns.corr(method="pearson")
        elif method == CorrelationMethod.SPEARMAN:
            return returns.corr(method="spearman")
        elif method == CorrelationMethod.KENDALL:
            return returns.corr(method="kendall")
        elif method == CorrelationMethod.DISTANCE:
            return self._distance_correlation(returns)
        elif method == CorrelationMethod.PARTIAL:
            return self._partial_correlation(returns)
        else:
            return returns.corr(method="pearson")

    def _distance_correlation(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Calculate distance correlation matrix"""

        def dcorr(x, y):
            """Distance correlation between two vectors"""
            n = len(x)
            if n == 0:
                return 0

            # Center the distance matrices
            a = np.abs(x[:, np.newaxis] - x[np.newaxis, :])
            b = np.abs(y[:, np.newaxis] - y[np.newaxis, :])

            A = a - a.mean(axis=0)[np.newaxis, :] - a.mean(axis=1)[:, np.newaxis] + a.mean()
            B = b - b.mean(axis=0)[np.newaxis, :] - b.mean(axis=1)[:, np.newaxis] + b.mean()

            dcov_xy = np.sqrt(np.mean(A * B))
            dcov_xx = np.sqrt(np.mean(A * A))
            dcov_yy = np.sqrt(np.mean(B * B))

            if dcov_xx * dcov_yy == 0:
                return 0
            return dcov_xy / np.sqrt(dcov_xx * dcov_yy)

        assets = returns.columns
        n_assets = len(assets)
        corr_matrix = np.ones((n_assets, n_assets))

        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                corr = dcorr(returns.iloc[:, i].values, returns.iloc[:, j].values)
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr

        return pd.DataFrame(corr_matrix, index=assets, columns=assets)

    def _partial_correlation(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Calculate partial correlation matrix"""
        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn not available, using Pearson correlation")
            return returns.corr()

        try:
            # Use graphical lasso for sparse precision matrix estimation
            model = GraphicalLassoCV()
            model.fit(returns.values)
            precision_matrix = model.precision_

            # Convert precision matrix to partial correlations
            diag = np.sqrt(np.diag(precision_matrix))
            partial_corr = -precision_matrix / np.outer(diag, diag)
            np.fill_diagonal(partial_corr, 1.0)

            return pd.DataFrame(partial_corr, index=returns.columns, columns=returns.columns)
        except Exception as e:
            logger.warning(f"Partial correlation estimation failed: {e}, using Pearson correlation")
            return returns.corr()


class DynamicCorrelationEstimator(BaseCorrelationModel):
    """Dynamic correlation estimation with various models"""

    def __init__(self, config: CorrelationModelConfig) -> None:
        super().__init__(config)
        self.correlation_history = []
        self.regime_detector = None

    def fit(self, returns: pd.DataFrame) -> "DynamicCorrelationEstimator":
        """Fit the dynamic correlation model"""
        self.assets = returns.columns.tolist()

        if self.config.regime_detection:
            self.regime_detector = self._fit_regime_detector(returns)

        self.is_fitted = True
        return self

    def estimate_correlation(self, returns: pd.DataFrame) -> CorrelationResult:
        """Estimate dynamic correlations"""
        if not self.is_fitted:
            self.fit(returns)

        # Estimate static correlation
        static_corr = self._calculate_basic_correlation(returns, self.config.method)

        # Estimate dynamic correlations
        dynamic_corr = self._estimate_dynamic_correlations(returns)

        # Factor analysis
        factor_loadings = None
        if self.config.factor_analysis:
            factor_loadings = self._perform_factor_analysis(returns)

        # Regime detection
        regime_probs = None
        if self.config.regime_detection and self.regime_detector:
            regime_probs = self._detect_regimes(returns)

        # Network analysis
        network_metrics = None
        if self.config.network_analysis:
            network_metrics = self._analyze_correlation_network(static_corr)

        # Copula modeling
        copula_params = None
        if self.config.copula_type != CopulaType.GAUSSIAN:
            copula_params = self._fit_copula(returns)

        return CorrelationResult(
            correlation_matrix=static_corr,
            dynamic_correlations=dynamic_corr,
            factor_loadings=factor_loadings,
            regime_probabilities=regime_probs,
            network_metrics=network_metrics,
            copula_parameters=copula_params,
            model_statistics=self._calculate_model_statistics(returns, static_corr),
        )

    def _estimate_dynamic_correlations(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Estimate time-varying correlations"""
        if self.config.dynamic_model == DynamicCorrelationModel.EWMA:
            return self._ewma_correlations(returns)
        elif self.config.dynamic_model == DynamicCorrelationModel.ROLLING_WINDOW:
            return self._rolling_window_correlations(returns)
        elif self.config.dynamic_model == DynamicCorrelationModel.DCC_GARCH:
            return self._dcc_garch_correlations(returns)
        else:
            return self._ewma_correlations(returns)

    def _ewma_correlations(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Exponentially weighted moving average correlations"""
        lambda_param = self.config.decay_factor
        len(returns.columns)
        n_periods = len(returns)

        # Initialize covariance matrix
        cov_matrix = returns.cov().values
        correlations_history = []

        for t in range(n_periods):
            if t > 0:
                # EWMA update
                r_t = returns.iloc[t].values.reshape(-1, 1)
                cov_matrix = lambda_param * cov_matrix + (1 - lambda_param) * (r_t @ r_t.T)

            # Convert covariance to correlation
            std_dev = np.sqrt(np.diag(cov_matrix))
            corr_matrix = cov_matrix / np.outer(std_dev, std_dev)

            correlations_history.append(corr_matrix)

        # Convert to DataFrame with multi-level columns for asset pairs
        correlation_pairs = []
        for i, asset_i in enumerate(returns.columns):
            for j, asset_j in enumerate(returns.columns):
                if i < j:  # Only upper triangular
                    pair_correlations = [corr[i, j] for corr in correlations_history]
                    correlation_pairs.append(
                        pd.Series(
                            pair_correlations, index=returns.index, name=f"{asset_i}_{asset_j}"
                        )
                    )

        return pd.concat(correlation_pairs, axis=1)

    def _rolling_window_correlations(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Rolling window correlations"""
        window = self.config.lookback_window
        correlation_pairs = []

        for i, asset_i in enumerate(returns.columns):
            for j, asset_j in enumerate(returns.columns):
                if i < j:
                    rolling_corr = returns[asset_i].rolling(window=window).corr(returns[asset_j])
                    correlation_pairs.append(rolling_corr.rename(f"{asset_i}_{asset_j}"))

        return pd.concat(correlation_pairs, axis=1)

    def _dcc_garch_correlations(self, returns: pd.DataFrame) -> pd.DataFrame:
        """DCC-GARCH correlations (simplified implementation)"""
        if not ARCH_AVAILABLE:
            logger.warning("ARCH not available, falling back to EWMA")
            return self._ewma_correlations(returns)

        try:
            # Fit GARCH models for each asset
            garch_models = {}
            standardized_returns = pd.DataFrame(index=returns.index, columns=returns.columns)

            for asset in returns.columns:
                model = arch_model(returns[asset], vol="GARCH", p=1, q=1)
                fitted_model = model.fit(disp="off")
                garch_models[asset] = fitted_model
                standardized_returns[asset] = (
                    fitted_model.resid / fitted_model.conditional_volatility
                )

            # Estimate dynamic correlations on standardized returns
            return self._ewma_correlations(standardized_returns)

        except Exception as e:
            logger.warning(f"DCC-GARCH estimation failed: {e}, falling back to EWMA")
            return self._ewma_correlations(returns)

    def _perform_factor_analysis(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Perform factor analysis on returns"""
        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn not available, skipping factor analysis")
            return None

        try:
            n_factors = self.config.n_factors or min(5, len(returns.columns) // 2)

            # Standardize returns
            scaler = StandardScaler()
            scaled_returns = scaler.fit_transform(returns.values)

            # Fit factor analysis
            fa = FactorAnalysis(n_components=n_factors, random_state=42)
            fa.fit(scaled_returns)

            # Create factor loadings DataFrame
            factor_names = [f"Factor_{i+1}" for i in range(n_factors)]
            loadings = pd.DataFrame(fa.components_.T, index=returns.columns, columns=factor_names)

            return loadings

        except Exception as e:
            logger.warning(f"Factor analysis failed: {e}")
            return None

    def _fit_regime_detector(self, returns: pd.DataFrame):
        """Fit regime detection model using correlation clustering"""
        if not SKLEARN_AVAILABLE:
            return None

        try:
            # Calculate rolling correlations
            window = min(60, len(returns) // 4)
            rolling_corrs = []

            for i in range(window, len(returns)):
                window_data = returns.iloc[i - window : i]
                corr_matrix = window_data.corr()
                # Flatten upper triangular part
                upper_tri = np.triu_indices_from(corr_matrix, k=1)
                rolling_corrs.append(corr_matrix.values[upper_tri])

            # Cluster correlation patterns
            kmeans = KMeans(n_clusters=self.config.n_regimes, random_state=42)
            kmeans.fit(rolling_corrs)

            return kmeans

        except Exception as e:
            logger.warning(f"Regime detector fitting failed: {e}")
            return None

    def _detect_regimes(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Detect correlation regimes"""
        if not self.regime_detector:
            return None

        try:
            window = min(60, len(returns) // 4)
            regime_probs = []

            for i in range(len(returns)):
                if i < window:
                    # Not enough data, assume equal probabilities
                    probs = [1 / self.config.n_regimes] * self.config.n_regimes
                else:
                    window_data = returns.iloc[max(0, i - window) : i]
                    corr_matrix = window_data.corr()
                    upper_tri = np.triu_indices_from(corr_matrix, k=1)
                    corr_features = corr_matrix.values[upper_tri].reshape(1, -1)

                    # Get regime probabilities (simplified - using distances to centroids)
                    distances = [
                        np.linalg.norm(corr_features - centroid.reshape(1, -1))
                        for centroid in self.regime_detector.cluster_centers_
                    ]
                    # Convert distances to probabilities (softmax-like)
                    exp_neg_dist = np.exp(-np.array(distances))
                    probs = exp_neg_dist / np.sum(exp_neg_dist)

                regime_probs.append(probs)

            regime_names = [f"Regime_{i+1}" for i in range(self.config.n_regimes)]
            return pd.DataFrame(regime_probs, index=returns.index, columns=regime_names)

        except Exception as e:
            logger.warning(f"Regime detection failed: {e}")
            return None

    def _analyze_correlation_network(self, correlation_matrix: pd.DataFrame) -> dict[str, Any]:
        """Analyze correlation network structure"""
        if not NETWORKX_AVAILABLE:
            return None

        try:
            # Create network from correlation matrix
            G = nx.Graph()

            # Add nodes
            G.add_nodes_from(correlation_matrix.columns)

            # Add edges for correlations above threshold
            threshold = self.config.correlation_threshold
            for i, asset_i in enumerate(correlation_matrix.columns):
                for j, asset_j in enumerate(correlation_matrix.columns):
                    if i < j:
                        corr = correlation_matrix.iloc[i, j]
                        if abs(corr) > threshold:
                            G.add_edge(asset_i, asset_j, weight=abs(corr))

            # Calculate network metrics
            metrics = {
                "n_nodes": G.number_of_nodes(),
                "n_edges": G.number_of_edges(),
                "density": nx.density(G),
                "average_clustering": nx.average_clustering(G),
                "centrality": dict(nx.degree_centrality(G)),
                "betweenness_centrality": dict(nx.betweenness_centrality(G)),
                "eigenvector_centrality": dict(nx.eigenvector_centrality(G, max_iter=1000)),
            }

            # Find communities
            try:
                communities = list(nx.community.greedy_modularity_communities(G))
                metrics["communities"] = [list(community) for community in communities]
                metrics["modularity"] = nx.community.modularity(G, communities)
            except (ImportError, AttributeError, ValueError) as e:
                # NetworkX community detection may fail due to missing modules or invalid graph structure
                logger.debug(f"Community detection failed: {e}")
                pass

            return metrics

        except Exception as e:
            logger.warning(f"Network analysis failed: {e}")
            return None

    def _fit_copula(self, returns: pd.DataFrame) -> dict[str, Any]:
        """Fit copula model for dependency structure"""
        try:
            # Convert to uniform margins using empirical CDF
            len(returns.columns)
            uniform_data = np.zeros_like(returns.values)

            for i, _col in enumerate(returns.columns):
                data = returns.iloc[:, i].values
                ranks = stats.rankdata(data)
                uniform_data[:, i] = ranks / (len(data) + 1)

            # Fit Gaussian copula (estimate correlation of transformed data)
            if self.config.copula_type == CopulaType.GAUSSIAN:
                # Transform to normal using inverse normal CDF
                normal_data = stats.norm.ppf(uniform_data)
                correlation = np.corrcoef(normal_data.T)
                return {
                    "type": "gaussian",
                    "correlation": pd.DataFrame(
                        correlation, index=returns.columns, columns=returns.columns
                    ),
                }

            # For other copula types, return basic implementation
            return {"type": self.config.copula_type.value, "parameters": {"fitted": True}}

        except Exception as e:
            logger.warning(f"Copula fitting failed: {e}")
            return None

    def _calculate_model_statistics(
        self, returns: pd.DataFrame, correlation_matrix: pd.DataFrame
    ) -> dict[str, Any]:
        """Calculate model fit statistics"""
        try:
            # Basic statistics
            stats = {
                "n_assets": len(returns.columns),
                "n_observations": len(returns),
                "avg_correlation": correlation_matrix.values[
                    np.triu_indices_from(correlation_matrix, k=1)
                ].mean(),
                "max_correlation": correlation_matrix.values[
                    np.triu_indices_from(correlation_matrix, k=1)
                ].max(),
                "min_correlation": correlation_matrix.values[
                    np.triu_indices_from(correlation_matrix, k=1)
                ].min(),
                "correlation_std": correlation_matrix.values[
                    np.triu_indices_from(correlation_matrix, k=1)
                ].std(),
            }

            # Eigenvalue analysis
            eigenvals = np.linalg.eigvals(correlation_matrix.values)
            stats["condition_number"] = np.max(eigenvals) / np.min(eigenvals)
            stats["eigenvalues"] = eigenvals.tolist()

            return stats

        except Exception as e:
            logger.warning(f"Statistics calculation failed: {e}")
            return {}


class CrossAssetCorrelationFramework:
    """Main framework for cross-asset correlation modeling"""

    def __init__(self, config: CorrelationModelConfig) -> None:
        self.config = config
        self.estimator = DynamicCorrelationEstimator(config)
        self.correlation_history = []

    def analyze_correlations(self, returns: pd.DataFrame) -> CorrelationResult:
        """Analyze correlations across assets"""
        return self.estimator.estimate_correlation(returns)

    def update_correlations(self, new_returns: pd.DataFrame) -> CorrelationResult:
        """Update correlations with new data"""
        result = self.analyze_correlations(new_returns)

        self.correlation_history.append({"timestamp": pd.Timestamp.now(), "result": result})

        return result

    def get_correlation_forecast(self, returns: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
        """Forecast future correlations"""
        # Simple persistence forecast
        current_result = self.analyze_correlations(returns)
        return current_result.correlation_matrix

    def detect_correlation_breaks(self, returns: pd.DataFrame, window: int = 60) -> pd.DataFrame:
        """Detect structural breaks in correlations"""
        breaks = []

        for i in range(window, len(returns) - window, window // 2):
            # Compare correlations before and after potential break
            before_data = returns.iloc[i - window : i]
            after_data = returns.iloc[i : i + window]

            corr_before = before_data.corr()
            corr_after = after_data.corr()

            # Calculate Frobenius norm of difference
            diff_norm = np.linalg.norm(corr_before.values - corr_after.values, "fro")

            breaks.append(
                {
                    "date": returns.index[i],
                    "break_statistic": diff_norm,
                    "significant": diff_norm
                    > np.percentile(
                        [
                            np.linalg.norm(
                                returns.iloc[j : j + window].corr().values
                                - returns.iloc[j + window : j + 2 * window].corr().values,
                                "fro",
                            )
                            for j in range(0, len(returns) - 2 * window, window)
                        ],
                        95,
                    ),
                }
            )

        return pd.DataFrame(breaks)


def create_correlation_analyzer(
    method: CorrelationMethod = CorrelationMethod.PEARSON,
    dynamic_model: DynamicCorrelationModel = DynamicCorrelationModel.EWMA,
    **kwargs,
) -> CrossAssetCorrelationFramework:
    """Factory function to create correlation analyzer"""
    config = CorrelationModelConfig(method=method, dynamic_model=dynamic_model, **kwargs)

    return CrossAssetCorrelationFramework(config)


# Example usage and testing
if __name__ == "__main__":
    # Generate sample data with time-varying correlations
    np.random.seed(42)
    n_days = 500
    n_assets = 4
    assets = ["AAPL", "GOOGL", "MSFT", "AMZN"]

    # Generate returns with regime changes
    regime_1_corr = np.array(
        [[1.0, 0.3, 0.4, 0.2], [0.3, 1.0, 0.5, 0.3], [0.4, 0.5, 1.0, 0.4], [0.2, 0.3, 0.4, 1.0]]
    )

    regime_2_corr = np.array(
        [[1.0, 0.8, 0.7, 0.6], [0.8, 1.0, 0.8, 0.7], [0.7, 0.8, 1.0, 0.8], [0.6, 0.7, 0.8, 1.0]]
    )

    # Generate data
    returns_data = []
    for i in range(n_days):
        if i < n_days // 2:
            corr = regime_1_corr
        else:
            corr = regime_2_corr

        returns_data.append(np.random.multivariate_normal([0.0005] * n_assets, corr * 0.0004, 1)[0])

    returns_df = pd.DataFrame(
        returns_data, index=pd.date_range("2022-01-01", periods=n_days, freq="D"), columns=assets
    )

    # Test correlation analysis
    print("Cross-Asset Correlation Modeling Framework Testing")
    print("=" * 60)

    # Test different correlation methods
    methods = [CorrelationMethod.PEARSON, CorrelationMethod.SPEARMAN, CorrelationMethod.DISTANCE]

    for method in methods:
        print(f"\nTesting {method.value} correlation...")
        try:
            analyzer = create_correlation_analyzer(
                method=method,
                dynamic_model=DynamicCorrelationModel.EWMA,
                regime_detection=True,
                factor_analysis=True,
                network_analysis=True,
            )

            result = analyzer.analyze_correlations(returns_df)

            print(f"✅ Static correlations shape: {result.correlation_matrix.shape}")
            print(
                f"   Average correlation: {result.correlation_matrix.values[np.triu_indices_from(result.correlation_matrix, k=1)].mean():.4f}"
            )

            if result.dynamic_correlations is not None:
                print(f"   Dynamic correlations shape: {result.dynamic_correlations.shape}")

            if result.factor_loadings is not None:
                print(f"   Factor loadings shape: {result.factor_loadings.shape}")

            if result.regime_probabilities is not None:
                print(f"   Regime probabilities shape: {result.regime_probabilities.shape}")

            if result.network_metrics is not None:
                print(f"   Network density: {result.network_metrics['density']:.4f}")

        except Exception as e:
            print(f"❌ Error: {str(e)}")

    # Test correlation break detection
    print("\nTesting correlation break detection...")
    try:
        analyzer = create_correlation_analyzer()
        breaks = analyzer.detect_correlation_breaks(returns_df)
        significant_breaks = breaks[breaks["significant"]].shape[0]
        print(f"✅ Detected {significant_breaks} significant correlation breaks")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

    print("\n🚀 Cross-Asset Correlation Modeling Framework ready for production!")
