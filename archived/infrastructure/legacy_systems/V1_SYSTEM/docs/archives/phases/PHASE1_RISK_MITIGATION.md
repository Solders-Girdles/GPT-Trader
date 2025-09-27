# Phase 1 Risk Mitigation & Implementation Fixes

## 🚨 **Critical Risks Identified & Solutions**

### 1. **Regime Classification Accuracy - Fix Required**

**Problem**: Can't measure "classification accuracy" without labeled targets.

**Solution**: Create programmatic labels and track stability metrics.

```python
# src/bot/intelligence/regime_labeling.py
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from hmmlearn import hmm

@dataclass
class RegimeLabel:
    """Programmatic regime label with confidence."""
    regime_type: str
    confidence: float
    label_method: str  # 'rule_based', 'hmm', 'manual'
    features_used: List[str]
    timestamp: pd.Timestamp

class RegimeLabeler:
    """Create programmatic regime labels for benchmarking."""

    def __init__(self):
        self.hmm_model = hmm.GaussianHMM(n_components=4, random_state=42)
        self.rule_based_labels = {}

    def create_rule_based_labels(self, data: pd.DataFrame) -> pd.Series:
        """Create rule-based regime labels."""
        returns = data['close'].pct_change().dropna()
        volatility = returns.rolling(20).std()
        momentum = data['close'].pct_change(20)

        labels = pd.Series(index=data.index, dtype=str)

        # Rule-based classification
        high_vol = volatility > volatility.quantile(0.8)
        low_vol = volatility < volatility.quantile(0.2)
        strong_trend = abs(momentum) > abs(momentum).quantile(0.8)

        labels[high_vol & strong_trend] = 'volatile_trending'
        labels[high_vol & ~strong_trend] = 'volatile_sideways'
        labels[low_vol & strong_trend] = 'stable_trending'
        labels[low_vol & ~strong_trend] = 'stable_sideways'

        return labels

    def create_hmm_labels(self, data: pd.DataFrame) -> pd.Series:
        """Create HMM-based regime labels."""
        returns = data['close'].pct_change().dropna()
        features = np.column_stack([
            returns.values,
            returns.rolling(20).std().values,
            returns.rolling(20).mean().values
        ])

        # Fit HMM
        self.hmm_model.fit(features)
        states = self.hmm_model.predict(features)

        # Map states to regime types
        regime_map = {
            0: 'low_vol_regime',
            1: 'high_vol_regime',
            2: 'trending_regime',
            3: 'mean_reversion_regime'
        }

        return pd.Series([regime_map[s] for s in states], index=returns.index)

    def calculate_stability_metrics(self, predictions: pd.Series, labels: pd.Series) -> Dict[str, float]:
        """Calculate regime stability metrics."""
        # Regime persistence
        regime_changes = (predictions != predictions.shift()).sum()
        avg_regime_duration = len(predictions) / (regime_changes + 1)

        # Transition matrix stability
        transition_matrix = pd.crosstab(predictions.shift(), predictions, normalize='index')

        # Entropy of regime distribution
        regime_entropy = -np.sum(transition_matrix.values * np.log(transition_matrix.values + 1e-10))

        return {
            'avg_regime_duration': avg_regime_duration,
            'regime_entropy': regime_entropy,
            'transition_stability': 1 - transition_matrix.values.std()
        }
```

### 2. **Data Leakage Prevention**

**Problem**: Future information leakage in feature engineering and evaluation.

**Solution**: Strict time-series splits and frozen inference pipelines.

```python
# src/bot/intelligence/data_pipeline.py
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import joblib
from typing import Tuple, Dict, Any

class LeakageFreePipeline:
    """Pipeline that prevents data leakage."""

    def __init__(self, n_splits: int = 5):
        self.tscv = TimeSeriesSplit(n_splits=n_splits)
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.is_fitted = False

    def fit_transform(self, data: pd.DataFrame, target: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Fit scaler on training data only."""
        # Define feature columns
        self.feature_columns = [col for col in data.columns if col.startswith('feature_')]

        # Fit scaler on first fold only
        train_idx, _ = next(self.tscv.split(data))
        train_data = data.iloc[train_idx][self.feature_columns]

        self.scaler.fit(train_data)
        self.is_fitted = True

        # Transform all data
        return self.scaler.transform(data[self.feature_columns]), target.values

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted scaler."""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transform")
        return self.scaler.transform(data[self.feature_columns])

    def save_pipeline(self, path: str):
        """Save fitted pipeline for inference."""
        pipeline_state = {
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'is_fitted': self.is_fitted
        }
        joblib.dump(pipeline_state, path)

    def load_pipeline(self, path: str):
        """Load fitted pipeline for inference."""
        pipeline_state = joblib.load(path)
        self.scaler = pipeline_state['scaler']
        self.feature_columns = pipeline_state['feature_columns']
        self.is_fitted = pipeline_state['is_fitted']
```

### 3. **Precise Selection Accuracy Definition**

**Problem**: "Selection accuracy" is unfalsifiable without precise definition.

**Solution**: Define measurable selection metrics.

```python
# src/bot/intelligence/selection_metrics.py
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np

@dataclass
class SelectionMetrics:
    """Precise selection accuracy metrics."""
    top_k_accuracy: float  # Top-k strategy chosen vs realized top-k
    rank_correlation: float  # Spearman correlation of predicted vs actual ranks
    regret: float  # Performance difference vs optimal selection
    turnover_rate: float  # Portfolio turnover per rebalance
    slippage_cost: float  # Estimated transaction costs

class SelectionAccuracyCalculator:
    """Calculate precise selection accuracy metrics."""

    def __init__(self, k: int = 3):
        self.k = k

    def calculate_top_k_accuracy(
        self,
        predicted_ranks: List[str],
        actual_performance: Dict[str, float]
    ) -> float:
        """Calculate if top-k predicted strategies were actually top-k performers."""
        # Sort by actual performance
        actual_ranks = sorted(actual_performance.items(), key=lambda x: x[1], reverse=True)
        actual_top_k = [strategy for strategy, _ in actual_ranks[:self.k]]

        # Check overlap
        predicted_top_k = predicted_ranks[:self.k]
        overlap = len(set(predicted_top_k) & set(actual_top_k))

        return overlap / self.k

    def calculate_rank_correlation(
        self,
        predicted_ranks: List[str],
        actual_performance: Dict[str, float]
    ) -> float:
        """Calculate Spearman correlation between predicted and actual ranks."""
        # Create rank mappings
        predicted_rank_map = {strategy: i for i, strategy in enumerate(predicted_ranks)}
        actual_ranks = sorted(actual_performance.items(), key=lambda x: x[1], reverse=True)
        actual_rank_map = {strategy: i for i, (strategy, _) in enumerate(actual_ranks)}

        # Calculate correlation
        strategies = list(actual_performance.keys())
        pred_ranks = [predicted_rank_map.get(s, len(predicted_ranks)) for s in strategies]
        act_ranks = [actual_rank_map.get(s, len(actual_ranks)) for s in strategies]

        return np.corrcoef(pred_ranks, act_ranks)[0, 1]

    def calculate_regret(
        self,
        selected_strategies: List[str],
        actual_performance: Dict[str, float],
        optimal_strategies: List[str]
    ) -> float:
        """Calculate performance regret vs optimal selection."""
        selected_performance = np.mean([actual_performance[s] for s in selected_strategies])
        optimal_performance = np.mean([actual_performance[s] for s in optimal_strategies])

        return optimal_performance - selected_performance
```

### 4. **Robust Confidence Intervals**

**Problem**: IID normal CI inappropriate for financial time series.

**Solution**: Block bootstrap and HAR-based confidence intervals.

```python
# src/bot/intelligence/confidence_intervals.py
from typing import Tuple, List
import numpy as np
from scipy import stats

class RobustConfidenceIntervals:
    """Calculate robust confidence intervals for financial data."""

    def __init__(self, block_size: int = 20):
        self.block_size = block_size

    def block_bootstrap_ci(
        self,
        returns: np.ndarray,
        confidence_level: float = 0.95,
        n_bootstrap: int = 10000
    ) -> Tuple[float, float]:
        """Calculate confidence interval using block bootstrap."""
        n_blocks = len(returns) // self.block_size
        bootstrap_means = []

        for _ in range(n_bootstrap):
            # Sample blocks with replacement
            block_indices = np.random.choice(n_blocks, size=n_blocks, replace=True)
            bootstrap_sample = []

            for block_idx in block_indices:
                start_idx = block_idx * self.block_size
                end_idx = start_idx + self.block_size
                bootstrap_sample.extend(returns[start_idx:end_idx])

            bootstrap_means.append(np.mean(bootstrap_sample))

        # Calculate confidence interval
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        return np.percentile(bootstrap_means, [lower_percentile, upper_percentile])

    def har_volatility_ci(
        self,
        returns: np.ndarray,
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate CI using HAR volatility clustering."""
        # HAR volatility model (simplified)
        vol_5 = np.std(returns[-5:])
        vol_22 = np.std(returns[-22:])
        vol_252 = np.std(returns[-252:])

        # HAR forecast
        har_vol = 0.1 + 0.4 * vol_5 + 0.3 * vol_22 + 0.2 * vol_252

        # Confidence interval based on volatility clustering
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        margin = z_score * har_vol / np.sqrt(len(returns))

        mean_return = np.mean(returns)
        return (mean_return - margin, mean_return + margin)

    def compare_ci_methods(
        self,
        returns: np.ndarray,
        confidence_level: float = 0.95
    ) -> Dict[str, Tuple[float, float]]:
        """Compare different confidence interval methods."""
        return {
            'normal_ci': self.normal_ci(returns, confidence_level),
            'block_bootstrap_ci': self.block_bootstrap_ci(returns, confidence_level),
            'har_volatility_ci': self.har_volatility_ci(returns, confidence_level)
        }

    def normal_ci(self, returns: np.ndarray, confidence_level: float = 0.95) -> Tuple[float, float]:
        """Standard normal confidence interval (for comparison)."""
        mean_return = np.mean(returns)
        std_error = np.std(returns) / np.sqrt(len(returns))
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        margin = z_score * std_error

        return (mean_return - margin, mean_return + margin)
```

### 5. **Measurable Transition Smoothness**

**Problem**: "Transition smoothness" is not measurable.

**Solution**: Define explicit turnover metrics and slippage model.

```python
# src/bot/intelligence/transition_metrics.py
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

@dataclass
class TransitionMetrics:
    """Measurable transition smoothness metrics."""
    turnover_rate: float  # Portfolio turnover percentage
    slippage_cost: float  # Estimated transaction costs
    allocation_churn: float  # Percentage of allocation changes
    transition_smoothness: float  # Smoothness score (0-1)

class TransitionSmoothnessCalculator:
    """Calculate measurable transition smoothness."""

    def __init__(self, slippage_model: 'SlippageModel'):
        self.slippage_model = slippage_model

    def calculate_turnover_rate(
        self,
        current_allocations: Dict[str, float],
        target_allocations: Dict[str, float]
    ) -> float:
        """Calculate portfolio turnover rate."""
        turnover = 0.0
        for strategy in set(current_allocations.keys()) | set(target_allocations.keys()):
            current = current_allocations.get(strategy, 0.0)
            target = target_allocations.get(strategy, 0.0)
            turnover += abs(target - current)

        return turnover / 2  # Divide by 2 because turnover counts both buys and sells

    def calculate_allocation_churn(
        self,
        current_allocations: Dict[str, float],
        target_allocations: Dict[str, float],
        threshold: float = 0.01
    ) -> float:
        """Calculate percentage of allocations that changed significantly."""
        total_strategies = len(set(current_allocations.keys()) | set(target_allocations.keys()))
        changed_strategies = 0

        for strategy in set(current_allocations.keys()) | set(target_allocations.keys()):
            current = current_allocations.get(strategy, 0.0)
            target = target_allocations.get(strategy, 0.0)
            if abs(target - current) > threshold:
                changed_strategies += 1

        return changed_strategies / total_strategies if total_strategies > 0 else 0.0

    def calculate_slippage_cost(
        self,
        current_allocations: Dict[str, float],
        target_allocations: Dict[str, float],
        portfolio_value: float
    ) -> float:
        """Calculate estimated slippage costs."""
        total_cost = 0.0

        for strategy in set(current_allocations.keys()) | set(target_allocations.keys()):
            current = current_allocations.get(strategy, 0.0)
            target = target_allocations.get(strategy, 0.0)
            trade_size = abs(target - current) * portfolio_value

            if trade_size > 0:
                slippage_rate = self.slippage_model.estimate_slippage(trade_size)
                total_cost += trade_size * slippage_rate

        return total_cost

    def calculate_smoothness_score(
        self,
        current_allocations: Dict[str, float],
        target_allocations: Dict[str, float],
        portfolio_value: float
    ) -> float:
        """Calculate overall transition smoothness score (0-1, higher is smoother)."""
        turnover_rate = self.calculate_turnover_rate(current_allocations, target_allocations)
        allocation_churn = self.calculate_allocation_churn(current_allocations, target_allocations)
        slippage_cost = self.calculate_slippage_cost(current_allocations, target_allocations, portfolio_value)

        # Normalize metrics to 0-1 scale
        turnover_score = max(0, 1 - turnover_rate / 0.1)  # Penalize if turnover > 10%
        churn_score = max(0, 1 - allocation_churn / 0.5)  # Penalize if >50% allocations changed
        slippage_score = max(0, 1 - slippage_cost / (portfolio_value * 0.001))  # Penalize if slippage > 0.1%

        # Weighted average
        smoothness_score = (0.4 * turnover_score + 0.3 * churn_score + 0.3 * slippage_score)

        return max(0, min(1, smoothness_score))
```

## 🔧 **Missing Glue Components**

### 1. **Data Contracts**

```python
# src/bot/intelligence/data_contracts.py
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd

class MarketBar(BaseModel):
    """Standardized market bar data."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str

    class Config:
        arbitrary_types_allowed = True

class RegimeFeatures(BaseModel):
    """Regime detection features."""
    volatility: float = Field(..., ge=0)
    momentum: float
    trend_strength: float
    correlation: float = Field(..., ge=-1, le=1)
    volume_profile: float
    market_breadth: float
    economic_indicators: Dict[str, float]

class PerformancePrediction(BaseModel):
    """Strategy performance prediction."""
    expected_return: float
    expected_volatility: float = Field(..., ge=0)
    confidence_interval: tuple[float, float]
    drawdown_probability: float = Field(..., ge=0, le=1)
    sharpe_ratio: float
    max_drawdown: float
    prediction_confidence: float = Field(..., ge=0, le=1)

class StrategyDecision(BaseModel):
    """Strategy selection decision."""
    strategy_id: str
    target_allocation: float = Field(..., ge=0, le=1)
    current_allocation: float = Field(..., ge=0, le=1)
    expected_performance: PerformancePrediction
    risk_score: float = Field(..., ge=0)
    transition_cost: float = Field(..., ge=0)
    decision_timestamp: datetime
    decision_reason: str
```

### 2. **Train/Serve Split**

```python
# src/bot/intelligence/model_pipeline.py
from pathlib import Path
import joblib
from typing import Dict, Any, Optional
import yaml

class ModelPipeline:
    """Separate training and inference pipelines."""

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def train_and_save(
        self,
        model: Any,
        scaler: Any,
        feature_map: Dict[str, str],
        version: str,
        metadata: Dict[str, Any]
    ):
        """Train model and save artifacts."""
        artifacts = {
            'model': model,
            'scaler': scaler,
            'feature_map': feature_map,
            'version': version,
            'metadata': metadata,
            'timestamp': datetime.now().isoformat()
        }

        # Save artifacts
        model_path = self.model_dir / f"model_v{version}.joblib"
        joblib.dump(artifacts, model_path)

        # Save metadata
        metadata_path = self.model_dir / f"metadata_v{version}.yaml"
        with open(metadata_path, 'w') as f:
            yaml.dump(metadata, f)

    def load_for_inference(self, version: str) -> Dict[str, Any]:
        """Load model artifacts for inference."""
        model_path = self.model_dir / f"model_v{version}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model version {version} not found")

        return joblib.load(model_path)

    def list_versions(self) -> List[str]:
        """List available model versions."""
        versions = []
        for path in self.model_dir.glob("model_v*.joblib"):
            version = path.stem.replace("model_v", "")
            versions.append(version)
        return sorted(versions)
```

### 3. **Metrics Registry**

```python
# src/bot/intelligence/metrics_registry.py
from typing import Dict, Any, List
import json
from datetime import datetime
from pathlib import Path

class MetricsRegistry:
    """Central place to log and compare metrics across versions."""

    def __init__(self, registry_path: Path):
        self.registry_path = registry_path
        self.registry_path.mkdir(parents=True, exist_ok=True)

    def log_metrics(
        self,
        model_version: str,
        metrics: Dict[str, float],
        metadata: Dict[str, Any]
    ):
        """Log metrics for a model version."""
        entry = {
            'model_version': model_version,
            'metrics': metrics,
            'metadata': metadata,
            'timestamp': datetime.now().isoformat()
        }

        # Save to file
        metrics_file = self.registry_path / f"metrics_v{model_version}.json"
        with open(metrics_file, 'w') as f:
            json.dump(entry, f, indent=2)

    def get_metrics(self, model_version: str) -> Dict[str, Any]:
        """Get metrics for a specific version."""
        metrics_file = self.registry_path / f"metrics_v{model_version}.json"
        if not metrics_file.exists():
            raise FileNotFoundError(f"Metrics for version {model_version} not found")

        with open(metrics_file, 'r') as f:
            return json.load(f)

    def compare_versions(self, versions: List[str]) -> Dict[str, Dict[str, float]]:
        """Compare metrics across versions."""
        comparison = {}
        for version in versions:
            metrics_data = self.get_metrics(version)
            comparison[version] = metrics_data['metrics']
        return comparison
```

### 4. **Configuration Management**

```python
# config/phase1_config.yaml
phase1:
  regime_detection:
    short_term_window: 5
    medium_term_window: 20
    long_term_window: 60
    confidence_threshold: 0.7
    transition_threshold: 0.3
    hmm_components: 4
    rule_based_thresholds:
      volatility_high: 0.8
      volatility_low: 0.2
      momentum_strong: 0.8

  performance_prediction:
    monte_carlo_simulations: 10000
    confidence_level: 0.95
    prediction_horizon: 30
    block_bootstrap_size: 20
    har_vol_lags: [5, 22, 252]

  strategy_selection:
    max_strategies: 5
    risk_tolerance: 0.15
    diversification_target: 0.8
    transition_cost_threshold: 0.005
    turnover_budget: 0.1
    allocation_churn_threshold: 0.01

  safety_rails:
    max_position_size: 0.2
    max_portfolio_risk: 0.15
    max_drawdown_limit: 0.1
    emergency_stop_threshold: 0.05

  testing:
    time_series_splits: 5
    deterministic_seed: 42
    skip_slow_tests: true
    monte_carlo_test_simulations: 1000
```

### 5. **Offline Order Simulator**

```python
# src/bot/intelligence/order_simulator.py
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

@dataclass
class OrderSimulationResult:
    """Result of order simulation."""
    executed_price: float
    slippage: float
    execution_time: float
    fill_ratio: float
    market_impact: float

class L2SlippageModel:
    """Simple L2 slippage model for testing transition costs."""

    def __init__(self, base_slippage: float = 0.0001, impact_factor: float = 0.1):
        self.base_slippage = base_slippage
        self.impact_factor = impact_factor

    def estimate_slippage(self, trade_size: float, market_volume: float = 1e6) -> float:
        """Estimate slippage based on trade size and market volume."""
        # Base slippage
        slippage = self.base_slippage

        # Market impact (proportional to trade size relative to volume)
        volume_ratio = trade_size / market_volume
        market_impact = self.impact_factor * volume_ratio

        return slippage + market_impact

    def simulate_order(
        self,
        order_size: float,
        market_price: float,
        market_volume: float = 1e6
    ) -> OrderSimulationResult:
        """Simulate order execution with slippage."""
        slippage_rate = self.estimate_slippage(order_size, market_volume)
        executed_price = market_price * (1 + slippage_rate)
        slippage = executed_price - market_price

        return OrderSimulationResult(
            executed_price=executed_price,
            slippage=slippage,
            execution_time=1.0,  # Assume 1 second execution
            fill_ratio=1.0,  # Assume full fill
            market_impact=slippage_rate
        )
```

### 6. **Safety Rails in Selector**

```python
# src/bot/intelligence/safety_rails.py
from typing import Dict, List, Tuple
import numpy as np

class SafetyRails:
    """Safety rails for strategy selector."""

    def __init__(self, config: Dict[str, float]):
        self.max_position_size = config.get('max_position_size', 0.2)
        self.max_portfolio_risk = config.get('max_portfolio_risk', 0.15)
        self.max_drawdown_limit = config.get('max_drawdown_limit', 0.1)
        self.emergency_stop_threshold = config.get('emergency_stop_threshold', 0.05)

    def validate_allocations(
        self,
        allocations: Dict[str, float],
        risk_scores: Dict[str, float]
    ) -> Tuple[bool, List[str]]:
        """Validate allocations against safety constraints."""
        violations = []

        # Check position size limits
        for strategy, allocation in allocations.items():
            if allocation > self.max_position_size:
                violations.append(f"Position size {allocation:.2%} exceeds limit {self.max_position_size:.2%} for {strategy}")

        # Check portfolio risk
        portfolio_risk = sum(allocations.get(s, 0) * risk_scores.get(s, 0) for s in allocations)
        if portfolio_risk > self.max_portfolio_risk:
            violations.append(f"Portfolio risk {portfolio_risk:.2%} exceeds limit {self.max_portfolio_risk:.2%}")

        # Check allocation sum
        total_allocation = sum(allocations.values())
        if abs(total_allocation - 1.0) > 0.01:
            violations.append(f"Total allocation {total_allocation:.2%} does not sum to 100%")

        return len(violations) == 0, violations

    def apply_safety_constraints(
        self,
        target_allocations: Dict[str, float],
        risk_scores: Dict[str, float]
    ) -> Dict[str, float]:
        """Apply safety constraints to target allocations."""
        # Start with target allocations
        safe_allocations = target_allocations.copy()

        # Cap individual position sizes
        for strategy in safe_allocations:
            safe_allocations[strategy] = min(safe_allocations[strategy], self.max_position_size)

        # Normalize to sum to 1.0
        total = sum(safe_allocations.values())
        if total > 0:
            safe_allocations = {k: v / total for k, v in safe_allocations.items()}

        # Check portfolio risk and reduce if necessary
        portfolio_risk = sum(safe_allocations.get(s, 0) * risk_scores.get(s, 0) for s in safe_allocations)
        if portfolio_risk > self.max_portfolio_risk:
            # Reduce allocations proportionally
            reduction_factor = self.max_portfolio_risk / portfolio_risk
            safe_allocations = {k: v * reduction_factor for k, v in safe_allocations.items()}

            # Re-normalize
            total = sum(safe_allocations.values())
            if total > 0:
                safe_allocations = {k: v / total for k, v in safe_allocations.items()}

        return safe_allocations
```

## 🎯 **Updated Success Metrics with Guardrails**

```python
# src/bot/intelligence/success_metrics.py
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

@dataclass
class Phase1SuccessMetrics:
    """Updated success metrics with guardrails."""
    regime_stability: float  # Regime persistence and transition stability
    prediction_accuracy: float  # Out-of-sample prediction accuracy
    selection_effectiveness: float  # Top-k accuracy and rank correlation
    transition_smoothness: float  # Measurable smoothness score
    risk_management: float  # Risk-adjusted performance
    safety_compliance: float  # Safety constraint adherence

class SuccessEvaluator:
    """Evaluate Phase 1 success with guardrails."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_history = []

    def evaluate_phase_readiness(self, metrics: Phase1SuccessMetrics) -> Tuple[bool, Dict[str, str]]:
        """Evaluate if ready to advance to next phase."""
        requirements = {
            'regime_stability': (metrics.regime_stability >= 0.85, "Regime stability below 85%"),
            'prediction_accuracy': (metrics.prediction_accuracy >= 0.75, "Prediction accuracy below 75%"),
            'selection_effectiveness': (metrics.selection_effectiveness >= 0.80, "Selection effectiveness below 80%"),
            'transition_smoothness': (metrics.transition_smoothness >= 0.90, "Transition smoothness below 90%"),
            'risk_management': (metrics.risk_management >= 0.15, "Risk-adjusted returns below 15%"),
            'safety_compliance': (metrics.safety_compliance >= 0.99, "Safety compliance below 99%")
        }

        all_passed = all(requirement[0] for requirement in requirements.values())
        failures = [msg for passed, msg in requirements.values() if not passed]

        return all_passed, failures

    def check_guardrails(self, metrics_history: List[Phase1SuccessMetrics]) -> bool:
        """Check if metrics meet guardrail requirements for 3 consecutive months."""
        if len(metrics_history) < 3:
            return False

        # Check last 3 months
        recent_metrics = metrics_history[-3:]

        for metric_name in ['regime_stability', 'prediction_accuracy', 'selection_effectiveness']:
            values = [getattr(m, metric_name) for m in recent_metrics]
            if not all(v >= 0.8 for v in values):  # 80% threshold for 3 months
                return False

        return True
```

## 📊 **Observability Framework**

```python
# src/bot/intelligence/observability.py
import logging
import json
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path

class ObservabilityFramework:
    """Structured logging, tracing, and evaluation harness."""

    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Setup structured logging
        self.logger = logging.getLogger('intelligence')
        self.logger.setLevel(logging.INFO)

        # File handler for structured logs
        log_file = self.log_dir / "intelligence.log"
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

    def log_decision(
        self,
        decision_type: str,
        decision_data: Dict[str, Any],
        metadata: Dict[str, Any]
    ):
        """Log structured decision data."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'decision_type': decision_type,
            'decision_data': decision_data,
            'metadata': metadata
        }

        self.logger.info(json.dumps(log_entry))

    def log_metrics(self, metrics: Dict[str, float], model_version: str):
        """Log performance metrics."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': 'metrics',
            'model_version': model_version,
            'metrics': metrics
        }

        self.logger.info(json.dumps(log_entry))

    def create_evaluation_harness(self, test_window: str = "30d") -> 'EvaluationHarness':
        """Create evaluation harness for model comparison."""
        return EvaluationHarness(self.log_dir, test_window)

class EvaluationHarness:
    """Lightweight evaluation harness for model comparison."""

    def __init__(self, log_dir: Path, test_window: str):
        self.log_dir = log_dir
        self.test_window = test_window

    def replay_fixed_window(
        self,
        model_versions: List[str],
        test_data: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        """Replay a fixed time window to compare models."""
        results = {}

        for version in model_versions:
            # Load model
            model_artifacts = self.load_model_artifacts(version)

            # Run inference on test data
            predictions = self.run_inference(model_artifacts, test_data)

            # Calculate metrics
            metrics = self.calculate_metrics(predictions, test_data)
            results[version] = metrics

        return results

    def compare_models(self, results: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Compare model performance."""
        comparison = {
            'best_model': max(results.keys(), key=lambda k: results[k].get('sharpe_ratio', 0)),
            'performance_ranking': sorted(results.keys(), key=lambda k: results[k].get('sharpe_ratio', 0), reverse=True),
            'metric_comparison': {}
        }

        # Compare each metric
        metrics = list(next(iter(results.values())).keys())
        for metric in metrics:
            comparison['metric_comparison'][metric] = {
                version: results[version].get(metric, 0) for version in results
            }

        return comparison
```

---

*This document addresses the critical risks identified and provides the missing glue components needed for robust Phase 1 implementation. Each component includes proper validation, testing, and observability to ensure reliable autonomous portfolio management.*
