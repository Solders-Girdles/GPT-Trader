# Phase 1 Implementation Guide: Enhanced Intelligence

## 🎯 **Overview**

Phase 1 focuses on enhancing the system's intelligence capabilities to support autonomous decision-making. This phase builds upon the existing foundation and introduces advanced market regime detection, strategy performance prediction, and dynamic strategy selection.

---

## 🏗️ **Architecture Overview**

### 📊 **System Components**
```
Enhanced Intelligence System
├── MultiTimeframeRegimeDetector
│   ├── ShortTermDetector (1-5 days)
│   ├── MediumTermDetector (1-4 weeks)
│   ├── LongTermDetector (1-6 months)
│   └── CrossAssetDetector
├── StrategyPerformancePredictor
│   ├── HistoricalAnalyzer
│   ├── MonteCarloSimulator
│   ├── ConfidenceIntervalCalculator
│   └── RiskPredictor
└── AutonomousStrategySelector
    ├── RegimeAssessor
    ├── PerformanceRanker
    ├── RiskAdjuster
    └── TransitionManager
```

### 🔄 **Data Flow**
1. **Market Data** → **Regime Detection** → **Performance Prediction** → **Strategy Selection**
2. **Historical Data** → **Model Training** → **Performance Validation** → **Model Deployment**
3. **Real-time Data** → **Continuous Monitoring** → **Adaptive Updates** → **Decision Making**

---

## 🧠 **1.1 Advanced Market Regime Detection**

### 📋 **Implementation Plan**

#### **Step 1: Multi-Timeframe Regime Detector**
```python
# src/bot/intelligence/regime_detection.py
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

@dataclass
class RegimeFeatures:
    """Features for regime detection."""
    volatility: float
    momentum: float
    trend_strength: float
    correlation: float
    volume_profile: float
    market_breadth: float
    economic_indicators: Dict[str, float]

@dataclass
class RegimeResult:
    """Regime detection result."""
    regime_type: str
    confidence: float
    features: RegimeFeatures
    transition_probability: float
    expected_duration: int

class MultiTimeframeRegimeDetector:
    """Advanced regime detection with multiple timeframes."""

    def __init__(self, config: Dict[str, any]):
        self.config = config
        self.short_term_model = RandomForestClassifier(n_estimators=100)
        self.medium_term_model = RandomForestClassifier(n_estimators=100)
        self.long_term_model = RandomForestClassifier(n_estimators=100)
        self.scaler = StandardScaler()

    def extract_features(self, data: pd.DataFrame) -> RegimeFeatures:
        """Extract regime detection features from market data."""
        # Volatility features
        returns = data['close'].pct_change().dropna()
        volatility = returns.rolling(20).std().iloc[-1]

        # Momentum features
        momentum = (data['close'].iloc[-1] / data['close'].iloc[-20] - 1)

        # Trend strength
        sma_short = data['close'].rolling(10).mean()
        sma_long = data['close'].rolling(50).mean()
        trend_strength = (sma_short.iloc[-1] - sma_long.iloc[-1]) / sma_long.iloc[-1]

        # Additional features...

        return RegimeFeatures(
            volatility=volatility,
            momentum=momentum,
            trend_strength=trend_strength,
            correlation=0.0,  # Calculate cross-asset correlation
            volume_profile=0.0,  # Calculate volume profile
            market_breadth=0.0,  # Calculate market breadth
            economic_indicators={}  # Add economic indicators
        )

    def detect_regime(self, data: pd.DataFrame, timeframe: str) -> RegimeResult:
        """Detect market regime for specified timeframe."""
        features = self.extract_features(data)

        # Select appropriate model based on timeframe
        if timeframe == "short":
            model = self.short_term_model
        elif timeframe == "medium":
            model = self.medium_term_model
        else:
            model = self.long_term_model

        # Make prediction
        feature_vector = self._features_to_vector(features)
        prediction = model.predict([feature_vector])[0]
        confidence = model.predict_proba([feature_vector]).max()

        return RegimeResult(
            regime_type=prediction,
            confidence=confidence,
            features=features,
            transition_probability=self._calculate_transition_probability(features),
            expected_duration=self._estimate_duration(features)
        )

    def _features_to_vector(self, features: RegimeFeatures) -> List[float]:
        """Convert features to vector for model input."""
        return [
            features.volatility,
            features.momentum,
            features.trend_strength,
            features.correlation,
            features.volume_profile,
            features.market_breadth
        ]

    def _calculate_transition_probability(self, features: RegimeFeatures) -> float:
        """Calculate probability of regime transition."""
        # Implement transition probability calculation
        return 0.1  # Placeholder

    def _estimate_duration(self, features: RegimeFeatures) -> int:
        """Estimate expected regime duration."""
        # Implement duration estimation
        return 30  # Placeholder
```

#### **Step 2: Cross-Asset Correlation Analysis**
```python
# src/bot/intelligence/correlation_analysis.py
class CrossAssetCorrelationAnalyzer:
    """Analyze correlations across multiple assets."""

    def __init__(self, assets: List[str]):
        self.assets = assets
        self.correlation_matrix = None
        self.correlation_regime = None

    def calculate_correlation_matrix(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate correlation matrix for multiple assets."""
        # Extract returns for all assets
        returns_data = {}
        for asset, asset_data in data.items():
            returns_data[asset] = asset_data['close'].pct_change().dropna()

        # Create correlation matrix
        returns_df = pd.DataFrame(returns_data)
        self.correlation_matrix = returns_df.corr()

        return self.correlation_matrix

    def detect_correlation_regime(self) -> str:
        """Detect correlation regime (high/low correlation)."""
        if self.correlation_matrix is None:
            raise ValueError("Correlation matrix not calculated")

        # Calculate average correlation
        avg_correlation = self.correlation_matrix.values[np.triu_indices_from(
            self.correlation_matrix.values, k=1)].mean()

        if avg_correlation > 0.7:
            self.correlation_regime = "high_correlation"
        elif avg_correlation < 0.3:
            self.correlation_regime = "low_correlation"
        else:
            self.correlation_regime = "moderate_correlation"

        return self.correlation_regime
```

#### **Step 3: Economic Indicator Integration**
```python
# src/bot/intelligence/economic_indicators.py
class EconomicIndicatorAnalyzer:
    """Analyze economic indicators for regime detection."""

    def __init__(self):
        self.indicators = {
            'vix': None,  # Volatility index
            'yield_curve': None,  # Yield curve data
            'pmi': None,  # Purchasing Managers Index
            'gdp': None,  # GDP data
            'inflation': None,  # Inflation data
        }

    def update_indicators(self, indicator_data: Dict[str, float]):
        """Update economic indicators."""
        for indicator, value in indicator_data.items():
            if indicator in self.indicators:
                self.indicators[indicator] = value

    def get_regime_signal(self) -> Dict[str, float]:
        """Get regime signals from economic indicators."""
        signals = {}

        # VIX-based signal
        if self.indicators['vix'] is not None:
            if self.indicators['vix'] > 30:
                signals['vix'] = 1.0  # High volatility regime
            elif self.indicators['vix'] < 15:
                signals['vix'] = 0.0  # Low volatility regime
            else:
                signals['vix'] = 0.5  # Moderate volatility regime

        # Yield curve signal
        if self.indicators['yield_curve'] is not None:
            if self.indicators['yield_curve'] < 0:
                signals['yield_curve'] = 1.0  # Inverted yield curve (recession)
            else:
                signals['yield_curve'] = 0.0  # Normal yield curve

        return signals
```

---

## 🎯 **1.2 Strategy Performance Prediction**

### 📋 **Implementation Plan**

#### **Step 1: Historical Performance Analyzer**
```python
# src/bot/intelligence/performance_prediction.py
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit

@dataclass
class PerformancePrediction:
    """Strategy performance prediction result."""
    expected_return: float
    expected_volatility: float
    confidence_interval: Tuple[float, float]
    drawdown_probability: float
    sharpe_ratio: float
    max_drawdown: float

class StrategyPerformancePredictor:
    """Predict strategy performance in different market conditions."""

    def __init__(self):
        self.return_model = GradientBoostingRegressor()
        self.volatility_model = GradientBoostingRegressor()
        self.drawdown_model = GradientBoostingRegressor()
        self.regime_performance_history = {}

    def analyze_historical_performance(
        self,
        strategy_results: pd.DataFrame,
        regime_data: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        """Analyze historical performance by regime."""
        # Merge strategy results with regime data
        combined_data = strategy_results.merge(
            regime_data, left_index=True, right_index=True, how='inner'
        )

        # Group by regime and calculate performance metrics
        regime_performance = {}
        for regime in combined_data['regime'].unique():
            regime_data = combined_data[combined_data['regime'] == regime]

            regime_performance[regime] = {
                'avg_return': regime_data['returns'].mean(),
                'avg_volatility': regime_data['returns'].std(),
                'avg_sharpe': regime_data['sharpe_ratio'].mean(),
                'avg_drawdown': regime_data['max_drawdown'].mean(),
                'win_rate': (regime_data['returns'] > 0).mean(),
                'sample_size': len(regime_data)
            }

        self.regime_performance_history = regime_performance
        return regime_performance

    def predict_performance(
        self,
        strategy_features: Dict[str, float],
        regime_features: Dict[str, float]
    ) -> PerformancePrediction:
        """Predict strategy performance given features and regime."""
        # Combine strategy and regime features
        combined_features = {**strategy_features, **regime_features}
        feature_vector = list(combined_features.values())

        # Make predictions
        expected_return = self.return_model.predict([feature_vector])[0]
        expected_volatility = self.volatility_model.predict([feature_vector])[0]
        expected_drawdown = self.drawdown_model.predict([feature_vector])[0]

        # Calculate confidence intervals
        confidence_interval = self._calculate_confidence_interval(
            expected_return, expected_volatility
        )

        # Calculate drawdown probability
        drawdown_probability = self._calculate_drawdown_probability(
            expected_return, expected_volatility
        )

        return PerformancePrediction(
            expected_return=expected_return,
            expected_volatility=expected_volatility,
            confidence_interval=confidence_interval,
            drawdown_probability=drawdown_probability,
            sharpe_ratio=expected_return / expected_volatility if expected_volatility > 0 else 0,
            max_drawdown=expected_drawdown
        )

    def _calculate_confidence_interval(
        self,
        expected_return: float,
        expected_volatility: float
    ) -> Tuple[float, float]:
        """Calculate confidence interval for return prediction."""
        # 95% confidence interval
        margin_of_error = 1.96 * expected_volatility / np.sqrt(252)  # Daily data
        return (expected_return - margin_of_error, expected_return + margin_of_error)

    def _calculate_drawdown_probability(
        self,
        expected_return: float,
        expected_volatility: float
    ) -> float:
        """Calculate probability of significant drawdown."""
        # Simplified calculation - can be enhanced with more sophisticated models
        if expected_volatility == 0:
            return 0.0

        # Probability of 10% drawdown
        z_score = -0.1 / expected_volatility
        return 1 - norm.cdf(z_score)
```

#### **Step 2: Monte Carlo Simulator**
```python
# src/bot/intelligence/monte_carlo.py
class MonteCarloSimulator:
    """Monte Carlo simulation for strategy performance."""

    def __init__(self, n_simulations: int = 10000):
        self.n_simulations = n_simulations

    def simulate_strategy_performance(
        self,
        initial_capital: float,
        expected_return: float,
        expected_volatility: float,
        time_horizon: int,
        regime_transitions: List[Dict[str, float]] = None
    ) -> Dict[str, List[float]]:
        """Simulate strategy performance using Monte Carlo."""
        results = {
            'capital_paths': [],
            'return_paths': [],
            'drawdown_paths': []
        }

        for _ in range(self.n_simulations):
            # Generate random returns
            returns = np.random.normal(
                expected_return / 252,  # Daily return
                expected_volatility / np.sqrt(252),  # Daily volatility
                time_horizon
            )

            # Calculate capital path
            capital_path = [initial_capital]
            for ret in returns:
                capital_path.append(capital_path[-1] * (1 + ret))

            # Calculate drawdown path
            peak = initial_capital
            drawdown_path = []
            for capital in capital_path:
                if capital > peak:
                    peak = capital
                drawdown = (capital - peak) / peak
                drawdown_path.append(drawdown)

            results['capital_paths'].append(capital_path)
            results['return_paths'].append(returns)
            results['drawdown_paths'].append(drawdown_path)

        return results

    def calculate_performance_metrics(
        self,
        simulation_results: Dict[str, List[float]]
    ) -> Dict[str, float]:
        """Calculate performance metrics from simulation results."""
        final_capitals = [path[-1] for path in simulation_results['capital_paths']]
        max_drawdowns = [min(path) for path in simulation_results['drawdown_paths']]

        return {
            'expected_final_capital': np.mean(final_capitals),
            'capital_std': np.std(final_capitals),
            'expected_max_drawdown': np.mean(max_drawdowns),
            'drawdown_std': np.std(max_drawdowns),
            'probability_of_loss': np.mean([c < simulation_results['capital_paths'][0][0] for c in final_capitals]),
            'probability_of_10pct_drawdown': np.mean([d < -0.1 for d in max_drawdowns])
        }
```

---

## 🔄 **1.3 Dynamic Strategy Selection**

### 📋 **Implementation Plan**

#### **Step 1: Autonomous Strategy Selector**
```python
# src/bot/intelligence/strategy_selector.py
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class StrategyCandidate:
    """Strategy candidate for selection."""
    strategy_id: str
    strategy_name: str
    expected_performance: PerformancePrediction
    current_allocation: float
    risk_score: float
    diversification_score: float
    transition_cost: float

@dataclass
class SelectionResult:
    """Strategy selection result."""
    selected_strategies: List[StrategyCandidate]
    target_allocations: Dict[str, float]
    expected_portfolio_return: float
    expected_portfolio_volatility: float
    expected_portfolio_sharpe: float
    transition_plan: List[Dict[str, any]]

class AutonomousStrategySelector:
    """Autonomously select optimal strategies based on current conditions."""

    def __init__(self, config: Dict[str, any]):
        self.config = config
        self.regime_detector = MultiTimeframeRegimeDetector(config)
        self.performance_predictor = StrategyPerformancePredictor()
        self.current_regime = None
        self.strategy_candidates = []

    def update_market_conditions(self, market_data: pd.DataFrame):
        """Update market conditions and regime detection."""
        self.current_regime = self.regime_detector.detect_regime(
            market_data, timeframe="medium"
        )

    def evaluate_strategies(
        self,
        strategies: List[Dict[str, any]],
        market_data: pd.DataFrame
    ) -> List[StrategyCandidate]:
        """Evaluate all available strategies."""
        candidates = []

        for strategy in strategies:
            # Predict performance for current regime
            strategy_features = self._extract_strategy_features(strategy)
            regime_features = self._extract_regime_features(self.current_regime)

            performance = self.performance_predictor.predict_performance(
                strategy_features, regime_features
            )

            # Calculate risk and diversification scores
            risk_score = self._calculate_risk_score(strategy, performance)
            diversification_score = self._calculate_diversification_score(
                strategy, candidates
            )

            # Calculate transition cost
            transition_cost = self._calculate_transition_cost(strategy)

            candidate = StrategyCandidate(
                strategy_id=strategy['id'],
                strategy_name=strategy['name'],
                expected_performance=performance,
                current_allocation=strategy.get('current_allocation', 0.0),
                risk_score=risk_score,
                diversification_score=diversification_score,
                transition_cost=transition_cost
            )

            candidates.append(candidate)

        self.strategy_candidates = candidates
        return candidates

    def select_optimal_strategies(
        self,
        target_risk: float,
        max_strategies: int = 5
    ) -> SelectionResult:
        """Select optimal strategies based on objectives."""
        # Sort candidates by risk-adjusted return
        sorted_candidates = sorted(
            self.strategy_candidates,
            key=lambda x: x.expected_performance.sharpe_ratio,
            reverse=True
        )

        # Apply risk and diversification constraints
        selected_candidates = []
        total_risk = 0.0

        for candidate in sorted_candidates:
            if len(selected_candidates) >= max_strategies:
                break

            # Check risk constraint
            candidate_risk = candidate.risk_score * candidate.expected_performance.expected_volatility
            if total_risk + candidate_risk <= target_risk:
                selected_candidates.append(candidate)
                total_risk += candidate_risk

        # Calculate optimal allocations
        target_allocations = self._calculate_optimal_allocations(
            selected_candidates, target_risk
        )

        # Calculate portfolio metrics
        portfolio_metrics = self._calculate_portfolio_metrics(
            selected_candidates, target_allocations
        )

        # Generate transition plan
        transition_plan = self._generate_transition_plan(
            selected_candidates, target_allocations
        )

        return SelectionResult(
            selected_strategies=selected_candidates,
            target_allocations=target_allocations,
            expected_portfolio_return=portfolio_metrics['return'],
            expected_portfolio_volatility=portfolio_metrics['volatility'],
            expected_portfolio_sharpe=portfolio_metrics['sharpe'],
            transition_plan=transition_plan
        )

    def _extract_strategy_features(self, strategy: Dict[str, any]) -> Dict[str, float]:
        """Extract features from strategy for performance prediction."""
        return {
            'lookback_period': strategy.get('lookback_period', 20),
            'threshold': strategy.get('threshold', 0.1),
            'position_size': strategy.get('position_size', 0.1),
            'stop_loss': strategy.get('stop_loss', 0.05),
            'take_profit': strategy.get('take_profit', 0.1),
            'strategy_type': self._encode_strategy_type(strategy.get('type', 'trend'))
        }

    def _extract_regime_features(self, regime: RegimeResult) -> Dict[str, float]:
        """Extract features from regime for performance prediction."""
        return {
            'regime_volatility': regime.features.volatility,
            'regime_momentum': regime.features.momentum,
            'regime_trend': regime.features.trend_strength,
            'regime_correlation': regime.features.correlation,
            'regime_confidence': regime.confidence,
            'transition_probability': regime.transition_probability
        }

    def _calculate_risk_score(self, strategy: Dict[str, any], performance: PerformancePrediction) -> float:
        """Calculate risk score for strategy."""
        # Combine volatility, drawdown probability, and strategy-specific risk factors
        volatility_risk = performance.expected_volatility
        drawdown_risk = performance.drawdown_probability
        strategy_risk = strategy.get('risk_factor', 1.0)

        return (volatility_risk + drawdown_risk) * strategy_risk

    def _calculate_diversification_score(self, strategy: Dict[str, any], existing_candidates: List[StrategyCandidate]) -> float:
        """Calculate diversification score for strategy."""
        if not existing_candidates:
            return 1.0  # First strategy gets full diversification score

        # Calculate correlation with existing strategies
        correlations = []
        for candidate in existing_candidates:
            correlation = self._calculate_strategy_correlation(strategy, candidate)
            correlations.append(abs(correlation))

        # Lower average correlation = higher diversification score
        avg_correlation = np.mean(correlations) if correlations else 0.0
        return 1.0 - avg_correlation

    def _calculate_strategy_correlation(self, strategy1: Dict[str, any], strategy2: StrategyCandidate) -> float:
        """Calculate correlation between two strategies."""
        # Simplified correlation calculation
        # In practice, this would use historical performance data
        return 0.1  # Placeholder

    def _calculate_transition_cost(self, strategy: Dict[str, any]) -> float:
        """Calculate cost of transitioning to strategy."""
        # Consider transaction costs, slippage, and market impact
        base_cost = 0.001  # 0.1% base transaction cost
        market_impact = strategy.get('market_impact', 0.0005)
        return base_cost + market_impact

    def _calculate_optimal_allocations(
        self,
        candidates: List[StrategyCandidate],
        target_risk: float
    ) -> Dict[str, float]:
        """Calculate optimal allocations using risk parity or similar approach."""
        # Simple equal allocation for now
        # Can be enhanced with risk parity, mean-variance optimization, etc.
        n_strategies = len(candidates)
        if n_strategies == 0:
            return {}

        equal_allocation = 1.0 / n_strategies
        return {candidate.strategy_id: equal_allocation for candidate in candidates}

    def _calculate_portfolio_metrics(
        self,
        candidates: List[StrategyCandidate],
        allocations: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate portfolio-level performance metrics."""
        total_return = 0.0
        total_volatility = 0.0

        for candidate in candidates:
            allocation = allocations.get(candidate.strategy_id, 0.0)
            total_return += allocation * candidate.expected_performance.expected_return
            total_volatility += allocation * candidate.expected_performance.expected_volatility

        return {
            'return': total_return,
            'volatility': total_volatility,
            'sharpe': total_return / total_volatility if total_volatility > 0 else 0
        }

    def _generate_transition_plan(
        self,
        candidates: List[StrategyCandidate],
        target_allocations: Dict[str, float]
    ) -> List[Dict[str, any]]:
        """Generate plan for transitioning to new allocations."""
        transition_plan = []

        for candidate in candidates:
            current_allocation = candidate.current_allocation
            target_allocation = target_allocations.get(candidate.strategy_id, 0.0)

            if abs(target_allocation - current_allocation) > 0.01:  # 1% threshold
                transition_plan.append({
                    'strategy_id': candidate.strategy_id,
                    'action': 'increase' if target_allocation > current_allocation else 'decrease',
                    'current_allocation': current_allocation,
                    'target_allocation': target_allocation,
                    'change_amount': target_allocation - current_allocation,
                    'estimated_cost': candidate.transition_cost * abs(target_allocation - current_allocation)
                })

        return transition_plan
```

---

### 🧩 Phase 1 Intelligence Toolkit (Scaffolding)

The following utilities are provided under `src/bot/intelligence/` and can be consumed directly or via a simple façade:

```python
from pathlib import Path
from bot.intelligence.facade import Phase1IntelligenceToolkit

toolkit = Phase1IntelligenceToolkit(block_size=20, top_k=3)

# Regime labels
rule_labels = toolkit.rule_based_regimes(market_df)

# Selection metrics
topk = toolkit.selection_top_k_accuracy(predicted_ranks, actual_performance)
rho  = toolkit.selection_rank_corr(predicted_ranks, actual_performance)
reg  = toolkit.selection_regret(selected, actual_performance, optimal)

# Transition smoothness
smooth = toolkit.transition_smoothness(curr_allocs, target_allocs, portfolio_value=100000)

# Confidence intervals
cis = toolkit.ci_compare(returns_array)

# Metrics registry
registry = toolkit.metrics_registry(Path("logs/metrics"))
registry.log_metrics("v1", {"sharpe": 1.2}, {"note": "baseline"})
```

These are dependency-light and safe to integrate incrementally. The orchestrator includes initial wiring for safety rails, transition smoothness, and observability.

---

## 🧪 **Testing Strategy**

### 📋 **Test Implementation**

#### **Unit Tests**
```python
# tests/intelligence/test_regime_detection.py
import pytest
import pandas as pd
import numpy as np
from bot.intelligence.regime_detection import MultiTimeframeRegimeDetector, RegimeFeatures

class TestMultiTimeframeRegimeDetector:
    """Test regime detection functionality."""

    @pytest.fixture
    def sample_data(self):
        """Create sample market data."""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        prices = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, 100)))

        return pd.DataFrame({
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)

    @pytest.fixture
    def detector(self):
        """Create regime detector instance."""
        config = {
            'short_term_window': 5,
            'medium_term_window': 20,
            'long_term_window': 60
        }
        return MultiTimeframeRegimeDetector(config)

    def test_feature_extraction(self, detector, sample_data):
        """Test feature extraction from market data."""
        features = detector.extract_features(sample_data)

        assert isinstance(features, RegimeFeatures)
        assert features.volatility > 0
        assert isinstance(features.momentum, float)
        assert isinstance(features.trend_strength, float)

    def test_regime_detection(self, detector, sample_data):
        """Test regime detection for different timeframes."""
        for timeframe in ['short', 'medium', 'long']:
            result = detector.detect_regime(sample_data, timeframe)

            assert result.regime_type in ['trending', 'sideways', 'volatile', 'crisis']
            assert 0 <= result.confidence <= 1
            assert result.expected_duration > 0
```

#### **Integration Tests**
```python
# tests/intelligence/test_strategy_selector.py
import pytest
from bot.intelligence.strategy_selector import AutonomousStrategySelector

class TestAutonomousStrategySelector:
    """Test autonomous strategy selection."""

    @pytest.fixture
    def selector(self):
        """Create strategy selector instance."""
        config = {
            'max_strategies': 5,
            'risk_tolerance': 0.15,
            'diversification_target': 0.8
        }
        return AutonomousStrategySelector(config)

    def test_strategy_evaluation(self, selector, sample_market_data, sample_strategies):
        """Test strategy evaluation process."""
        # Update market conditions
        selector.update_market_conditions(sample_market_data)

        # Evaluate strategies
        candidates = selector.evaluate_strategies(sample_strategies, sample_market_data)

        assert len(candidates) == len(sample_strategies)
        for candidate in candidates:
            assert candidate.expected_performance.expected_return is not None
            assert candidate.risk_score >= 0
            assert candidate.diversification_score >= 0

    def test_strategy_selection(self, selector, sample_market_data, sample_strategies):
        """Test optimal strategy selection."""
        # Setup
        selector.update_market_conditions(sample_market_data)
        selector.evaluate_strategies(sample_strategies, sample_market_data)

        # Select strategies
        result = selector.select_optimal_strategies(target_risk=0.15)

        assert len(result.selected_strategies) <= 5
        assert sum(result.target_allocations.values()) == pytest.approx(1.0, abs=0.01)
        assert result.expected_portfolio_sharpe > 0
```

---

## 📊 **Performance Monitoring**

### 📋 **Monitoring Implementation**

#### **Performance Metrics**
```python
# src/bot/intelligence/monitoring.py
class IntelligencePerformanceMonitor:
    """Monitor performance of intelligence components."""

    def __init__(self):
        self.regime_accuracy_history = []
        self.prediction_accuracy_history = []
        self.selection_performance_history = []

    def track_regime_accuracy(self, predicted_regime: str, actual_regime: str):
        """Track regime detection accuracy."""
        accuracy = 1.0 if predicted_regime == actual_regime else 0.0
        self.regime_accuracy_history.append(accuracy)

    def track_prediction_accuracy(self, predicted_return: float, actual_return: float):
        """Track performance prediction accuracy."""
        error = abs(predicted_return - actual_return)
        self.prediction_accuracy_history.append(error)

    def track_selection_performance(self, selected_strategies: List[str], actual_performance: float):
        """Track strategy selection performance."""
        self.selection_performance_history.append({
            'strategies': selected_strategies,
            'performance': actual_performance
        })

    def get_performance_summary(self) -> Dict[str, float]:
        """Get performance summary."""
        return {
            'regime_accuracy': np.mean(self.regime_accuracy_history) if self.regime_accuracy_history else 0.0,
            'prediction_error': np.mean(self.prediction_accuracy_history) if self.prediction_accuracy_history else 0.0,
            'selection_performance': np.mean([p['performance'] for p in self.selection_performance_history]) if self.selection_performance_history else 0.0
        }
```

---

## 🚀 **Deployment Strategy**

### 📋 **Deployment Steps**

1. **Phase 1.1 Deployment** (Week 1-2)
   - Deploy MultiTimeframeRegimeDetector
   - Implement basic regime detection
   - Add monitoring and logging

2. **Phase 1.2 Deployment** (Week 3-4)
   - Deploy StrategyPerformancePredictor
   - Implement performance prediction
   - Add Monte Carlo simulation

3. **Phase 1.3 Deployment** (Week 5-6)
   - Deploy AutonomousStrategySelector
   - Implement strategy selection
   - Add transition management

4. **Integration and Testing** (Week 7-8)
   - End-to-end testing
   - Performance validation
   - Documentation updates

### 🔧 **Configuration Management**

```python
# config/intelligence_config.yaml
intelligence:
  regime_detection:
    short_term_window: 5
    medium_term_window: 20
    long_term_window: 60
    confidence_threshold: 0.7
    transition_threshold: 0.3

  performance_prediction:
    monte_carlo_simulations: 10000
    confidence_level: 0.95
    prediction_horizon: 30

  strategy_selection:
    max_strategies: 5
    risk_tolerance: 0.15
    diversification_target: 0.8
    transition_cost_threshold: 0.005
```

---

## 📈 **Success Metrics**

### 🎯 **Phase 1 Success Criteria**

1. **Regime Detection**
   - Classification accuracy > 85%
   - Transition prediction lead time > 3 days
   - False positive rate < 15%

2. **Performance Prediction**
   - Prediction accuracy > 75%
   - Risk prediction accuracy > 80%
   - Confidence interval reliability > 90%

3. **Strategy Selection**
   - Selection accuracy > 80%
   - Portfolio performance improvement > 10%
   - Strategy transition smoothness > 90%

### 📊 **Monitoring Dashboard**

Create a monitoring dashboard to track:
- Real-time regime detection results
- Performance prediction accuracy
- Strategy selection decisions
- Portfolio performance metrics
- System health and reliability

---

*This implementation guide provides a detailed roadmap for Phase 1 of the autonomous portfolio management system. Each component is designed to be modular, testable, and extensible for future phases.*
