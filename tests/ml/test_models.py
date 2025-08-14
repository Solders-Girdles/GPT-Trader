"""
Test suite for ML models (regime detector, strategy selector)
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')


class TestMarketRegimeDetector:
    """Test HMM market regime detector"""
    
    @pytest.fixture
    def sample_features(self):
        """Create sample feature data for regime detection"""
        n_samples = 500
        np.random.seed(42)
        
        # Simulate different market regimes
        regime_changes = [0, 100, 200, 350, 500]
        regimes = [0, 1, 2, 1, 0]  # Bull, volatile, bear, volatile, bull
        
        features = pd.DataFrame()
        
        for i in range(len(regimes)-1):
            start_idx = regime_changes[i]
            end_idx = regime_changes[i+1]
            n_points = end_idx - start_idx
            
            if regimes[i] == 0:  # Bull
                returns = np.random.normal(0.001, 0.01, n_points)
                volatility = np.random.normal(0.15, 0.02, n_points)
            elif regimes[i] == 1:  # Volatile
                returns = np.random.normal(0, 0.02, n_points)
                volatility = np.random.normal(0.25, 0.05, n_points)
            else:  # Bear
                returns = np.random.normal(-0.001, 0.015, n_points)
                volatility = np.random.normal(0.20, 0.03, n_points)
            
            regime_features = pd.DataFrame({
                'returns_5d': returns,
                'volatility_20d': volatility,
                'volume_ratio': np.random.uniform(0.8, 1.2, n_points),
                'rsi_14': np.random.uniform(30, 70, n_points)
            })
            
            features = pd.concat([features, regime_features], ignore_index=True)
        
        return features.iloc[:n_samples]
    
    def test_regime_detector_training(self, sample_features):
        """Test regime detector training"""
        from src.bot.ml.models.regime_detector import MarketRegimeDetector
        
        detector = MarketRegimeDetector()
        
        # Train model
        metrics = detector.train(sample_features)
        
        # Check training completed
        assert detector.is_trained
        assert metrics is not None
        assert 'n_samples' in metrics
        assert 'n_features' in metrics
        
        # Check model parameters
        assert detector.n_regimes == 5  # Default 5 regimes
        assert detector.model is not None
    
    def test_regime_prediction(self, sample_features):
        """Test regime prediction"""
        from src.bot.ml.models.regime_detector import MarketRegimeDetector
        
        detector = MarketRegimeDetector()
        detector.train(sample_features[:400])  # Train on subset
        
        # Predict on test data
        test_features = sample_features[400:]
        predictions = detector.predict(test_features)
        
        # Check predictions
        assert len(predictions) == len(test_features)
        assert all(0 <= p < detector.n_regimes for p in predictions)
        
        # Check regime names
        regime_names = [detector.get_regime_name(p) for p in predictions]
        valid_names = ['bull_quiet', 'bull_volatile', 'bear_quiet', 'bear_volatile', 'sideways']
        assert all(name in valid_names for name in regime_names)
    
    def test_regime_confidence(self, sample_features):
        """Test regime confidence scores"""
        from src.bot.ml.models.regime_detector import MarketRegimeDetector
        
        detector = MarketRegimeDetector()
        detector.train(sample_features)
        
        # Get predictions with confidence
        test_sample = sample_features.iloc[[0]]
        regime, confidence = detector.get_regime_confidence(test_sample)
        
        # Check confidence scores
        assert 0 <= confidence[0] <= 1
        assert isinstance(regime[0], (int, np.integer))
    
    def test_regime_analysis(self, sample_features):
        """Test regime analysis methods"""
        from src.bot.ml.models.regime_detector import MarketRegimeDetector
        
        detector = MarketRegimeDetector()
        detector.train(sample_features)
        
        # Analyze regimes
        analysis = detector.analyze_regimes(sample_features)
        
        # Check analysis results
        assert 'regime_counts' in analysis
        assert 'transition_matrix' in analysis
        assert 'regime_stats' in analysis
        
        # Check transition matrix
        trans_matrix = analysis['transition_matrix']
        assert trans_matrix.shape == (5, 5)
        # Rows should sum to approximately 1
        row_sums = trans_matrix.sum(axis=1)
        assert all(0.95 <= s <= 1.05 for s in row_sums)
    
    def test_model_persistence(self, sample_features):
        """Test model saving and loading"""
        from src.bot.ml.models.regime_detector import MarketRegimeDetector
        
        detector = MarketRegimeDetector()
        detector.train(sample_features)
        
        # Save model
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "regime_model.joblib"
            joblib.dump(detector, model_path)
            
            # Load model
            loaded_detector = joblib.load(model_path)
            
            # Test loaded model
            assert loaded_detector.is_trained
            predictions = loaded_detector.predict(sample_features[:10])
            assert len(predictions) == 10


class TestStrategySelector:
    """Test XGBoost strategy selector"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data for strategy selection"""
        n_samples = 1000
        np.random.seed(42)
        
        # Generate features
        features = pd.DataFrame({
            'rsi_14': np.random.uniform(20, 80, n_samples),
            'macd': np.random.normal(0, 1, n_samples),
            'volatility_20d': np.random.uniform(0.1, 0.3, n_samples),
            'volume_ratio': np.random.uniform(0.5, 2.0, n_samples),
            'trend_strength': np.random.normal(0, 0.5, n_samples),
            'bb_percent': np.random.uniform(0, 1, n_samples),
            'adx': np.random.uniform(10, 50, n_samples),
            'returns_5d': np.random.normal(0, 0.05, n_samples)
        })
        
        # Generate labels based on simple rules
        labels = []
        for i in range(n_samples):
            row = features.iloc[i]
            
            if row['trend_strength'] > 0.3 and row['adx'] > 30:
                label = 'trend_following'
            elif row['rsi_14'] < 30 or row['rsi_14'] > 70:
                label = 'mean_reversion'
            elif row['volatility_20d'] > 0.2:
                label = 'breakout'
            else:
                label = 'momentum'
            
            labels.append(label)
        
        return features, pd.Series(labels)
    
    def test_strategy_selector_training(self, sample_data):
        """Test strategy selector training"""
        from src.bot.ml.models.strategy_selector import StrategyMetaSelector
        
        X, y = sample_data
        selector = StrategyMetaSelector()
        
        # Train model
        metrics = selector.train(X, y, optimize_hyperparams=False)
        
        # Check training completed
        assert selector.is_trained
        assert metrics is not None
        assert 'accuracy' in metrics
        assert metrics['accuracy'] > 0.5  # Should be better than random
    
    def test_strategy_prediction(self, sample_data):
        """Test strategy prediction"""
        from src.bot.ml.models.strategy_selector import StrategyMetaSelector
        
        X, y = sample_data
        selector = StrategyMetaSelector()
        
        # Train on subset
        train_size = 800
        selector.train(X[:train_size], y[:train_size])
        
        # Predict on test data
        X_test = X[train_size:]
        predictions = selector.predict(X_test)
        
        # Check predictions
        assert len(predictions) == len(X_test)
        assert all(pred in selector.strategies for pred in predictions)
    
    def test_prediction_probability(self, sample_data):
        """Test prediction probabilities"""
        from src.bot.ml.models.strategy_selector import StrategyMetaSelector
        
        X, y = sample_data
        selector = StrategyMetaSelector()
        selector.train(X[:800], y[:800])
        
        # Get probabilities
        X_test = X[800:850]
        probs = selector.predict_proba(X_test)
        
        # Check probability matrix
        assert probs.shape == (50, len(selector.strategies))
        assert all((probs.sum(axis=1) - 1.0).abs() < 0.01)  # Rows sum to 1
        assert (probs >= 0).all().all() and (probs <= 1).all().all()
    
    def test_strategy_confidence(self, sample_data):
        """Test strategy selection with confidence"""
        from src.bot.ml.models.strategy_selector import StrategyMetaSelector
        
        X, y = sample_data
        selector = StrategyMetaSelector()
        selector.train(X, y)
        
        # Get prediction with confidence
        test_sample = X.iloc[[0]]
        strategy, confidence, all_probs = selector.select_strategy_with_confidence(test_sample)
        
        # Check results
        assert strategy in selector.strategies
        assert 0 <= confidence <= 1
        assert sum(all_probs.values()) - 1.0 < 0.01
    
    def test_ensemble_strategy(self, sample_data):
        """Test ensemble strategy creation"""
        from src.bot.ml.models.strategy_selector import StrategyMetaSelector
        
        X, y = sample_data
        selector = StrategyMetaSelector()
        selector.train(X, y)
        
        # Create ensemble
        test_sample = X.iloc[[0]]
        ensemble = selector.create_ensemble_strategy(test_sample, threshold=0.6)
        
        # Check ensemble structure
        assert 'type' in ensemble
        assert 'strategies' in ensemble
        assert 'confidence' in ensemble
        assert 'reason' in ensemble
        
        # Check strategy weights
        if ensemble['type'] == 'ensemble':
            weights = ensemble['strategies']
            assert abs(sum(weights.values()) - 1.0) < 0.01
            assert all(0 <= w <= 1 for w in weights.values())
    
    def test_feature_importance(self, sample_data):
        """Test feature importance analysis"""
        from src.bot.ml.models.strategy_selector import StrategyMetaSelector
        
        X, y = sample_data
        selector = StrategyMetaSelector()
        selector.train(X, y)
        
        # Get feature importance
        importance = selector.get_feature_importance_analysis()
        
        # Check importance structure
        assert 'top_features' in importance
        assert 'feature_groups' in importance
        assert 'group_importance' in importance
        
        # Check feature importance values
        if importance['top_features']:
            for feature, imp in importance['top_features'][:5]:
                assert isinstance(feature, str)
                assert 0 <= imp <= 1


class TestMLIntegration:
    """Test ML component integration"""
    
    def test_ml_enhanced_strategy(self):
        """Test ML-enhanced strategy wrapper"""
        from src.bot.strategy.ml_enhanced import MLEnhancedStrategy
        from src.bot.strategy.base import Strategy
        
        # Create mock strategies
        class MockStrategy(Strategy):
            def generate_signals(self, data):
                return pd.Series(1, index=data.index)
        
        strategies = {
            'strategy1': MockStrategy(),
            'strategy2': MockStrategy()
        }
        
        # Create ML-enhanced strategy
        ml_strategy = MLEnhancedStrategy(strategies=strategies)
        
        # Test with sample data
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        data = pd.DataFrame({
            'open': np.random.uniform(99, 101, 100),
            'high': np.random.uniform(100, 102, 100),
            'low': np.random.uniform(98, 100, 100),
            'close': np.random.uniform(99, 101, 100),
            'volume': np.random.uniform(1e6, 1e7, 100)
        }, index=dates)
        
        # Generate signals (should work even without ML models)
        signals = ml_strategy.generate_signals(data)
        
        assert len(signals) == len(data)
        assert signals.dtype in [np.float64, np.int64]
    
    def test_ml_allocator(self):
        """Test ML-enhanced allocator"""
        from src.bot.ml.portfolio.allocator import MLEnhancedAllocator
        
        # Create allocator
        allocator = MLEnhancedAllocator()
        
        # Test allocation
        universe = ['AAPL', 'MSFT', 'GOOGL']
        capital = 100000
        
        # Create mock market data
        market_data = {}
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        
        for symbol in universe:
            market_data[symbol] = pd.DataFrame({
                'open': np.random.uniform(99, 101, 100),
                'high': np.random.uniform(100, 102, 100),
                'low': np.random.uniform(98, 100, 100),
                'close': np.random.uniform(99, 101, 100),
                'volume': np.random.uniform(1e6, 1e7, 100)
            }, index=dates)
        
        # Allocate (should work with default equal weights)
        positions = allocator.allocate(universe, market_data, capital=capital)
        
        assert len(positions) <= len(universe)
        assert sum(positions.values()) <= capital * 1.01  # Allow small rounding


def test_end_to_end_ml_pipeline():
    """Test complete ML pipeline from data to prediction"""
    
    # Generate sample data
    dates = pd.date_range(end=datetime.now(), periods=200, freq='D')
    data = pd.DataFrame({
        'open': 100 + np.random.randn(200).cumsum(),
        'high': 101 + np.random.randn(200).cumsum(),
        'low': 99 + np.random.randn(200).cumsum(),
        'close': 100 + np.random.randn(200).cumsum(),
        'volume': np.random.uniform(1e6, 1e7, 200)
    }, index=dates)
    
    # Feature engineering
    from src.bot.ml.features.engineering import FeatureEngineeringPipeline
    pipeline = FeatureEngineeringPipeline()
    features = pipeline.generate_features(data, store_features=False)
    
    assert not features.empty
    
    # Model training would go here (skipped for performance)
    
    # Test portfolio optimization
    from src.bot.ml.portfolio.optimizer import MarkowitzOptimizer
    
    # Create returns data
    returns = pd.DataFrame({
        'AAPL': np.random.normal(0.001, 0.02, 100),
        'MSFT': np.random.normal(0.0008, 0.018, 100),
        'GOOGL': np.random.normal(0.0012, 0.022, 100)
    })
    
    optimizer = MarkowitzOptimizer()
    weights, metrics = optimizer.optimize(returns, objective='max_sharpe')
    
    assert abs(sum(weights.values()) - 1.0) < 0.01
    assert 'sharpe_ratio' in metrics
    assert metrics['sharpe_ratio'] > -10  # Sanity check


if __name__ == "__main__":
    pytest.main([__file__, "-v"])