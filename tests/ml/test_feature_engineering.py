"""
Test suite for ML feature engineering components
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class TestTechnicalFeatures:
    """Test technical feature engineering"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data"""
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        
        # Generate realistic price data
        np.random.seed(42)
        close_prices = 100 * (1 + np.random.randn(100).cumsum() * 0.01)
        
        data = pd.DataFrame({
            'open': close_prices * (1 + np.random.uniform(-0.01, 0.01, 100)),
            'high': close_prices * (1 + np.random.uniform(0, 0.02, 100)),
            'low': close_prices * (1 + np.random.uniform(-0.02, 0, 100)),
            'close': close_prices,
            'volume': np.random.uniform(1e6, 1e7, 100)
        }, index=dates)
        
        return data
    
    def test_technical_indicators(self, sample_data):
        """Test technical indicator calculation"""
        from src.bot.ml.features.technical import TechnicalFeatureEngineer
        
        engineer = TechnicalFeatureEngineer()
        features = engineer.generate_features(sample_data)
        
        # Check that features were generated
        assert not features.empty
        assert len(features.columns) > 50  # Should have many features
        
        # Check specific indicators
        expected_indicators = [
            'rsi_14', 'macd', 'macd_signal',
            'bb_upper', 'bb_lower', 'bb_percent',
            'sma_20', 'ema_20', 'atr_14',
            'adx', 'obv', 'volume_ratio'
        ]
        
        for indicator in expected_indicators:
            assert any(indicator in col for col in features.columns), f"Missing {indicator}"
        
        # Check data quality
        assert features.isna().sum().sum() / features.size < 0.3  # Less than 30% NaN
    
    def test_momentum_features(self, sample_data):
        """Test momentum feature calculation"""
        from src.bot.ml.features.technical import TechnicalFeatureEngineer
        
        engineer = TechnicalFeatureEngineer()
        features = engineer.generate_features(sample_data)
        
        # Check momentum features
        momentum_cols = [col for col in features.columns if 'momentum' in col.lower()]
        assert len(momentum_cols) > 0
        
        # Check RSI values are in valid range
        rsi_cols = [col for col in features.columns if 'rsi' in col]
        for col in rsi_cols:
            valid_data = features[col].dropna()
            assert (valid_data >= 0).all() and (valid_data <= 100).all()
    
    def test_volatility_features(self, sample_data):
        """Test volatility feature calculation"""
        from src.bot.ml.features.technical import TechnicalFeatureEngineer
        
        engineer = TechnicalFeatureEngineer()
        features = engineer.generate_features(sample_data)
        
        # Check volatility features
        vol_cols = [col for col in features.columns if 'volatility' in col.lower() or 'atr' in col.lower()]
        assert len(vol_cols) > 0
        
        # Check ATR is positive
        atr_cols = [col for col in features.columns if 'atr' in col]
        for col in atr_cols:
            valid_data = features[col].dropna()
            assert (valid_data >= 0).all()


class TestMarketRegimeFeatures:
    """Test market regime feature engineering"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample market data"""
        dates = pd.date_range(end=datetime.now(), periods=200, freq='D')
        
        np.random.seed(42)
        returns = np.random.randn(200) * 0.02
        close_prices = 100 * (1 + returns).cumprod()
        
        data = pd.DataFrame({
            'open': close_prices * 0.99,
            'high': close_prices * 1.01,
            'low': close_prices * 0.98,
            'close': close_prices,
            'volume': np.random.uniform(1e6, 1e7, 200)
        }, index=dates)
        
        return data
    
    def test_regime_features(self, sample_data):
        """Test regime feature generation"""
        from src.bot.ml.features.market_regime import MarketRegimeFeatures
        
        engineer = MarketRegimeFeatures()
        features = engineer.generate_features(sample_data)
        
        # Check features were generated
        assert not features.empty
        assert len(features.columns) > 30  # Should have many regime features
        
        # Check specific feature categories
        assert any('return' in col for col in features.columns)
        assert any('volatility' in col for col in features.columns)
        assert any('volume' in col for col in features.columns)
    
    def test_statistical_features(self, sample_data):
        """Test statistical feature calculation"""
        from src.bot.ml.features.market_regime import MarketRegimeFeatures
        
        engineer = MarketRegimeFeatures()
        features = engineer.generate_features(sample_data)
        
        # Check for advanced statistical features
        statistical_features = ['skewness', 'kurtosis', 'hurst', 'entropy']
        
        for feature in statistical_features:
            assert any(feature in col.lower() for col in features.columns), f"Missing {feature}"
    
    def test_microstructure_features(self, sample_data):
        """Test microstructure feature calculation"""
        from src.bot.ml.features.market_regime import MarketRegimeFeatures
        
        engineer = MarketRegimeFeatures()
        features = engineer.generate_features(sample_data)
        
        # Check microstructure features
        micro_features = ['spread', 'high_low_ratio', 'close_to_high']
        
        for feature in micro_features:
            matching_cols = [col for col in features.columns if feature in col.lower()]
            assert len(matching_cols) > 0, f"Missing {feature}"


class TestFeatureEngineeringPipeline:
    """Test complete feature engineering pipeline"""
    
    @pytest.fixture
    def sample_data(self):
        """Create comprehensive sample data"""
        dates = pd.date_range(end=datetime.now(), periods=250, freq='D')
        
        np.random.seed(42)
        trend = np.linspace(100, 120, 250)
        noise = np.random.randn(250) * 2
        close_prices = trend + noise
        
        data = pd.DataFrame({
            'open': close_prices * (1 + np.random.uniform(-0.01, 0.01, 250)),
            'high': close_prices * (1 + np.random.uniform(0, 0.02, 250)),
            'low': close_prices * (1 + np.random.uniform(-0.02, 0, 250)),
            'close': close_prices,
            'volume': np.random.uniform(5e6, 1.5e7, 250)
        }, index=dates)
        
        return data
    
    def test_pipeline_integration(self, sample_data):
        """Test complete pipeline integration"""
        from src.bot.ml.features.engineering import FeatureEngineeringPipeline
        
        pipeline = FeatureEngineeringPipeline()
        features = pipeline.generate_features(sample_data, store_features=False)
        
        # Check comprehensive feature set
        assert not features.empty
        assert len(features.columns) > 100  # Should have 200+ features
        
        # Check data quality
        nan_ratio = features.isna().sum().sum() / features.size
        assert nan_ratio < 0.4  # Less than 40% NaN (expected due to indicators)
        
        # Check feature diversity
        feature_categories = {
            'technical': ['rsi', 'macd', 'bb', 'sma', 'ema'],
            'regime': ['return', 'volatility', 'volume'],
            'statistical': ['skew', 'kurt', 'hurst']
        }
        
        for category, keywords in feature_categories.items():
            category_found = False
            for keyword in keywords:
                if any(keyword in col.lower() for col in features.columns):
                    category_found = True
                    break
            assert category_found, f"Missing {category} features"
    
    def test_feature_selection(self, sample_data):
        """Test feature selection in pipeline"""
        from src.bot.ml.features.engineering import FeatureEngineeringPipeline
        
        pipeline = FeatureEngineeringPipeline()
        
        # Test with feature selection
        features_all = pipeline.generate_features(sample_data, store_features=False)
        
        # Should have cleaned features
        assert not features_all.empty
        
        # Check that highly correlated features might be reduced
        correlation_matrix = features_all.corr()
        high_corr_pairs = 0
        
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                if abs(correlation_matrix.iloc[i, j]) > 0.95:
                    high_corr_pairs += 1
        
        # Should have some correlation but not too many perfect correlations
        assert high_corr_pairs < len(features_all.columns) * 0.1  # Less than 10% highly correlated
    
    def test_feature_scaling(self, sample_data):
        """Test feature scaling and normalization"""
        from src.bot.ml.features.engineering import FeatureEngineeringPipeline
        
        pipeline = FeatureEngineeringPipeline()
        features = pipeline.generate_features(sample_data, store_features=False)
        
        # Check that features are reasonably scaled
        # Most features should be within reasonable range
        numeric_features = features.select_dtypes(include=[np.number])
        
        for col in numeric_features.columns:
            valid_data = features[col].dropna()
            if len(valid_data) > 0:
                # Check for extreme outliers
                q1 = valid_data.quantile(0.01)
                q99 = valid_data.quantile(0.99)
                range_size = q99 - q1
                
                # Most financial features should not have extreme ranges
                if 'volume' not in col.lower() and 'price' not in col.lower():
                    assert range_size < 1000, f"Feature {col} has extreme range: {range_size}"


def test_feature_persistence():
    """Test feature persistence and loading"""
    # This would test database storage if implemented
    pass


def test_feature_quality_metrics():
    """Test feature quality assessment"""
    from src.bot.ml.features.engineering import FeatureEngineeringPipeline
    
    # Create sample data
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    data = pd.DataFrame({
        'open': np.random.uniform(99, 101, 100),
        'high': np.random.uniform(100, 102, 100),
        'low': np.random.uniform(98, 100, 100),
        'close': np.random.uniform(99, 101, 100),
        'volume': np.random.uniform(1e6, 1e7, 100)
    }, index=dates)
    
    pipeline = FeatureEngineeringPipeline()
    features = pipeline.generate_features(data, store_features=False)
    
    # Calculate quality metrics
    quality_metrics = {
        'completeness': 1 - (features.isna().sum().sum() / features.size),
        'n_features': len(features.columns),
        'n_samples': len(features),
        'feature_variance': features.var().mean()
    }
    
    # Assert quality thresholds
    assert quality_metrics['completeness'] > 0.5  # At least 50% complete
    assert quality_metrics['n_features'] > 50  # Substantial feature set
    assert quality_metrics['feature_variance'] > 0  # Features have variance


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])