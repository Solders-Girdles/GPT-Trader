"""
ML-enhanced strategy wrapper that combines multiple strategies with ML selection
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime

from .base import Strategy
from ..ml.models import MarketRegimeDetector, StrategyMetaSelector
from ..ml.features import FeatureEngineeringPipeline


class MLEnhancedStrategy(Strategy):
    """Strategy wrapper that uses ML to select and combine strategies"""
    
    def __init__(self,
                 strategies: Dict[str, Strategy],
                 regime_detector: Optional[MarketRegimeDetector] = None,
                 strategy_selector: Optional[StrategyMetaSelector] = None,
                 feature_pipeline: Optional[FeatureEngineeringPipeline] = None,
                 db_manager=None):
        """Initialize ML-enhanced strategy
        
        Args:
            strategies: Dictionary of available strategies
            regime_detector: Trained regime detector model
            strategy_selector: Trained strategy selector model
            feature_pipeline: Feature engineering pipeline
            db_manager: Database manager
        """
        super().__init__()
        
        self.logger = logging.getLogger(__name__)
        self.strategies = strategies
        self.db_manager = db_manager
        
        # ML models
        self.regime_detector = regime_detector
        self.strategy_selector = strategy_selector
        self.feature_pipeline = feature_pipeline or FeatureEngineeringPipeline(db_manager=db_manager)
        
        # Current state
        self.current_strategy = None
        self.current_regime = None
        self.current_confidence = 0.0
        self.ensemble_weights = {}
        
        # Performance tracking
        self.strategy_history = []
        self.regime_history = []
        self.performance_metrics = {}
        
        # Configuration
        self.use_regime_detection = regime_detector is not None
        self.use_strategy_selection = strategy_selector is not None
        self.ensemble_threshold = 0.6  # Confidence threshold for single strategy
        self.min_confidence = 0.3  # Minimum confidence to trade
        
        self.logger.info(f"Initialized ML-enhanced strategy with {len(strategies)} strategies")
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals using ML-selected strategy
        
        Args:
            data: OHLCV market data
            
        Returns:
            Series of trading signals
        """
        # Generate features
        features = self.feature_pipeline.generate_features(data, store_features=False)
        
        # Get current features (last row)
        current_features = features.iloc[[-1]]
        
        # Detect market regime if available
        if self.use_regime_detection:
            regime, confidence = self.regime_detector.get_regime_confidence(current_features)
            self.current_regime = regime[-1]
            regime_confidence = confidence[-1]
            self.logger.info(f"Current regime: {self.current_regime} (confidence: {regime_confidence:.1%})")
        else:
            self.current_regime = 'unknown'
            regime_confidence = 1.0
        
        # Select strategy if available
        if self.use_strategy_selection:
            strategy_name, strategy_confidence, all_probs = (
                self.strategy_selector.select_strategy_with_confidence(current_features)
            )
            
            # Create ensemble if confidence is low
            ensemble_info = self.strategy_selector.create_ensemble_strategy(
                current_features, 
                threshold=self.ensemble_threshold
            )
            
            if ensemble_info['type'] == 'ensemble':
                # Use weighted ensemble of strategies
                signals = self._generate_ensemble_signals(data, ensemble_info['strategies'])
                self.current_strategy = 'ensemble'
                self.ensemble_weights = ensemble_info['strategies']
                self.current_confidence = ensemble_info['confidence']
                
                self.logger.info(f"Using ensemble strategy: {self.ensemble_weights}")
            else:
                # Use single strategy
                self.current_strategy = strategy_name
                self.ensemble_weights = {strategy_name: 1.0}
                self.current_confidence = strategy_confidence
                
                if strategy_name in self.strategies:
                    signals = self.strategies[strategy_name].generate_signals(data)
                    self.logger.info(f"Using strategy: {strategy_name} (confidence: {strategy_confidence:.1%})")
                else:
                    self.logger.warning(f"Strategy {strategy_name} not found, using default")
                    signals = self._generate_default_signals(data)
        else:
            # Fallback to regime-based strategy selection
            signals = self._select_strategy_by_regime(data)
        
        # Adjust signals based on confidence
        signals = self._adjust_signals_by_confidence(signals)
        
        # Record history
        self._record_history(self.current_strategy, self.current_regime, self.current_confidence)
        
        return signals
    
    def _generate_ensemble_signals(self, data: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
        """Generate signals from weighted ensemble of strategies
        
        Args:
            data: OHLCV market data
            weights: Strategy weights
            
        Returns:
            Weighted ensemble signals
        """
        ensemble_signals = pd.Series(0, index=data.index, dtype=float)
        
        for strategy_name, weight in weights.items():
            if strategy_name in self.strategies and weight > 0:
                strategy_signals = self.strategies[strategy_name].generate_signals(data)
                
                # Convert signals to numeric if needed
                if strategy_signals.dtype == bool:
                    strategy_signals = strategy_signals.astype(float)
                
                ensemble_signals += strategy_signals * weight
        
        # Normalize to [-1, 0, 1] range
        ensemble_signals = np.sign(ensemble_signals)
        
        return ensemble_signals
    
    def _select_strategy_by_regime(self, data: pd.DataFrame) -> pd.Series:
        """Select strategy based on market regime
        
        Args:
            data: OHLCV market data
            
        Returns:
            Trading signals
        """
        # Define regime-strategy mapping
        regime_strategy_map = {
            'bull_quiet': 'trend_following',
            'bull_volatile': 'momentum',
            'bear_quiet': 'mean_reversion',
            'bear_volatile': 'defensive',
            'sideways': 'range_trading',
            'unknown': 'balanced'
        }
        
        # Get strategy for current regime
        strategy_name = regime_strategy_map.get(self.current_regime, 'balanced')
        
        # Map to actual strategy if available
        if strategy_name == 'trend_following' and 'trend_breakout' in self.strategies:
            strategy_name = 'trend_breakout'
        elif strategy_name == 'momentum' and 'demo_ma' in self.strategies:
            strategy_name = 'demo_ma'
        elif strategy_name not in self.strategies:
            # Use first available strategy as fallback
            strategy_name = list(self.strategies.keys())[0] if self.strategies else None
        
        if strategy_name and strategy_name in self.strategies:
            self.current_strategy = strategy_name
            self.ensemble_weights = {strategy_name: 1.0}
            return self.strategies[strategy_name].generate_signals(data)
        else:
            return self._generate_default_signals(data)
    
    def _generate_default_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate default signals when no strategy is selected
        
        Args:
            data: OHLCV market data
            
        Returns:
            Default signals (no position)
        """
        return pd.Series(0, index=data.index, dtype=float)
    
    def _adjust_signals_by_confidence(self, signals: pd.Series) -> pd.Series:
        """Adjust signal strength based on confidence
        
        Args:
            signals: Raw signals
            
        Returns:
            Adjusted signals
        """
        if self.current_confidence < self.min_confidence:
            # No trading with very low confidence
            self.logger.warning(f"Confidence too low ({self.current_confidence:.1%}), no position")
            return pd.Series(0, index=signals.index, dtype=float)
        
        # Scale signals by confidence (optional)
        # For now, return signals as-is if confidence is above threshold
        return signals
    
    def _record_history(self, strategy: str, regime: str, confidence: float):
        """Record strategy selection history
        
        Args:
            strategy: Selected strategy name
            regime: Current market regime
            confidence: Selection confidence
        """
        record = {
            'timestamp': datetime.now(),
            'strategy': strategy,
            'regime': regime,
            'confidence': confidence,
            'ensemble_weights': self.ensemble_weights.copy()
        }
        
        self.strategy_history.append(record)
        
        # Keep only recent history (last 1000 records)
        if len(self.strategy_history) > 1000:
            self.strategy_history = self.strategy_history[-1000:]
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current state of ML-enhanced strategy
        
        Returns:
            Dictionary with current state
        """
        return {
            'current_strategy': self.current_strategy,
            'current_regime': self.current_regime,
            'current_confidence': self.current_confidence,
            'ensemble_weights': self.ensemble_weights,
            'use_regime_detection': self.use_regime_detection,
            'use_strategy_selection': self.use_strategy_selection,
            'available_strategies': list(self.strategies.keys()),
            'timestamp': datetime.now().isoformat()
        }
    
    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze performance of ML strategy selection
        
        Returns:
            Dictionary with performance analysis
        """
        if not self.strategy_history:
            return {}
        
        # Count strategy usage
        strategy_counts = {}
        regime_counts = {}
        
        for record in self.strategy_history:
            strategy = record['strategy']
            regime = record['regime']
            
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        # Calculate average confidence
        avg_confidence = np.mean([r['confidence'] for r in self.strategy_history])
        
        # Find most common strategy per regime
        regime_strategy_preference = {}
        for regime in regime_counts.keys():
            regime_records = [r for r in self.strategy_history if r['regime'] == regime]
            if regime_records:
                regime_strategies = {}
                for record in regime_records:
                    s = record['strategy']
                    regime_strategies[s] = regime_strategies.get(s, 0) + 1
                
                preferred = max(regime_strategies, key=regime_strategies.get)
                regime_strategy_preference[regime] = preferred
        
        analysis = {
            'total_decisions': len(self.strategy_history),
            'strategy_usage': strategy_counts,
            'regime_distribution': regime_counts,
            'average_confidence': float(avg_confidence),
            'regime_strategy_preference': regime_strategy_preference,
            'ensemble_usage_rate': sum(1 for r in self.strategy_history if r['strategy'] == 'ensemble') / len(self.strategy_history)
        }
        
        return analysis
    
    def update_models(self, 
                     regime_detector: Optional[MarketRegimeDetector] = None,
                     strategy_selector: Optional[StrategyMetaSelector] = None):
        """Update ML models
        
        Args:
            regime_detector: New regime detector
            strategy_selector: New strategy selector
        """
        if regime_detector:
            self.regime_detector = regime_detector
            self.use_regime_detection = True
            self.logger.info("Updated regime detector")
        
        if strategy_selector:
            self.strategy_selector = strategy_selector
            self.use_strategy_selection = True
            self.logger.info("Updated strategy selector")
    
    def add_strategy(self, name: str, strategy: Strategy):
        """Add a new strategy to the ensemble
        
        Args:
            name: Strategy name
            strategy: Strategy instance
        """
        self.strategies[name] = strategy
        self.logger.info(f"Added strategy: {name}")