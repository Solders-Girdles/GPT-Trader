"""
ML Pipeline Integration Module for Trading Orchestrator

This module provides enhanced ML integration with caching, confidence filtering,
and regime-aware position sizing.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json

logger = logging.getLogger(__name__)


@dataclass
class MLDecision:
    """Represents a complete ML-driven trading decision"""
    symbol: str
    strategy: str
    confidence: float
    expected_return: float
    regime: str
    regime_confidence: float
    position_size: float
    risk_adjusted_size: float
    decision: str  # 'buy', 'sell', 'hold'
    reasoning: List[str]
    timestamp: datetime


class MLPipelineIntegrator:
    """
    Integrates ML components (strategy selection, regime detection, position sizing)
    into a cohesive trading decision pipeline with caching and validation.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger("ml_integrator")
        
        # Configuration
        self.min_confidence = self.config.get('min_confidence', 0.6)
        self.max_position_size = self.config.get('max_position_size', 0.2)
        self.enable_caching = self.config.get('enable_caching', True)
        self.cache_ttl = self.config.get('cache_ttl_minutes', 5)
        
        # Cache for ML predictions
        self._prediction_cache = {}
        self._cache_timestamps = {}
        
        # Load ML components
        self._load_ml_components()
        
    def _load_ml_components(self):
        """Load ML feature slices with error handling"""
        try:
            from ..features.ml_strategy import ml_strategy
            self.ml_strategy = ml_strategy
            self.logger.info("ML strategy component loaded")
        except ImportError as e:
            self.logger.warning(f"ML strategy not available: {e}")
            self.ml_strategy = None
            
        try:
            from ..features.market_regime import market_regime
            self.market_regime = market_regime
            self.logger.info("Market regime component loaded")
        except ImportError as e:
            self.logger.warning(f"Market regime not available: {e}")
            self.market_regime = None
            
        try:
            from ..features.position_sizing import position_sizing
            self.position_sizing = position_sizing
            self.logger.info("Position sizing component loaded")
        except ImportError as e:
            self.logger.warning(f"Position sizing not available: {e}")
            self.position_sizing = None
    
    def make_trading_decision(
        self, 
        symbol: str, 
        portfolio_value: float,
        current_positions: Dict[str, float] = None
    ) -> MLDecision:
        """
        Make a complete ML-driven trading decision.
        
        Args:
            symbol: Stock symbol to analyze
            portfolio_value: Total portfolio value
            current_positions: Current position sizes by symbol
            
        Returns:
            MLDecision with strategy, confidence, position size, and action
        """
        self.logger.info(f"Making ML trading decision for {symbol}")
        
        reasoning = []
        current_positions = current_positions or {}
        
        # Step 1: Detect market regime
        regime, regime_confidence = self._get_market_regime(symbol)
        reasoning.append(f"Market regime: {regime} (confidence: {regime_confidence:.1%})")
        
        # Step 2: Select best strategy with caching
        strategy, confidence, expected_return = self._get_strategy_prediction(
            symbol, regime
        )
        reasoning.append(f"Selected strategy: {strategy} (confidence: {confidence:.1%})")
        
        # Step 3: Apply confidence filtering
        if confidence < self.min_confidence:
            reasoning.append(f"Confidence {confidence:.1%} below threshold {self.min_confidence:.1%}")
            return MLDecision(
                symbol=symbol,
                strategy=strategy,
                confidence=confidence,
                expected_return=expected_return,
                regime=regime,
                regime_confidence=regime_confidence,
                position_size=0.0,
                risk_adjusted_size=0.0,
                decision='hold',
                reasoning=reasoning,
                timestamp=datetime.now()
            )
        
        # Step 4: Calculate position size with ML adjustments
        base_position_size = self._calculate_ml_position_size(
            confidence=confidence,
            expected_return=expected_return,
            regime=regime,
            portfolio_value=portfolio_value
        )
        
        # Step 5: Apply risk adjustments
        risk_adjusted_size = self._apply_risk_adjustments(
            base_size=base_position_size,
            symbol=symbol,
            regime=regime,
            current_positions=current_positions
        )
        
        reasoning.append(f"Position size: {risk_adjusted_size:.1%} of portfolio")
        
        # Step 6: Determine trading action
        current_position = current_positions.get(symbol, 0.0)
        decision = self._determine_action(
            target_size=risk_adjusted_size,
            current_size=current_position,
            confidence=confidence
        )
        
        reasoning.append(f"Trading decision: {decision}")
        
        return MLDecision(
            symbol=symbol,
            strategy=strategy,
            confidence=confidence,
            expected_return=expected_return,
            regime=regime,
            regime_confidence=regime_confidence,
            position_size=base_position_size,
            risk_adjusted_size=risk_adjusted_size,
            decision=decision,
            reasoning=reasoning,
            timestamp=datetime.now()
        )
    
    def _get_market_regime(self, symbol: str) -> Tuple[str, float]:
        """Get market regime with fallback"""
        if not self.market_regime:
            return "unknown", 0.5
            
        try:
            # Check cache first
            cache_key = f"regime_{symbol}"
            if self._is_cache_valid(cache_key):
                return self._prediction_cache[cache_key]
            
            # Get fresh prediction
            regime_analysis = self.market_regime.detect_regime(
                symbol, lookback_days=60
            )
            
            regime = regime_analysis.current_regime.value
            confidence = regime_analysis.confidence
            
            # Cache result
            if self.enable_caching:
                self._cache_prediction(cache_key, (regime, confidence))
            
            return regime, confidence
            
        except Exception as e:
            self.logger.warning(f"Regime detection failed: {e}")
            return "unknown", 0.5
    
    def _get_strategy_prediction(
        self, 
        symbol: str, 
        regime: str
    ) -> Tuple[str, float, float]:
        """Get ML strategy prediction with caching"""
        if not self.ml_strategy:
            return "momentum", 0.5, 0.05
        
        try:
            # Create cache key including regime
            cache_key = f"strategy_{symbol}_{regime}"
            if self._is_cache_valid(cache_key):
                return self._prediction_cache[cache_key]
            
            # Get fresh prediction
            predictions = self.ml_strategy.predict_best_strategy(
                symbol, lookback_days=60, top_n=1
            )
            
            if predictions and len(predictions) > 0:
                best = predictions[0]
                strategy = best.strategy.value
                confidence = best.confidence
                expected_return = best.expected_return / 100  # Convert to decimal
                
                # Cache result
                if self.enable_caching:
                    self._cache_prediction(cache_key, (strategy, confidence, expected_return))
                
                return strategy, confidence, expected_return
            else:
                return "momentum", 0.5, 0.05
                
        except Exception as e:
            self.logger.warning(f"Strategy prediction failed: {e}")
            return "momentum", 0.5, 0.05
    
    def _calculate_ml_position_size(
        self,
        confidence: float,
        expected_return: float,
        regime: str,
        portfolio_value: float
    ) -> float:
        """Calculate position size using ML signals"""
        
        # Base size from confidence (Kelly-like)
        base_size = min(confidence * expected_return * 2, self.max_position_size)
        
        # Regime adjustments
        regime_multipliers = {
            'BULL_QUIET': 1.2,
            'BULL_VOLATILE': 0.9,
            'SIDEWAYS_QUIET': 0.8,
            'SIDEWAYS_VOLATILE': 0.6,
            'BEAR_QUIET': 0.5,
            'BEAR_VOLATILE': 0.3,
            'CRISIS': 0.1
        }
        
        regime_mult = regime_multipliers.get(regime, 1.0)
        adjusted_size = base_size * regime_mult
        
        # Ensure within bounds
        return max(0.01, min(adjusted_size, self.max_position_size))
    
    def _apply_risk_adjustments(
        self,
        base_size: float,
        symbol: str,
        regime: str,
        current_positions: Dict[str, float]
    ) -> float:
        """Apply portfolio-level risk adjustments"""
        
        # Concentration risk
        total_exposure = sum(current_positions.values())
        if total_exposure > 0.8:  # 80% invested
            base_size *= 0.5  # Reduce new positions
        
        # Single stock limit
        current_position = current_positions.get(symbol, 0.0)
        max_single_position = 0.15  # 15% max per stock
        
        if current_position + base_size > max_single_position:
            base_size = max(0, max_single_position - current_position)
        
        # Crisis mode
        if regime == 'CRISIS':
            base_size *= 0.3
        
        return base_size
    
    def _determine_action(
        self,
        target_size: float,
        current_size: float,
        confidence: float
    ) -> str:
        """Determine trading action based on target vs current position"""
        
        # Threshold for changes (avoid tiny trades)
        min_trade_size = 0.01  # 1% of portfolio
        
        size_diff = target_size - current_size
        
        if abs(size_diff) < min_trade_size:
            return 'hold'
        elif size_diff > 0:
            # Only buy if confidence is high enough
            if confidence >= self.min_confidence:
                return 'buy'
            else:
                return 'hold'
        else:
            # Sell if overweight
            return 'sell'
    
    def _cache_prediction(self, key: str, value: Any):
        """Cache ML prediction with timestamp"""
        self._prediction_cache[key] = value
        self._cache_timestamps[key] = datetime.now()
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached prediction is still valid"""
        if not self.enable_caching:
            return False
            
        if key not in self._prediction_cache:
            return False
            
        timestamp = self._cache_timestamps.get(key)
        if not timestamp:
            return False
            
        age_minutes = (datetime.now() - timestamp).total_seconds() / 60
        return age_minutes < self.cache_ttl
    
    def clear_cache(self):
        """Clear prediction cache"""
        self._prediction_cache.clear()
        self._cache_timestamps.clear()
        self.logger.info("ML prediction cache cleared")
    
    def get_portfolio_ml_decisions(
        self,
        symbols: List[str],
        portfolio_value: float,
        current_positions: Dict[str, float] = None
    ) -> List[MLDecision]:
        """
        Get ML decisions for multiple symbols.
        
        Args:
            symbols: List of symbols to analyze
            portfolio_value: Total portfolio value
            current_positions: Current positions
            
        Returns:
            List of ML decisions sorted by confidence
        """
        decisions = []
        
        for symbol in symbols:
            try:
                decision = self.make_trading_decision(
                    symbol, portfolio_value, current_positions
                )
                decisions.append(decision)
            except Exception as e:
                self.logger.error(f"Failed to get decision for {symbol}: {e}")
        
        # Sort by confidence * expected return
        decisions.sort(
            key=lambda d: d.confidence * d.expected_return,
            reverse=True
        )
        
        return decisions


def create_ml_integrator(config: Optional[Dict] = None) -> MLPipelineIntegrator:
    """Factory function to create ML integrator"""
    return MLPipelineIntegrator(config)