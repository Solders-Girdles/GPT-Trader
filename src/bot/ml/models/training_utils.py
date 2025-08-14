"""
Training utilities for ML models
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
from pathlib import Path
import json

from .regime_detector import MarketRegimeDetector
from ..features.engineering import FeatureEngineeringPipeline
from ...core.base import ComponentConfig


class ModelTrainer:
    """Utility class for training ML models"""
    
    def __init__(self, db_manager=None):
        """Initialize model trainer
        
        Args:
            db_manager: Database manager for persistence
        """
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
        
    def train_regime_detector(self, 
                            data: pd.DataFrame,
                            symbol: str = None,
                            n_regimes: int = 5,
                            save_model: bool = True) -> Tuple[MarketRegimeDetector, Dict[str, float]]:
        """Train a market regime detector
        
        Args:
            data: OHLCV data for training
            symbol: Optional symbol identifier
            n_regimes: Number of regimes to detect
            save_model: Whether to save the trained model
            
        Returns:
            Tuple of (trained detector, metrics)
        """
        self.logger.info(f"Training regime detector with {len(data)} samples")
        
        # Generate features
        feature_pipeline = FeatureEngineeringPipeline(db_manager=self.db_manager)
        features = feature_pipeline.generate_features(data, symbol=symbol)
        
        # Initialize regime detector
        detector = MarketRegimeDetector(
            db_manager=self.db_manager,
            n_components=n_regimes
        )
        
        # Train model
        metrics = detector.train(features)
        
        # Save model if requested
        if save_model:
            model_id = detector.save_model()
            self.logger.info(f"Saved regime detector model: {model_id}")
            
            # Register as active model
            if self.db_manager:
                self._set_active_model('MarketRegimeDetector', model_id)
        
        return detector, metrics
    
    def prepare_training_data(self,
                            symbol: str,
                            start_date: str,
                            end_date: str,
                            source: str = 'yfinance') -> pd.DataFrame:
        """Prepare training data from various sources
        
        Args:
            symbol: Symbol to fetch
            start_date: Start date for data
            end_date: End date for data
            source: Data source ('yfinance', 'database', etc.)
            
        Returns:
            DataFrame with OHLCV data
        """
        if source == 'database' and self.db_manager:
            # Load from database
            query = """
                SELECT timestamp, open, high, low, close, volume
                FROM bars
                WHERE symbol = ?
                AND timestamp >= ?
                AND timestamp <= ?
                ORDER BY timestamp
            """
            
            results = self.db_manager.fetch_all(query, (symbol, start_date, end_date))
            
            if results:
                df = pd.DataFrame(results)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                return df
        
        elif source == 'yfinance':
            # Use yfinance to fetch data
            try:
                import yfinance as yf
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date)
                
                # Rename columns to lowercase
                df.columns = df.columns.str.lower()
                
                return df
            except Exception as e:
                self.logger.error(f"Error fetching data from yfinance: {e}")
                raise
        
        else:
            raise ValueError(f"Unsupported data source: {source}")
    
    def cross_validate_regime_detector(self,
                                      data: pd.DataFrame,
                                      n_splits: int = 5,
                                      n_regimes_range: List[int] = [3, 4, 5, 6]) -> Dict[int, Dict]:
        """Cross-validate regime detector with different parameters
        
        Args:
            data: OHLCV data
            n_splits: Number of cross-validation splits
            n_regimes_range: Range of regime counts to test
            
        Returns:
            Dictionary of results for each regime count
        """
        results = {}
        
        # Generate features once
        feature_pipeline = FeatureEngineeringPipeline(db_manager=self.db_manager)
        features = feature_pipeline.generate_features(data)
        
        # Test different numbers of regimes
        for n_regimes in n_regimes_range:
            self.logger.info(f"Testing {n_regimes} regimes")
            
            scores = []
            split_size = len(features) // n_splits
            
            for i in range(n_splits):
                # Create train/test split
                test_start = i * split_size
                test_end = (i + 1) * split_size
                
                train_data = pd.concat([
                    features.iloc[:test_start],
                    features.iloc[test_end:]
                ])
                test_data = features.iloc[test_start:test_end]
                
                # Train model
                detector = MarketRegimeDetector(n_components=n_regimes)
                detector.train(train_data)
                
                # Evaluate on test set
                test_score = detector.model.score(detector._prepare_features(test_data))
                scores.append(test_score)
            
            results[n_regimes] = {
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'scores': scores
            }
        
        # Find optimal number of regimes
        best_n = max(results.keys(), key=lambda k: results[k]['mean_score'])
        self.logger.info(f"Optimal number of regimes: {best_n}")
        
        return results
    
    def _set_active_model(self, model_type: str, model_id: str):
        """Set a model as active in the database
        
        Args:
            model_type: Type of model
            model_id: ID of model to activate
        """
        if not self.db_manager:
            return
        
        try:
            # Deactivate other models of same type
            self.db_manager.execute(
                "UPDATE ml_models SET is_active = 0 WHERE model_type = ?",
                (model_type,)
            )
            
            # Activate this model
            self.db_manager.execute(
                "UPDATE ml_models SET is_active = 1 WHERE model_id = ?",
                (model_id,)
            )
            
            self.logger.info(f"Set {model_id} as active {model_type}")
            
        except Exception as e:
            self.logger.error(f"Error setting active model: {e}")


class RegimeAnalyzer:
    """Analyze market regimes and their characteristics"""
    
    def __init__(self, detector: MarketRegimeDetector):
        """Initialize regime analyzer
        
        Args:
            detector: Trained regime detector
        """
        self.detector = detector
        self.logger = logging.getLogger(__name__)
    
    def analyze_regime_transitions(self, 
                                  data: pd.DataFrame,
                                  lookback_days: int = 250) -> Dict[str, Any]:
        """Analyze regime transitions over historical data
        
        Args:
            data: OHLCV data
            lookback_days: Days to analyze
            
        Returns:
            Dictionary with transition analysis
        """
        # Generate features
        feature_pipeline = FeatureEngineeringPipeline()
        features = feature_pipeline.generate_features(data)
        
        # Get regime predictions
        regimes, confidence = self.detector.get_regime_confidence(features)
        
        # Analyze transitions
        transitions = []
        for i in range(1, len(regimes)):
            if regimes[i] != regimes[i-1]:
                transitions.append({
                    'date': data.index[i],
                    'from_regime': regimes[i-1],
                    'to_regime': regimes[i],
                    'confidence': float(confidence[i])
                })
        
        # Calculate transition frequencies
        transition_counts = {}
        for trans in transitions:
            key = f"{trans['from_regime']}_to_{trans['to_regime']}"
            transition_counts[key] = transition_counts.get(key, 0) + 1
        
        # Get regime durations
        regime_durations = self._calculate_regime_durations(regimes)
        
        # Calculate regime returns
        regime_returns = self._calculate_regime_returns(data, regimes)
        
        analysis = {
            'total_transitions': len(transitions),
            'transitions_per_day': len(transitions) / lookback_days,
            'transition_counts': transition_counts,
            'recent_transitions': transitions[-10:] if transitions else [],
            'regime_durations': regime_durations,
            'regime_returns': regime_returns,
            'current_regime': regimes[-1],
            'current_confidence': float(confidence[-1])
        }
        
        return analysis
    
    def _calculate_regime_durations(self, regimes: np.ndarray) -> Dict[str, Dict]:
        """Calculate duration statistics for each regime
        
        Args:
            regimes: Array of regime predictions
            
        Returns:
            Dictionary with duration statistics
        """
        durations = {}
        
        current_regime = regimes[0]
        current_start = 0
        
        for i in range(1, len(regimes)):
            if regimes[i] != current_regime:
                # Regime ended
                regime_name = current_regime
                duration = i - current_start
                
                if regime_name not in durations:
                    durations[regime_name] = []
                durations[regime_name].append(duration)
                
                # Start new regime
                current_regime = regimes[i]
                current_start = i
        
        # Add last regime
        if current_regime not in durations:
            durations[current_regime] = []
        durations[current_regime].append(len(regimes) - current_start)
        
        # Calculate statistics
        duration_stats = {}
        for regime, dur_list in durations.items():
            duration_stats[regime] = {
                'mean_duration': float(np.mean(dur_list)),
                'std_duration': float(np.std(dur_list)),
                'min_duration': int(np.min(dur_list)),
                'max_duration': int(np.max(dur_list)),
                'occurrences': len(dur_list)
            }
        
        return duration_stats
    
    def _calculate_regime_returns(self, 
                                 data: pd.DataFrame,
                                 regimes: np.ndarray) -> Dict[str, Dict]:
        """Calculate return statistics for each regime
        
        Args:
            data: OHLCV data
            regimes: Array of regime predictions
            
        Returns:
            Dictionary with return statistics
        """
        returns = data['close'].pct_change().values
        
        regime_returns = {}
        
        for regime in np.unique(regimes):
            mask = regimes == regime
            regime_rets = returns[mask]
            
            # Remove NaN values
            regime_rets = regime_rets[~np.isnan(regime_rets)]
            
            if len(regime_rets) > 0:
                regime_returns[regime] = {
                    'mean_return': float(np.mean(regime_rets) * 252),  # Annualized
                    'std_return': float(np.std(regime_rets) * np.sqrt(252)),  # Annualized
                    'sharpe_ratio': float(np.mean(regime_rets) / (np.std(regime_rets) + 1e-10) * np.sqrt(252)),
                    'min_return': float(np.min(regime_rets)),
                    'max_return': float(np.max(regime_rets)),
                    'positive_days': float(np.sum(regime_rets > 0) / len(regime_rets))
                }
            else:
                regime_returns[regime] = {
                    'mean_return': 0.0,
                    'std_return': 0.0,
                    'sharpe_ratio': 0.0,
                    'min_return': 0.0,
                    'max_return': 0.0,
                    'positive_days': 0.0
                }
        
        return regime_returns
    
    def get_regime_trading_signals(self,
                                  current_regime: str,
                                  confidence: float) -> Dict[str, Any]:
        """Generate trading signals based on regime
        
        Args:
            current_regime: Current market regime
            confidence: Confidence in regime prediction
            
        Returns:
            Dictionary with trading recommendations
        """
        # Define regime-based strategies
        regime_strategies = {
            'bull_quiet': {
                'position': 'long',
                'size': 1.0,
                'stop_loss': 0.03,
                'take_profit': 0.08,
                'strategy': 'trend_following'
            },
            'bull_volatile': {
                'position': 'long',
                'size': 0.7,
                'stop_loss': 0.05,
                'take_profit': 0.12,
                'strategy': 'momentum'
            },
            'bear_quiet': {
                'position': 'short',
                'size': 0.5,
                'stop_loss': 0.03,
                'take_profit': 0.06,
                'strategy': 'mean_reversion'
            },
            'bear_volatile': {
                'position': 'cash',
                'size': 0.2,
                'stop_loss': 0.02,
                'take_profit': 0.04,
                'strategy': 'defensive'
            },
            'sideways': {
                'position': 'neutral',
                'size': 0.5,
                'stop_loss': 0.02,
                'take_profit': 0.04,
                'strategy': 'range_trading'
            }
        }
        
        # Get strategy for current regime
        strategy = regime_strategies.get(current_regime, regime_strategies['sideways'])
        
        # Adjust for confidence
        if confidence < 0.6:
            strategy['size'] *= 0.5  # Reduce position size for low confidence
        
        signals = {
            'regime': current_regime,
            'confidence': confidence,
            'position': strategy['position'],
            'position_size': strategy['size'],
            'stop_loss': strategy['stop_loss'],
            'take_profit': strategy['take_profit'],
            'recommended_strategy': strategy['strategy'],
            'timestamp': datetime.now().isoformat()
        }
        
        return signals