"""
Hidden Markov Model for market regime detection
"""

import numpy as np
import pandas as pd
from hmmlearn import hmm
from typing import Dict, List, Tuple, Optional, Any
import json
import logging
from datetime import datetime
from sklearn.preprocessing import StandardScaler

from ..base import MLModel
from ...core.base import ComponentConfig


class MarketRegimeDetector(MLModel):
    """Hidden Markov Model for detecting market regimes"""
    
    # Define market regime types
    REGIMES = {
        0: 'bull_quiet',      # Low volatility uptrend
        1: 'bull_volatile',   # High volatility uptrend
        2: 'bear_quiet',      # Low volatility downtrend
        3: 'bear_volatile',   # High volatility downtrend
        4: 'sideways'         # Range-bound market
    }
    
    REGIME_COLORS = {
        'bull_quiet': 'green',
        'bull_volatile': 'lightgreen',
        'bear_quiet': 'red',
        'bear_volatile': 'darkred',
        'sideways': 'gray'
    }
    
    def __init__(self, 
                 config: Optional[ComponentConfig] = None,
                 db_manager=None,
                 n_components: int = 5,
                 covariance_type: str = 'full'):
        """Initialize market regime detector
        
        Args:
            config: Component configuration
            db_manager: Database manager
            n_components: Number of hidden states (regimes)
            covariance_type: Type of covariance matrix ('full', 'diag', 'tied', 'spherical')
        """
        if config is None:
            config = ComponentConfig(
                component_id='regime_detector',
                component_type='ml_model'
            )
        super().__init__(config, db_manager)
        
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.logger = logging.getLogger(__name__)
        
        # Initialize HMM model
        self.model = hmm.GaussianHMM(
            n_components=n_components,
            covariance_type=covariance_type,
            n_iter=100,
            tol=0.01,
            random_state=42,
            init_params='stmc',
            params='stmc'
        )
        
        # Feature configuration
        self.feature_columns = [
            'returns_1d',
            'returns_5d', 
            'volatility_20d',
            'volume_ratio',
            'trend_strength_20d'
        ]
        
        # Scaler for feature normalization
        self.scaler = StandardScaler()
        
        # Regime statistics
        self.regime_stats = {}
        self.transition_matrix = None
        
    def train(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Dict[str, float]:
        """Train HMM on market features
        
        Args:
            X: DataFrame with market features
            y: Optional labels (not used for unsupervised HMM)
            
        Returns:
            Dictionary of training metrics
        """
        self.logger.info(f"Training HMM with {self.n_components} regimes")
        
        # Select and prepare features
        X_train = self._prepare_features(X)
        
        if len(X_train) < 100:
            raise ValueError(f"Insufficient data for training: {len(X_train)} samples")
        
        # Fit the model
        try:
            self.model.fit(X_train)
            self.is_trained = True
            
            # Calculate training metrics
            log_likelihood = self.model.score(X_train)
            aic = self.model.aic(X_train)
            bic = self.model.bic(X_train)
            
            # Store transition matrix
            self.transition_matrix = self.model.transmat_
            
            # Calculate regime statistics
            states = self.model.predict(X_train)
            self._calculate_regime_statistics(X, states)
            
            self.metrics = {
                'log_likelihood': float(log_likelihood),
                'aic': float(aic),
                'bic': float(bic),
                'n_samples': len(X_train),
                'n_regimes': self.n_components,
                'convergence': self.model.monitor_.converged
            }
            
            self.logger.info(f"HMM training complete. Log-likelihood: {log_likelihood:.2f}")
            
        except Exception as e:
            self.logger.error(f"HMM training failed: {e}")
            raise
        
        return self.metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict market regimes
        
        Args:
            X: DataFrame with market features
            
        Returns:
            Array of regime predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Prepare features
        X_pred = self._prepare_features(X)
        
        # Predict hidden states
        states = self.model.predict(X_pred)
        
        # Get regime names
        regimes = np.array([self.REGIMES[state] for state in states])
        
        # Store predictions if database available
        if self.db_manager:
            self._store_predictions(X.index, regimes, states)
        
        return regimes
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get regime probabilities
        
        Args:
            X: DataFrame with market features
            
        Returns:
            Array of regime probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Prepare features
        X_pred = self._prepare_features(X)
        
        # Get posterior probabilities
        _, posteriors = self.model.score_samples(X_pred)
        
        return posteriors
    
    def get_regime_confidence(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Get regime predictions with confidence scores
        
        Args:
            X: DataFrame with market features
            
        Returns:
            Tuple of (regime names, confidence scores)
        """
        # Get probabilities
        probs = self.predict_proba(X)
        
        # Get most likely regime and confidence
        states = np.argmax(probs, axis=1)
        confidence = np.max(probs, axis=1)
        
        # Convert to regime names
        regimes = np.array([self.REGIMES[state] for state in states])
        
        return regimes, confidence
    
    def get_transition_probabilities(self) -> pd.DataFrame:
        """Get regime transition probability matrix
        
        Returns:
            DataFrame with transition probabilities
        """
        if self.transition_matrix is None:
            raise ValueError("Model must be trained first")
        
        # Create DataFrame with regime names
        regime_names = [self.REGIMES[i] for i in range(self.n_components)]
        
        trans_df = pd.DataFrame(
            self.transition_matrix,
            index=regime_names,
            columns=regime_names
        )
        
        return trans_df
    
    def get_regime_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistical properties of each regime
        
        Returns:
            Dictionary of regime statistics
        """
        return self.regime_stats
    
    def _prepare_features(self, X: pd.DataFrame) -> np.ndarray:
        """Prepare features for HMM
        
        Args:
            X: Raw feature DataFrame
            
        Returns:
            Normalized feature array
        """
        # Select required features
        available_features = [col for col in self.feature_columns if col in X.columns]
        
        if not available_features:
            raise ValueError(f"None of required features found: {self.feature_columns}")
        
        # Use available features
        X_selected = X[available_features].copy()
        
        # Fill missing values
        X_selected = X_selected.ffill().fillna(0)
        
        # Remove infinite values
        X_selected = X_selected.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Normalize features
        if self.is_trained:
            X_normalized = self.scaler.transform(X_selected)
        else:
            X_normalized = self.scaler.fit_transform(X_selected)
        
        return X_normalized
    
    def _calculate_regime_statistics(self, X: pd.DataFrame, states: np.ndarray):
        """Calculate statistics for each regime
        
        Args:
            X: Original feature DataFrame
            states: Predicted regime states
        """
        self.regime_stats = {}
        
        # Calculate returns if available
        if 'returns_1d' in X.columns:
            returns = X['returns_1d'].values
        elif 'close' in X.columns:
            returns = X['close'].pct_change().values
        else:
            returns = np.zeros(len(X))
        
        # Calculate volatility if available
        if 'volatility_20d' in X.columns:
            volatility = X['volatility_20d'].values
        else:
            volatility = pd.Series(returns).rolling(20).std().values
        
        # Calculate statistics for each regime
        for state in range(self.n_components):
            regime_name = self.REGIMES[state]
            mask = states == state
            
            if np.sum(mask) > 0:
                regime_returns = returns[mask]
                regime_volatility = volatility[mask]
                
                self.regime_stats[regime_name] = {
                    'mean_return': float(np.nanmean(regime_returns)),
                    'std_return': float(np.nanstd(regime_returns)),
                    'mean_volatility': float(np.nanmean(regime_volatility)),
                    'frequency': float(np.sum(mask) / len(states)),
                    'avg_duration': self._calculate_avg_duration(states, state),
                    'sharpe_ratio': float(np.nanmean(regime_returns) / (np.nanstd(regime_returns) + 1e-10))
                }
            else:
                self.regime_stats[regime_name] = {
                    'mean_return': 0.0,
                    'std_return': 0.0,
                    'mean_volatility': 0.0,
                    'frequency': 0.0,
                    'avg_duration': 0.0,
                    'sharpe_ratio': 0.0
                }
    
    def _calculate_avg_duration(self, states: np.ndarray, state: int) -> float:
        """Calculate average duration of a regime
        
        Args:
            states: Array of all states
            state: Specific state to analyze
            
        Returns:
            Average duration in periods
        """
        durations = []
        current_duration = 0
        
        for s in states:
            if s == state:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                    current_duration = 0
        
        # Add last duration if still in state
        if current_duration > 0:
            durations.append(current_duration)
        
        return float(np.mean(durations)) if durations else 0.0
    
    def _store_predictions(self, timestamps, regimes: np.ndarray, states: np.ndarray):
        """Store regime predictions in database
        
        Args:
            timestamps: Index with timestamps
            regimes: Regime names
            states: Numeric state values
        """
        if not self.db_manager:
            return
        
        for ts, regime, state in zip(timestamps, regimes, states):
            try:
                # Get confidence if available
                confidence = 0.0  # Could be enhanced with probability
                
                self.db_manager.execute(
                    """INSERT OR REPLACE INTO market_regimes 
                       (date, regime_id, regime_name, confidence, model_version)
                       VALUES (?, ?, ?, ?, ?)""",
                    (ts, int(state), regime, confidence, self.model_id)
                )
            except Exception as e:
                self.logger.error(f"Error storing regime prediction: {e}")
    
    def plot_regimes(self, X: pd.DataFrame, prices: pd.Series = None) -> Dict[str, Any]:
        """Generate regime visualization data
        
        Args:
            X: Feature DataFrame
            prices: Optional price series for overlay
            
        Returns:
            Dictionary with plotting data
        """
        # Get regime predictions
        regimes, confidence = self.get_regime_confidence(X)
        
        # Prepare plotting data
        plot_data = {
            'timestamps': X.index.tolist(),
            'regimes': regimes.tolist(),
            'confidence': confidence.tolist(),
            'colors': [self.REGIME_COLORS[r] for r in regimes],
            'regime_stats': self.regime_stats,
            'transition_matrix': self.get_transition_probabilities().to_dict()
        }
        
        # Add price data if provided
        if prices is not None:
            plot_data['prices'] = prices.tolist()
        
        return plot_data
    
    def analyze_current_regime(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Analyze the current market regime
        
        Args:
            X: Recent feature data
            
        Returns:
            Dictionary with regime analysis
        """
        # Get current regime
        regimes, confidence = self.get_regime_confidence(X)
        current_regime = regimes[-1]
        current_confidence = confidence[-1]
        
        # Get transition probabilities from current regime
        trans_probs = self.get_transition_probabilities()
        next_regime_probs = trans_probs.loc[current_regime].to_dict()
        
        # Find most likely next regime
        most_likely_next = max(next_regime_probs, key=next_regime_probs.get)
        
        # Get regime statistics
        current_stats = self.regime_stats.get(current_regime, {})
        
        analysis = {
            'current_regime': current_regime,
            'confidence': float(current_confidence),
            'regime_statistics': current_stats,
            'next_regime_probabilities': next_regime_probs,
            'most_likely_next_regime': most_likely_next,
            'regime_color': self.REGIME_COLORS[current_regime],
            'timestamp': datetime.now().isoformat()
        }
        
        return analysis