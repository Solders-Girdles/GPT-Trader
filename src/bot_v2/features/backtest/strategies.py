"""
Local strategy implementations for backtesting.

This is a self-contained module with all strategies needed for backtesting.
No external dependencies - everything is local to maintain isolation.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

from ...errors import StrategyError, ValidationError
from ...validation import validate_inputs, PositiveNumberValidator, RangeValidator

logger = logging.getLogger(__name__)


class BacktestStrategy(ABC):
    """Base class for backtest strategies."""
    
    def __init__(self, **params):
        """Initialize with parameters."""
        self.params = params
        self._validate_parameters()
        logger.debug(
            f"Initialized {self.__class__.__name__} with parameters: {params}",
            extra={'strategy': self.__class__.__name__, 'params': params}
        )
    
    @abstractmethod
    def run(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals from data."""
        pass
    
    @abstractmethod
    def get_required_periods(self) -> int:
        """Return minimum periods needed."""
        pass
    
    def _validate_parameters(self) -> None:
        """Validate strategy parameters. Override in subclasses."""
        pass
    
    def _safe_rolling_calculation(self, series: pd.Series, window: int, operation: str) -> pd.Series:
        """Safely perform rolling calculations with error handling."""
        try:
            if len(series) < window:
                raise StrategyError(
                    f"Insufficient data for {operation}: need {window} periods, got {len(series)}",
                    strategy_name=self.__class__.__name__,
                    context={'operation': operation, 'window': window, 'available': len(series)}
                )
            
            if operation == 'mean':
                return series.rolling(window=window, min_periods=window).mean()
            elif operation == 'std':
                return series.rolling(window=window, min_periods=window).std()
            elif operation == 'max':
                return series.rolling(window=window, min_periods=window).max()
            elif operation == 'min':
                return series.rolling(window=window, min_periods=window).min()
            else:
                raise StrategyError(
                    f"Unsupported rolling operation: {operation}",
                    strategy_name=self.__class__.__name__,
                    context={'operation': operation}
                )
        except Exception as e:
            if isinstance(e, StrategyError):
                raise
            raise StrategyError(
                f"Failed to calculate {operation} with window {window}",
                strategy_name=self.__class__.__name__,
                context={'operation': operation, 'window': window, 'error': str(e)}
            ) from e


class SimpleMAStrategy(BacktestStrategy):
    """Simple moving average crossover strategy."""
    
    def __init__(self, fast_period: int = 10, slow_period: int = 30, **kwargs):
        # Store parameters before calling super()
        self.fast_period = fast_period
        self.slow_period = slow_period
        super().__init__(fast_period=fast_period, slow_period=slow_period, **kwargs)
    
    def _validate_parameters(self) -> None:
        """Validate MA strategy parameters."""
        if self.fast_period <= 0:
            raise ValidationError(
                "Fast period must be positive",
                field="fast_period",
                value=self.fast_period
            )
        
        if self.slow_period <= 0:
            raise ValidationError(
                "Slow period must be positive",
                field="slow_period",
                value=self.slow_period
            )
        
        if self.fast_period >= self.slow_period:
            raise ValidationError(
                "Fast period must be less than slow period",
                field="period_relationship",
                value=f"fast={self.fast_period}, slow={self.slow_period}"
            )
    
    def run(self, data: pd.DataFrame) -> pd.Series:
        """Generate MA crossover signals."""
        try:
            # Validate input data
            if 'close' not in data.columns:
                raise StrategyError(
                    "Missing 'close' column in data",
                    strategy_name=self.__class__.__name__,
                    context={'available_columns': list(data.columns)}
                )
            
            # Calculate moving averages with error handling
            fast_ma = self._safe_rolling_calculation(data['close'], self.fast_period, 'mean')
            slow_ma = self._safe_rolling_calculation(data['close'], self.slow_period, 'mean')
            
            # Initialize signals
            signals = pd.Series(index=data.index, dtype=int)
            signals[:] = 0
            
            # Check for valid MA values
            if fast_ma.isna().all() or slow_ma.isna().all():
                raise StrategyError(
                    "All moving average values are NaN",
                    strategy_name=self.__class__.__name__,
                    context={
                        'fast_period': self.fast_period,
                        'slow_period': self.slow_period,
                        'data_length': len(data)
                    }
                )
            
            # Generate crossover signals
            fast_above = fast_ma > slow_ma
            fast_above_prev = fast_ma.shift(1) <= slow_ma.shift(1)
            fast_below = fast_ma < slow_ma
            fast_below_prev = fast_ma.shift(1) >= slow_ma.shift(1)
            
            # Buy when fast crosses above slow
            signals[fast_above & fast_above_prev] = 1
            # Sell when fast crosses below slow
            signals[fast_below & fast_below_prev] = -1
            
            return signals.fillna(0)
            
        except Exception as e:
            if isinstance(e, StrategyError):
                raise
            raise StrategyError(
                f"Failed to generate MA crossover signals",
                strategy_name=self.__class__.__name__,
                context={
                    'fast_period': self.fast_period,
                    'slow_period': self.slow_period,
                    'data_shape': data.shape,
                    'error': str(e)
                }
            ) from e
    
    def get_required_periods(self) -> int:
        """Need at least slow_period + 1 for crossover detection."""
        return self.slow_period + 1


class MomentumStrategy(BacktestStrategy):
    """Momentum-based trading strategy."""
    
    def __init__(self, lookback: int = 20, threshold: float = 0.02, **kwargs):
        self.lookback = lookback
        self.threshold = threshold
        super().__init__(lookback=lookback, threshold=threshold, **kwargs)
    
    def _validate_parameters(self) -> None:
        """Validate momentum strategy parameters."""
        if self.lookback <= 0:
            raise ValidationError(
                "Lookback period must be positive",
                field="lookback",
                value=self.lookback
            )
        
        if not 0 < self.threshold <= 1.0:
            raise ValidationError(
                "Threshold must be between 0 and 1",
                field="threshold",
                value=self.threshold
            )
    
    def run(self, data: pd.DataFrame) -> pd.Series:
        """Generate momentum signals."""
        try:
            # Validate input data
            if 'close' not in data.columns:
                raise StrategyError(
                    "Missing 'close' column in data",
                    strategy_name=self.__class__.__name__,
                    context={'available_columns': list(data.columns)}
                )
            
            # Calculate returns with validation
            if len(data) <= self.lookback:
                raise StrategyError(
                    f"Insufficient data for momentum calculation: need > {self.lookback}, got {len(data)}",
                    strategy_name=self.__class__.__name__,
                    context={'required': self.lookback + 1, 'available': len(data)}
                )
            
            returns = data['close'].pct_change(self.lookback)
            
            # Check for valid returns
            if returns.isna().all():
                raise StrategyError(
                    "All momentum returns are NaN",
                    strategy_name=self.__class__.__name__,
                    context={'lookback': self.lookback, 'data_length': len(data)}
                )
            
            # Initialize signals
            signals = pd.Series(index=data.index, dtype=int)
            signals[:] = 0
            
            # Generate momentum signals with safe comparisons
            valid_returns = returns.dropna()
            if len(valid_returns) > 0:
                # Buy on strong positive momentum
                signals[returns > self.threshold] = 1
                # Sell on strong negative momentum
                signals[returns < -self.threshold] = -1
            
            return signals.fillna(0)
            
        except Exception as e:
            if isinstance(e, StrategyError):
                raise
            raise StrategyError(
                f"Failed to generate momentum signals",
                strategy_name=self.__class__.__name__,
                context={
                    'lookback': self.lookback,
                    'threshold': self.threshold,
                    'data_shape': data.shape,
                    'error': str(e)
                }
            ) from e
    
    def get_required_periods(self) -> int:
        """Need lookback + 1 periods."""
        return self.lookback + 1


class MeanReversionStrategy(BacktestStrategy):
    """Mean reversion trading strategy."""
    
    def __init__(self, period: int = 20, num_std: float = 2.0, **kwargs):
        self.period = period
        self.num_std = num_std
        super().__init__(period=period, num_std=num_std, **kwargs)
    
    def _validate_parameters(self) -> None:
        """Validate mean reversion strategy parameters."""
        if self.period <= 0:
            raise ValidationError(
                "Period must be positive",
                field="period",
                value=self.period
            )
        
        if self.num_std <= 0:
            raise ValidationError(
                "Number of standard deviations must be positive",
                field="num_std",
                value=self.num_std
            )
    
    def run(self, data: pd.DataFrame) -> pd.Series:
        """Generate mean reversion signals using Bollinger Bands."""
        try:
            # Validate input data
            if 'close' not in data.columns:
                raise StrategyError(
                    "Missing 'close' column in data",
                    strategy_name=self.__class__.__name__,
                    context={'available_columns': list(data.columns)}
                )
            
            # Calculate Bollinger Bands with error handling
            mean = self._safe_rolling_calculation(data['close'], self.period, 'mean')
            std = self._safe_rolling_calculation(data['close'], self.period, 'std')
            
            # Check for valid calculations
            if mean.isna().all() or std.isna().all():
                raise StrategyError(
                    "Bollinger Band calculations resulted in all NaN values",
                    strategy_name=self.__class__.__name__,
                    context={'period': self.period, 'data_length': len(data)}
                )
            
            upper_band = mean + (std * self.num_std)
            lower_band = mean - (std * self.num_std)
            
            # Initialize signals
            signals = pd.Series(index=data.index, dtype=int)
            signals[:] = 0
            
            # Generate signals with safe comparisons
            valid_bands = ~(upper_band.isna() | lower_band.isna())
            
            # Buy when price touches lower band (oversold)
            buy_condition = valid_bands & (data['close'] <= lower_band)
            signals[buy_condition] = 1
            
            # Sell when price touches upper band (overbought)
            sell_condition = valid_bands & (data['close'] >= upper_band)
            signals[sell_condition] = -1
            
            return signals.fillna(0)
            
        except Exception as e:
            if isinstance(e, StrategyError):
                raise
            raise StrategyError(
                f"Failed to generate mean reversion signals",
                strategy_name=self.__class__.__name__,
                context={
                    'period': self.period,
                    'num_std': self.num_std,
                    'data_shape': data.shape,
                    'error': str(e)
                }
            ) from e
    
    def get_required_periods(self) -> int:
        """Need period periods for stats."""
        return self.period


class VolatilityStrategy(BacktestStrategy):
    """Volatility-based trading strategy."""
    
    def __init__(self, period: int = 20, vol_threshold: float = 0.02, **kwargs):
        self.period = period
        self.vol_threshold = vol_threshold
        super().__init__(period=period, vol_threshold=vol_threshold, **kwargs)
    
    def _validate_parameters(self) -> None:
        """Validate volatility strategy parameters."""
        if self.period <= 0:
            raise ValidationError(
                "Period must be positive",
                field="period",
                value=self.period
            )
        
        if self.vol_threshold <= 0:
            raise ValidationError(
                "Volatility threshold must be positive",
                field="vol_threshold",
                value=self.vol_threshold
            )
    
    def run(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals based on volatility."""
        try:
            # Validate input data
            if 'close' not in data.columns:
                raise StrategyError(
                    "Missing 'close' column in data",
                    strategy_name=self.__class__.__name__,
                    context={'available_columns': list(data.columns)}
                )
            
            # Calculate returns with validation
            if len(data) <= self.period:
                raise StrategyError(
                    f"Insufficient data for volatility calculation: need > {self.period}, got {len(data)}",
                    strategy_name=self.__class__.__name__,
                    context={'required': self.period + 1, 'available': len(data)}
                )
            
            returns = data['close'].pct_change()
            volatility = self._safe_rolling_calculation(returns, self.period, 'std')
            
            # Check for valid volatility values
            if volatility.isna().all():
                raise StrategyError(
                    "All volatility values are NaN",
                    strategy_name=self.__class__.__name__,
                    context={'period': self.period, 'data_length': len(data)}
                )
            
            # Initialize signals
            signals = pd.Series(index=data.index, dtype=int)
            signals[:] = 0
            
            # Trade only in low volatility environments
            valid_vol = ~volatility.isna()
            low_vol = valid_vol & (volatility < self.vol_threshold)
            
            # Use simple momentum in low vol periods
            momentum = data['close'].pct_change(self.period)
            valid_momentum = ~momentum.isna()
            
            # Generate signals with safe conditions
            buy_condition = low_vol & valid_momentum & (momentum > 0.01)
            sell_condition = low_vol & valid_momentum & (momentum < -0.01)
            
            signals[buy_condition] = 1
            signals[sell_condition] = -1
            
            return signals.fillna(0)
            
        except Exception as e:
            if isinstance(e, StrategyError):
                raise
            raise StrategyError(
                f"Failed to generate volatility signals",
                strategy_name=self.__class__.__name__,
                context={
                    'period': self.period,
                    'vol_threshold': self.vol_threshold,
                    'data_shape': data.shape,
                    'error': str(e)
                }
            ) from e
    
    def get_required_periods(self) -> int:
        """Need period + 1 for volatility calc."""
        return self.period + 1


class BreakoutStrategy(BacktestStrategy):
    """Price breakout trading strategy."""
    
    def __init__(self, lookback: int = 20, **kwargs):
        self.lookback = lookback
        super().__init__(lookback=lookback, **kwargs)
    
    def _validate_parameters(self) -> None:
        """Validate breakout strategy parameters."""
        if self.lookback <= 0:
            raise ValidationError(
                "Lookback period must be positive",
                field="lookback",
                value=self.lookback
            )
    
    def run(self, data: pd.DataFrame) -> pd.Series:
        """Generate breakout signals."""
        try:
            # Validate input data
            required_cols = ['high', 'low', 'close']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                raise StrategyError(
                    f"Missing required columns: {missing_cols}",
                    strategy_name=self.__class__.__name__,
                    context={'available_columns': list(data.columns), 'missing': missing_cols}
                )
            
            # Calculate rolling highs and lows with error handling
            high_rolling = self._safe_rolling_calculation(data['high'], self.lookback, 'max')
            low_rolling = self._safe_rolling_calculation(data['low'], self.lookback, 'min')
            
            # Check for valid calculations
            if high_rolling.isna().all() or low_rolling.isna().all():
                raise StrategyError(
                    "All rolling high/low values are NaN",
                    strategy_name=self.__class__.__name__,
                    context={'lookback': self.lookback, 'data_length': len(data)}
                )
            
            # Initialize signals
            signals = pd.Series(index=data.index, dtype=int)
            signals[:] = 0
            
            # Generate breakout signals with safe comparisons
            prev_high = high_rolling.shift(1)
            prev_low = low_rolling.shift(1)
            
            valid_high = ~prev_high.isna()
            valid_low = ~prev_low.isna()
            
            # Buy on upward breakout
            buy_condition = valid_high & (data['close'] > prev_high)
            signals[buy_condition] = 1
            
            # Sell on downward breakout
            sell_condition = valid_low & (data['close'] < prev_low)
            signals[sell_condition] = -1
            
            return signals.fillna(0)
            
        except Exception as e:
            if isinstance(e, StrategyError):
                raise
            raise StrategyError(
                f"Failed to generate breakout signals",
                strategy_name=self.__class__.__name__,
                context={
                    'lookback': self.lookback,
                    'data_shape': data.shape,
                    'error': str(e)
                }
            ) from e
    
    def get_required_periods(self) -> int:
        """Need lookback + 1 periods."""
        return self.lookback + 1


# Local strategy factory - no external dependencies
STRATEGY_MAP = {
    'SimpleMAStrategy': SimpleMAStrategy,
    'MomentumStrategy': MomentumStrategy,
    'MeanReversionStrategy': MeanReversionStrategy,
    'VolatilityStrategy': VolatilityStrategy,
    'BreakoutStrategy': BreakoutStrategy,
}


def create_local_strategy(name: str, **params) -> BacktestStrategy:
    """
    Create a strategy instance locally.
    
    Args:
        name: Strategy name
        **params: Strategy parameters
        
    Returns:
        Strategy instance
        
    Raises:
        StrategyError: If strategy creation fails
        ValidationError: If parameters are invalid
    """
    logger.debug(
        f"Creating strategy '{name}' with parameters: {params}",
        extra={'strategy_name': name, 'params': params}
    )
    
    # Validate strategy name
    if not name or not isinstance(name, str):
        raise ValidationError(
            "Strategy name must be a non-empty string",
            field="strategy_name",
            value=name
        )
    
    # Check if strategy exists
    if name not in STRATEGY_MAP:
        available = ', '.join(STRATEGY_MAP.keys())
        raise StrategyError(
            f"Unknown strategy '{name}'",
            strategy_name=name,
            context={
                'available_strategies': list(STRATEGY_MAP.keys()),
                'requested_strategy': name
            }
        )
    
    # Create strategy instance with error handling
    try:
        strategy_class = STRATEGY_MAP[name]
        strategy = strategy_class(**params)
        
        logger.info(
            f"Successfully created {name} strategy",
            extra={
                'strategy_name': name,
                'strategy_class': strategy_class.__name__,
                'params': params
            }
        )
        
        return strategy
        
    except (ValidationError, StrategyError):
        # Re-raise validation and strategy errors as-is
        raise
    except Exception as e:
        # Wrap other exceptions in StrategyError
        raise StrategyError(
            f"Failed to create strategy '{name}'",
            strategy_name=name,
            context={
                'params': params,
                'strategy_class': STRATEGY_MAP.get(name, {}).get('__name__', 'Unknown'),
                'creation_error': str(e)
            }
        ) from e


def list_local_strategies() -> list:
    """List available local strategies."""
    return list(STRATEGY_MAP.keys())


def validate_strategy_parameters(strategy_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate strategy parameters before creation.
    
    Args:
        strategy_name: Name of the strategy
        params: Parameters to validate
        
    Returns:
        Validated parameters
        
    Raises:
        StrategyError: If validation fails
    """
    try:
        # Create a temporary instance to validate parameters
        strategy = create_local_strategy(strategy_name, **params)
        return strategy.params
    except Exception as e:
        raise StrategyError(
            f"Parameter validation failed for {strategy_name}",
            strategy_name=strategy_name,
            context={'params': params, 'validation_error': str(e)}
        ) from e


def get_strategy_info(strategy_name: str) -> Dict[str, Any]:
    """
    Get information about a strategy.
    
    Args:
        strategy_name: Name of the strategy
        
    Returns:
        Dictionary with strategy information
    """
    if strategy_name not in STRATEGY_MAP:
        return {
            'name': strategy_name,
            'exists': False,
            'error': 'Strategy not found'
        }
    
    try:
        strategy_class = STRATEGY_MAP[strategy_name]
        # Create instance with default parameters to get info
        temp_strategy = strategy_class()
        
        return {
            'name': strategy_name,
            'exists': True,
            'class_name': strategy_class.__name__,
            'doc': strategy_class.__doc__ or 'No description available',
            'required_periods': temp_strategy.get_required_periods(),
            'default_params': temp_strategy.params
        }
    except Exception as e:
        return {
            'name': strategy_name,
            'exists': True,
            'error': f"Failed to get strategy info: {str(e)}"
        }