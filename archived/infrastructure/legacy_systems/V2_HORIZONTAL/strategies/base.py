"""
Strategy base classes and interfaces.
Provides the foundation for all trading strategies in GPT-Trader V2.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass
import pandas as pd


@dataclass
class StrategyConfig:
    """Configuration for strategy instances."""
    name: str
    description: str
    parameters: Dict[str, Any]
    enabled: bool = True
    
    def validate(self) -> bool:
        """Validate configuration parameters."""
        return bool(self.name and self.parameters is not None)


@dataclass
class StrategyMetrics:
    """Metrics for strategy performance tracking."""
    total_signals: int = 0
    buy_signals: int = 0
    sell_signals: int = 0
    hold_signals: int = 0
    
    @property
    def signal_rate(self) -> float:
        """Percentage of non-hold signals."""
        if self.total_signals == 0:
            return 0.0
        return ((self.buy_signals + self.sell_signals) / self.total_signals) * 100
    
    def update_from_signals(self, signals: pd.Series) -> None:
        """Update metrics from signal series."""
        self.total_signals = len(signals)
        self.buy_signals = (signals == 1).sum()
        self.sell_signals = (signals == -1).sum()
        self.hold_signals = (signals == 0).sum()


class StrategyBase(ABC):
    """
    Abstract base class for all trading strategies.
    
    Defines the interface that all strategies must implement and provides
    common functionality for validation, metrics tracking, and error handling.
    """
    
    def __init__(self, config: StrategyConfig):
        """
        Initialize strategy with configuration.
        
        Args:
            config: Strategy configuration object
        """
        if not config.validate():
            raise ValueError(f"Invalid strategy configuration: {config}")
            
        self.config = config
        self.metrics = StrategyMetrics()
        self._is_initialized = False
        
    @property
    def name(self) -> str:
        """Get strategy name."""
        return self.config.name
    
    @property
    def description(self) -> str:
        """Get strategy description."""
        return self.config.description
    
    @property
    def parameters(self) -> Dict[str, Any]:
        """Get strategy parameters."""
        return self.config.parameters.copy()
    
    @property
    def is_enabled(self) -> bool:
        """Check if strategy is enabled."""
        return self.config.enabled
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals from market data.
        
        Args:
            data: DataFrame with OHLCV data, must have 'Close' column
            
        Returns:
            Series of signals: 1 = buy, -1 = sell, 0 = hold
            Index must match input data index
        """
        pass
    
    @abstractmethod
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate input data meets strategy requirements.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            True if data is valid for this strategy
        """
        pass
    
    @abstractmethod
    def get_required_periods(self) -> int:
        """
        Get minimum number of periods required for signal generation.
        
        Returns:
            Minimum periods needed for strategy to work
        """
        pass
    
    def set_parameters(self, **kwargs) -> None:
        """
        Update strategy parameters.
        
        Args:
            **kwargs: Parameter name-value pairs
        """
        for key, value in kwargs.items():
            if key in self.config.parameters:
                self.config.parameters[key] = value
            else:
                raise ValueError(f"Unknown parameter: {key}")
    
    def get_parameter(self, name: str, default: Any = None) -> Any:
        """
        Get a specific parameter value.
        
        Args:
            name: Parameter name
            default: Default value if parameter doesn't exist
            
        Returns:
            Parameter value
        """
        return self.config.parameters.get(name, default)
    
    def validate_signals(self, signals: pd.Series) -> bool:
        """
        Validate generated signals.
        
        Args:
            signals: Generated signal series
            
        Returns:
            True if signals are valid
        """
        # Check for valid signal values
        valid_values = {-1, 0, 1}
        if not signals.isin(valid_values).all():
            return False
            
        # Check for excessive signal frequency (basic sanity check)
        signal_rate = (signals != 0).mean()
        if signal_rate > 0.5:  # More than 50% signals seems excessive
            return False
            
        return True
    
    def run(self, data: pd.DataFrame) -> pd.Series:
        """
        Main entry point for signal generation with validation.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            Validated trading signals
        """
        # Validate input data
        if not self.validate_data(data):
            raise ValueError(f"Invalid data for strategy {self.name}")
        
        # Check minimum periods requirement
        if len(data) < self.get_required_periods():
            raise ValueError(
                f"Insufficient data: need {self.get_required_periods()} periods, "
                f"got {len(data)}"
            )
        
        # Generate signals
        signals = self.generate_signals(data)
        
        # Validate signals
        if not self.validate_signals(signals):
            raise ValueError(f"Invalid signals generated by {self.name}")
        
        # Update metrics
        self.metrics.update_from_signals(signals)
        
        return signals
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get strategy status and metrics.
        
        Returns:
            Dictionary with status information
        """
        return {
            'name': self.name,
            'description': self.description,
            'enabled': self.is_enabled,
            'parameters': self.parameters,
            'metrics': {
                'total_signals': self.metrics.total_signals,
                'buy_signals': self.metrics.buy_signals,
                'sell_signals': self.metrics.sell_signals,
                'hold_signals': self.metrics.hold_signals,
                'signal_rate': self.metrics.signal_rate
            }
        }
    
    def reset_metrics(self) -> None:
        """Reset strategy metrics."""
        self.metrics = StrategyMetrics()
    
    def __repr__(self) -> str:
        """String representation of strategy."""
        return f"{self.__class__.__name__}(name='{self.name}', enabled={self.is_enabled})"