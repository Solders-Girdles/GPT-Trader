"""
Adapter to connect existing strategies to the new interface system.

This allows our already-built strategies to work seamlessly with the
new component architecture.
"""

from typing import Dict, Any
import pandas as pd
from core.interfaces import IStrategy, ComponentConfig
from strategies import StrategyBase


class StrategyAdapter(IStrategy):
    """
    Adapter that wraps existing strategies to work with IStrategy interface.
    
    This allows us to use all the strategies we've already built without
    modification.
    """
    
    def __init__(self, config: ComponentConfig, strategy: StrategyBase = None):
        """
        Initialize the adapter.
        
        Args:
            config: Component configuration
            strategy: The strategy to wrap (can be set later)
        """
        super().__init__(config)
        self.strategy = strategy
    
    def set_strategy(self, strategy: StrategyBase) -> None:
        """Set or change the wrapped strategy."""
        self.strategy = strategy
        self.name = f"{self.config.name}_{strategy.name}" if strategy else self.config.name
    
    def initialize(self) -> None:
        """Initialize the component."""
        if self.strategy and hasattr(self.strategy, 'initialize'):
            self.strategy.initialize()
    
    def shutdown(self) -> None:
        """Cleanup and shutdown the component."""
        if self.strategy and hasattr(self.strategy, 'shutdown'):
            self.strategy.shutdown()
    
    def analyze(self, data: pd.DataFrame) -> pd.Series:
        """
        Analyze market data and generate signals.
        
        Adapts the existing strategy's run() method to the IStrategy interface.
        """
        if not self.strategy:
            raise ValueError("No strategy set in adapter")
        
        # Use the existing strategy's run method
        return self.strategy.run(data)
    
    def get_required_history(self) -> int:
        """Get number of historical periods required."""
        if not self.strategy:
            return 0
        
        return self.strategy.get_required_periods()
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters."""
        if not self.strategy:
            return {}
        
        return self.strategy.parameters.copy()
    
    def set_parameters(self, **kwargs) -> None:
        """Update strategy parameters."""
        if self.strategy:
            self.strategy.set_parameters(**kwargs)