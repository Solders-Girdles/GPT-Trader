"""
Core interfaces that define the contracts for all system components.

These interfaces ensure components can be developed independently and will
work together seamlessly when integrated.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import pandas as pd
from dataclasses import dataclass


@dataclass
class ComponentConfig:
    """Base configuration for all components."""
    name: str
    enabled: bool = True
    config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.config is None:
            self.config = {}


class Component(ABC):
    """Base class for all system components."""
    
    def __init__(self, config: ComponentConfig):
        self.config = config
        self.name = config.name
        self.enabled = config.enabled
        self._listeners: List[Callable] = []
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the component."""
        pass
    
    @abstractmethod
    def shutdown(self) -> None:
        """Cleanup and shutdown the component."""
        pass
    
    def subscribe(self, callback: Callable) -> None:
        """Subscribe to component events."""
        self._listeners.append(callback)
    
    def notify(self, event: Any) -> None:
        """Notify all listeners of an event."""
        for listener in self._listeners:
            listener(event)


class IDataProvider(Component):
    """Interface for market data providers."""
    
    @abstractmethod
    def get_historical_data(
        self, 
        symbol: str, 
        start: datetime, 
        end: datetime,
        interval: str = '1d'
    ) -> pd.DataFrame:
        """Fetch historical market data."""
        pass
    
    @abstractmethod
    def get_realtime_data(self, symbols: List[str]) -> Dict[str, pd.Series]:
        """Fetch real-time market data."""
        pass
    
    @abstractmethod
    def subscribe_to_feed(self, symbols: List[str], callback: Callable) -> None:
        """Subscribe to real-time data feed."""
        pass


class IStrategy(Component):
    """Interface for trading strategies."""
    
    @abstractmethod
    def analyze(self, data: pd.DataFrame) -> pd.Series:
        """
        Analyze market data and generate signals.
        
        Returns:
            Series with values: 1 (buy), -1 (sell), 0 (hold)
        """
        pass
    
    @abstractmethod
    def get_required_history(self) -> int:
        """Get number of historical periods required."""
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters."""
        pass
    
    @abstractmethod
    def set_parameters(self, **kwargs) -> None:
        """Update strategy parameters."""
        pass


class IRiskManager(Component):
    """Interface for risk management."""
    
    @abstractmethod
    def validate_order(self, order: Any, portfolio: Any) -> tuple[bool, str]:
        """
        Validate an order against risk rules.
        
        Returns:
            (is_valid, reason_if_rejected)
        """
        pass
    
    @abstractmethod
    def calculate_position_size(
        self, 
        signal_strength: float,
        portfolio_value: float,
        current_positions: Dict[str, float]
    ) -> float:
        """Calculate appropriate position size based on risk."""
        pass
    
    @abstractmethod
    def get_risk_metrics(self, portfolio: Any) -> Dict[str, float]:
        """Calculate current risk metrics."""
        pass
    
    @abstractmethod
    def should_close_position(self, position: Any, current_price: float) -> bool:
        """Determine if a position should be closed for risk reasons."""
        pass


class IPortfolioAllocator(Component):
    """Interface for portfolio allocation."""
    
    @abstractmethod
    def allocate(
        self,
        signals: Dict[str, float],
        portfolio_value: float,
        current_positions: Dict[str, float],
        risk_constraints: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Allocate portfolio based on signals and constraints.
        
        Returns:
            Dict of symbol -> target_allocation (as fraction of portfolio)
        """
        pass
    
    @abstractmethod
    def rebalance_required(
        self,
        current_positions: Dict[str, float],
        target_allocations: Dict[str, float],
        threshold: float = 0.05
    ) -> bool:
        """Check if rebalancing is needed."""
        pass
    
    @abstractmethod
    def get_allocation_metrics(self) -> Dict[str, Any]:
        """Get allocation performance metrics."""
        pass


class IExecutor(Component):
    """Interface for trade execution."""
    
    @abstractmethod
    def execute_order(self, order: Any) -> Any:
        """
        Execute a trading order.
        
        Returns:
            Trade object with execution details
        """
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        pass
    
    @abstractmethod
    def get_order_status(self, order_id: str) -> str:
        """Get status of an order."""
        pass
    
    @abstractmethod
    def get_open_orders(self) -> List[Any]:
        """Get list of open orders."""
        pass


class IAnalytics(Component):
    """Interface for performance analytics."""
    
    @abstractmethod
    def calculate_returns(self, trades: List[Any]) -> pd.Series:
        """Calculate returns from trades."""
        pass
    
    @abstractmethod
    def calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio."""
        pass
    
    @abstractmethod
    def calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """Calculate maximum drawdown."""
        pass
    
    @abstractmethod
    def generate_report(self, trades: List[Any], positions: List[Any]) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        pass


class IBacktester(Component):
    """Interface for backtesting engine."""
    
    @abstractmethod
    def run(
        self,
        strategy: IStrategy,
        data_provider: IDataProvider,
        start_date: datetime,
        end_date: datetime,
        initial_capital: float = 10000
    ) -> Dict[str, Any]:
        """
        Run a backtest.
        
        Returns:
            Dict with results including trades, metrics, equity curve
        """
        pass
    
    @abstractmethod
    def optimize(
        self,
        strategy: IStrategy,
        parameter_ranges: Dict[str, List[Any]],
        metric: str = 'sharpe_ratio'
    ) -> Dict[str, Any]:
        """Optimize strategy parameters."""
        pass
    
    @abstractmethod
    def walk_forward_analysis(
        self,
        strategy: IStrategy,
        data_provider: IDataProvider,
        window_size: int,
        step_size: int
    ) -> Dict[str, Any]:
        """Perform walk-forward analysis."""
        pass