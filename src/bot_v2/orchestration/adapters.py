"""
Adapter classes to standardize slice interfaces for the orchestrator
"""
from typing import Any, Dict, Optional, Union, List
import logging
import pandas as pd
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class SliceAdapter(ABC):
    """Base class for slice adapters"""
    
    def __init__(self, slice_module: Any, slice_name: str):
        self.module = slice_module
        self.slice_name = slice_name
        self.logger = logging.getLogger(f"adapter.{slice_name}")
    
    @abstractmethod
    def call(self, *args, **kwargs) -> Any:
        """Standard interface for calling the slice"""
        pass
    
    def is_available(self) -> bool:
        """Check if the slice is available and functional"""
        return self.module is not None


class DataAdapter(SliceAdapter):
    """Adapter for data slice - handles different data provider patterns"""
    
    def get_data(self, symbol: str, period: str = "60d") -> Optional[pd.DataFrame]:
        """Standardized data retrieval interface"""
        if not self.is_available():
            self.logger.warning("Data slice not available")
            return None
        
        try:
            # Try get_data_provider pattern first (most common in our slices)
            if hasattr(self.module, 'get_data_provider'):
                provider = self.module.get_data_provider()
                if hasattr(provider, 'get_historical_data'):
                    return provider.get_historical_data(symbol, period)
                elif hasattr(provider, 'get_data'):
                    return provider.get_data(symbol, period)
            
            # Try direct get_data method
            if hasattr(self.module, 'get_data'):
                return self.module.get_data(symbol, period)
            
            # Try get_historical_data directly on module
            if hasattr(self.module, 'get_historical_data'):
                return self.module.get_historical_data(symbol, period)
            
            self.logger.error(f"No compatible data method found in {self.slice_name}")
            return None
            
        except Exception as e:
            self.logger.error(f"Error retrieving data for {symbol}: {e}")
            return None
    
    def call(self, symbol: str, period: str = "60d") -> Optional[pd.DataFrame]:
        """Standard interface implementation"""
        return self.get_data(symbol, period)


class AnalyzeAdapter(SliceAdapter):
    """Adapter for analyze slice - standardizes market analysis"""
    
    def analyze(self, symbol: str, data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Standardized analysis interface"""
        if not self.is_available():
            return {"error": "Analyze slice not available"}
        
        try:
            # Try standard analyze method
            if hasattr(self.module, 'analyze'):
                if data is not None:
                    return self.module.analyze(data, symbol)
                else:
                    return self.module.analyze(symbol)
            
            # Try analyze_symbol method
            if hasattr(self.module, 'analyze_symbol'):
                return self.module.analyze_symbol(symbol, data)
            
            # Try run_analysis method
            if hasattr(self.module, 'run_analysis'):
                return self.module.run_analysis(symbol, data)
            
            return {"error": f"No compatible analyze method found in {self.slice_name}"}
            
        except Exception as e:
            self.logger.error(f"Error analyzing {symbol}: {e}")
            return {"error": str(e)}
    
    def call(self, symbol: str, data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Standard interface implementation"""
        return self.analyze(symbol, data)


class BacktestAdapter(SliceAdapter):
    """Adapter for backtest slice - standardizes backtesting interface"""
    
    def backtest(self, symbol: str, strategy_config: Dict[str, Any]) -> Dict[str, Any]:
        """Standardized backtesting interface"""
        if not self.is_available():
            return {"error": "Backtest slice not available"}
        
        try:
            # Try standard backtest method
            if hasattr(self.module, 'backtest'):
                return self.module.backtest(symbol, strategy_config)
            
            # Try run_backtest method
            if hasattr(self.module, 'run_backtest'):
                return self.module.run_backtest(symbol, strategy_config)
            
            # Try backtest_strategy method
            if hasattr(self.module, 'backtest_strategy'):
                return self.module.backtest_strategy(symbol, strategy_config)
            
            return {"error": f"No compatible backtest method found in {self.slice_name}"}
            
        except Exception as e:
            self.logger.error(f"Error backtesting {symbol}: {e}")
            return {"error": str(e)}
    
    def call(self, symbol: str, strategy_config: Dict[str, Any]) -> Dict[str, Any]:
        """Standard interface implementation"""
        return self.backtest(symbol, strategy_config)


class OptimizeAdapter(SliceAdapter):
    """Adapter for optimize slice - standardizes parameter optimization"""
    
    def optimize(self, symbol: str, strategy_type: str, param_ranges: Dict[str, Any]) -> Dict[str, Any]:
        """Standardized optimization interface"""
        if not self.is_available():
            return {"error": "Optimize slice not available"}
        
        try:
            # Try standard optimize method
            if hasattr(self.module, 'optimize'):
                return self.module.optimize(symbol, strategy_type, param_ranges)
            
            # Try optimize_strategy method
            if hasattr(self.module, 'optimize_strategy'):
                return self.module.optimize_strategy(symbol, strategy_type, param_ranges)
            
            # Try run_optimization method
            if hasattr(self.module, 'run_optimization'):
                return self.module.run_optimization(symbol, strategy_type, param_ranges)
            
            return {"error": f"No compatible optimize method found in {self.slice_name}"}
            
        except Exception as e:
            self.logger.error(f"Error optimizing {symbol}: {e}")
            return {"error": str(e)}
    
    def call(self, symbol: str, strategy_type: str, param_ranges: Dict[str, Any]) -> Dict[str, Any]:
        """Standard interface implementation"""
        return self.optimize(symbol, strategy_type, param_ranges)


class MLStrategyAdapter(SliceAdapter):
    """Adapter for ML strategy selection slice"""
    
    def predict_best_strategy(self, symbol: str, market_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Standardized ML strategy prediction interface"""
        if not self.is_available():
            return {"strategy": "momentum", "confidence": 0.0, "error": "ML slice not available"}
        
        try:
            # Try standard predict_best_strategy method
            if hasattr(self.module, 'predict_best_strategy'):
                if market_data is not None:
                    return self.module.predict_best_strategy(symbol, market_data)
                else:
                    return self.module.predict_best_strategy(symbol)
            
            # Try select_strategy method
            if hasattr(self.module, 'select_strategy'):
                return self.module.select_strategy(symbol, market_data)
            
            # Try ml_strategy_selection method
            if hasattr(self.module, 'ml_strategy_selection'):
                return self.module.ml_strategy_selection(symbol, market_data)
            
            # Fallback to default strategy
            return {"strategy": "momentum", "confidence": 0.5, "error": f"No ML method found, using default"}
            
        except Exception as e:
            self.logger.error(f"Error predicting strategy for {symbol}: {e}")
            return {"strategy": "momentum", "confidence": 0.0, "error": str(e)}
    
    def call(self, symbol: str, market_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Standard interface implementation"""
        return self.predict_best_strategy(symbol, market_data)


class MarketRegimeAdapter(SliceAdapter):
    """Adapter for market regime detection slice"""
    
    def detect_regime(self, symbol: str, data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Standardized regime detection interface"""
        if not self.is_available():
            return {"regime": "unknown", "confidence": 0.0, "error": "Regime slice not available"}
        
        try:
            # Try standard detect_regime method
            if hasattr(self.module, 'detect_regime'):
                if data is not None:
                    return self.module.detect_regime(data, symbol)
                else:
                    return self.module.detect_regime(symbol)
            
            # Try get_market_regime method
            if hasattr(self.module, 'get_market_regime'):
                return self.module.get_market_regime(symbol, data)
            
            # Fallback to default regime
            return {"regime": "sideways_quiet", "confidence": 0.5, "error": "No regime method found, using default"}
            
        except Exception as e:
            self.logger.error(f"Error detecting regime for {symbol}: {e}")
            return {"regime": "unknown", "confidence": 0.0, "error": str(e)}
    
    def call(self, symbol: str, data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Standard interface implementation"""
        return self.detect_regime(symbol, data)


class PositionSizingAdapter(SliceAdapter):
    """Adapter for position sizing slice"""
    
    def calculate_position_size(self, symbol: str, confidence: float, portfolio_value: float, 
                                risk_tolerance: float = 0.02) -> Dict[str, Any]:
        """Standardized position sizing interface"""
        if not self.is_available():
            return {"position_size": 0.1, "error": "Position sizing slice not available"}
        
        try:
            # Try standard calculate_position_size method
            if hasattr(self.module, 'calculate_position_size'):
                return self.module.calculate_position_size(symbol, confidence, portfolio_value, risk_tolerance)
            
            # Try kelly_position_size method
            if hasattr(self.module, 'kelly_position_size'):
                return self.module.kelly_position_size(confidence, portfolio_value, risk_tolerance)
            
            # Fallback to simple percentage
            return {"position_size": min(0.1 * confidence, 0.2), "error": "No position sizing method found, using default"}
            
        except Exception as e:
            self.logger.error(f"Error calculating position size for {symbol}: {e}")
            return {"position_size": 0.05, "error": str(e)}
    
    def call(self, symbol: str, confidence: float, portfolio_value: float, 
             risk_tolerance: float = 0.02) -> Dict[str, Any]:
        """Standard interface implementation"""
        return self.calculate_position_size(symbol, confidence, portfolio_value, risk_tolerance)


class MonitorAdapter(SliceAdapter):
    """Adapter for monitoring slice"""
    
    def get_status(self) -> Dict[str, Any]:
        """Standardized monitoring interface"""
        if not self.is_available():
            return {"status": "unknown", "error": "Monitor slice not available"}
        
        try:
            # Try standard get_status method
            if hasattr(self.module, 'get_status'):
                return self.module.get_status()
            
            # Try monitor method
            if hasattr(self.module, 'monitor'):
                return self.module.monitor()
            
            # Try health_check method
            if hasattr(self.module, 'health_check'):
                return self.module.health_check()
            
            return {"status": "unknown", "error": "No monitoring method found"}
            
        except Exception as e:
            self.logger.error(f"Error getting monitor status: {e}")
            return {"status": "error", "error": str(e)}
    
    def log_event(self, event: Dict[str, Any]) -> bool:
        """Log an event to the monitoring system"""
        if not self.is_available():
            return False
        
        try:
            if hasattr(self.module, 'log_event'):
                self.module.log_event(event)
                return True
            elif hasattr(self.module, 'log'):
                self.module.log(event)
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error logging event: {e}")
            return False
    
    def call(self) -> Dict[str, Any]:
        """Standard interface implementation"""
        return self.get_status()


class TradingAdapter(SliceAdapter):
    """Base adapter for trading slices (paper_trade, live_trade)"""
    
    def execute_trade(self, symbol: str, action: str, quantity: float, 
                     strategy_info: Dict[str, Any]) -> Dict[str, Any]:
        """Standardized trading interface"""
        if not self.is_available():
            return {"success": False, "error": f"{self.slice_name} slice not available"}
        
        try:
            # Try slice-specific execute method
            if hasattr(self.module, f'execute_{self.slice_name}'):
                method = getattr(self.module, f'execute_{self.slice_name}')
                return method(symbol, action, quantity, strategy_info)
            
            # Try generic execute method
            if hasattr(self.module, 'execute'):
                return self.module.execute(symbol, action, quantity, strategy_info)
            
            # Try trade method
            if hasattr(self.module, 'trade'):
                return self.module.trade(symbol, action, quantity, strategy_info)
            
            return {"success": False, "error": f"No trading method found in {self.slice_name}"}
            
        except Exception as e:
            self.logger.error(f"Error executing {action} trade for {symbol}: {e}")
            return {"success": False, "error": str(e)}
    
    def call(self, symbol: str, action: str, quantity: float, 
             strategy_info: Dict[str, Any]) -> Dict[str, Any]:
        """Standard interface implementation"""
        return self.execute_trade(symbol, action, quantity, strategy_info)


class AdaptivePortfolioAdapter(SliceAdapter):
    """Adapter for adaptive portfolio slice"""
    
    def run_adaptive_strategy(self, portfolio_value: float, 
                             config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Standardized adaptive portfolio interface"""
        if not self.is_available():
            return {"success": False, "error": "Adaptive portfolio slice not available"}
        
        try:
            # Try standard run_adaptive_strategy method
            if hasattr(self.module, 'run_adaptive_strategy'):
                if config:
                    return self.module.run_adaptive_strategy(portfolio_value, config)
                else:
                    return self.module.run_adaptive_strategy(portfolio_value)
            
            # Try adaptive_portfolio method
            if hasattr(self.module, 'adaptive_portfolio'):
                return self.module.adaptive_portfolio(portfolio_value, config)
            
            return {"success": False, "error": "No adaptive portfolio method found"}
            
        except Exception as e:
            self.logger.error(f"Error running adaptive strategy: {e}")
            return {"success": False, "error": str(e)}
    
    def call(self, portfolio_value: float, 
             config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Standard interface implementation"""
        return self.run_adaptive_strategy(portfolio_value, config)


class AdapterFactory:
    """Factory for creating appropriate adapters for different slice types"""
    
    ADAPTER_MAP = {
        'data': DataAdapter,
        'analyze': AnalyzeAdapter,
        'backtest': BacktestAdapter,
        'optimize': OptimizeAdapter,
        'paper_trade': TradingAdapter,
        'live_trade': TradingAdapter,
        'monitor': MonitorAdapter,
        'ml_strategy': MLStrategyAdapter,
        'market_regime': MarketRegimeAdapter,
        'position_sizing': PositionSizingAdapter,
        'adaptive_portfolio': AdaptivePortfolioAdapter
    }
    
    @classmethod
    def create_adapter(cls, slice_name: str, slice_module: Any) -> SliceAdapter:
        """Create appropriate adapter for a slice"""
        adapter_class = cls.ADAPTER_MAP.get(slice_name, SliceAdapter)
        return adapter_class(slice_module, slice_name)
    
    @classmethod
    def create_adapters_for_registry(cls, registry) -> Dict[str, SliceAdapter]:
        """Create adapters for all slices in a registry"""
        adapters = {}
        
        for slice_name in registry.list_available_slices():
            slice_module = registry.get_slice(slice_name)
            if slice_module:
                adapters[slice_name] = cls.create_adapter(slice_name, slice_module)
        
        return adapters