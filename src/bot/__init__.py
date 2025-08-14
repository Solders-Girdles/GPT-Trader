"""
GPT-Trader - AI-Powered Trading Strategy Platform

A comprehensive algorithmic trading research framework designed for rapid strategy development,
backtesting, optimization, and live trading.

Key Features:
- Modular Strategy Architecture
- Advanced Optimization Methods
- Live Trading with Paper Trading
- AI-Powered Strategy Discovery
- Comprehensive Analytics
"""

from typing import Any

__version__ = "2.0.0"
__author__ = "RJ + GPT-5"


# Core exports - use lazy importing for heavy dependencies
def _lazy_import_backtest() -> Any:
    """Lazy import for backtest engine to avoid matplotlib conflicts"""
    from .backtest.engine_portfolio import run_backtest

    return run_backtest


# Import lightweight components immediately
from .config import TradingConfig, get_config
from .strategy.base import Strategy

# Create a compatibility alias for old code
settings = get_config()


# Lazy imports for heavy components
def _get_run_backtest() -> Any:
    """Get run_backtest function with lazy loading"""
    return _lazy_import_backtest()


# Import other utility modules that don't conflict
try:
    from .exceptions import ConfigurationError, DataError, GPTTraderError, StrategyError
except ImportError:
    # Graceful fallback if modules aren't available
    class GPTTraderError(Exception):
        pass

    class ConfigurationError(GPTTraderError):
        pass

    class DataError(GPTTraderError):
        pass

    class StrategyError(GPTTraderError):
        pass


# Try to import other components gracefully
try:
    from .health import check_health, get_health_summary, health_checker, is_system_healthy
except ImportError:
    # Create stub functions if health module not available
    def health_checker() -> bool:
        return True

    def check_health() -> dict[str, str]:
        return {"status": "ok"}

    def is_system_healthy() -> bool:
        return True

    def get_health_summary() -> dict[str, str]:
        return {"status": "ok"}


try:
    from .performance import get_performance_summary, performance_monitor, profile_function
except ImportError:
    # Create stub functions if performance module not available
    def performance_monitor(func: Any) -> Any:
        return func

    def profile_function(func: Any) -> Any:
        return func

    def get_performance_summary() -> dict[str, str]:
        return {"status": "ok"}


try:
    from .optimization.engine import OptimizationEngine
except ImportError:
    # Stub class if optimization engine not available
    from typing import Any

    class OptimizationEngine:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass


try:
    from .live.production_orchestrator import ProductionOrchestrator
except ImportError:
    # Stub class if production orchestrator not available
    class ProductionOrchestrator:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass


# Provide run_backtest through lazy loading
def run_backtest(*args: Any, **kwargs: Any) -> Any:
    """Lazy-loaded run_backtest function"""
    actual_run_backtest = _get_run_backtest()
    return actual_run_backtest(*args, **kwargs)


__all__ = [
    # Core components
    "Strategy",
    "TradingConfig",
    "get_config",
    "run_backtest",
    "OptimizationEngine",
    "ProductionOrchestrator",
    "settings",
    # Exceptions
    "GPTTraderError",
    "ConfigurationError",
    "DataError",
    "StrategyError",
    # Performance monitoring
    "performance_monitor",
    "profile_function",
    "get_performance_summary",
    # Health monitoring
    "health_checker",
    "check_health",
    "is_system_healthy",
    "get_health_summary",
    # Package metadata
    "__version__",
    "__author__",
]
