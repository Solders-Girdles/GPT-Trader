"""
Strategy factory and registry for GPT-Trader V2.
Provides centralized strategy creation and management.
"""

from typing import Dict, Type, List, Any, Optional
from dataclasses import dataclass
import inspect

from .base import StrategyBase, StrategyConfig


@dataclass
class StrategyInfo:
    """Information about a registered strategy."""
    name: str
    class_type: Type[StrategyBase]
    description: str
    default_parameters: Dict[str, Any]
    required_parameters: List[str]
    
    @classmethod
    def from_strategy_class(cls, strategy_class: Type[StrategyBase]) -> 'StrategyInfo':
        """Create StrategyInfo from a strategy class."""
        # Get constructor signature to determine parameters
        sig = inspect.signature(strategy_class.__init__)
        params = {}
        required = []
        
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
                
            if param.default == inspect.Parameter.empty:
                required.append(param_name)
            else:
                params[param_name] = param.default
        
        return cls(
            name=strategy_class.__name__,
            class_type=strategy_class,
            description=strategy_class.__doc__ or "No description available",
            default_parameters=params,
            required_parameters=required
        )


class StrategyRegistry:
    """Registry for managing available strategies."""
    
    def __init__(self):
        self._strategies: Dict[str, StrategyInfo] = {}
        
    def register(self, strategy_class: Type[StrategyBase]) -> None:
        """
        Register a strategy class.
        
        Args:
            strategy_class: Strategy class to register
        """
        if not issubclass(strategy_class, StrategyBase):
            raise ValueError(f"{strategy_class} must inherit from StrategyBase")
        
        info = StrategyInfo.from_strategy_class(strategy_class)
        self._strategies[info.name] = info
        
    def unregister(self, name: str) -> bool:
        """
        Unregister a strategy.
        
        Args:
            name: Strategy name to unregister
            
        Returns:
            True if strategy was found and removed
        """
        return self._strategies.pop(name, None) is not None
    
    def get_strategy_info(self, name: str) -> Optional[StrategyInfo]:
        """
        Get information about a registered strategy.
        
        Args:
            name: Strategy name
            
        Returns:
            StrategyInfo or None if not found
        """
        return self._strategies.get(name)
    
    def list_strategies(self) -> List[str]:
        """Get list of registered strategy names."""
        return list(self._strategies.keys())
    
    def get_all_info(self) -> Dict[str, StrategyInfo]:
        """Get all registered strategy information."""
        return self._strategies.copy()
    
    def is_registered(self, name: str) -> bool:
        """Check if strategy is registered."""
        return name in self._strategies


class StrategyFactory:
    """Factory for creating strategy instances."""
    
    def __init__(self, registry: Optional[StrategyRegistry] = None):
        """
        Initialize factory with optional registry.
        
        Args:
            registry: Strategy registry to use, creates new one if None
        """
        self.registry = registry or StrategyRegistry()
        
    def create_strategy(
        self, 
        name: str, 
        config_name: Optional[str] = None,
        **kwargs
    ) -> StrategyBase:
        """
        Create a strategy instance.
        
        Args:
            name: Strategy class name
            config_name: Custom name for the strategy instance
            **kwargs: Strategy parameters
            
        Returns:
            Strategy instance
        """
        info = self.registry.get_strategy_info(name)
        if info is None:
            available = ", ".join(self.registry.list_strategies())
            raise ValueError(f"Unknown strategy: {name}. Available: {available}")
        
        # Merge default parameters with provided ones
        parameters = info.default_parameters.copy()
        parameters.update(kwargs)
        
        # Check for required parameters
        missing = set(info.required_parameters) - set(parameters.keys())
        if missing:
            raise ValueError(f"Missing required parameters for {name}: {missing}")
        
        # Create the strategy instance
        try:
            return info.class_type(**parameters)
        except Exception as e:
            raise ValueError(f"Failed to create strategy {name}: {e}") from e
    
    def create_from_config(self, config: StrategyConfig) -> StrategyBase:
        """
        Create strategy from configuration object.
        
        Args:
            config: Strategy configuration
            
        Returns:
            Strategy instance
        """
        # Extract strategy class name from config name if it contains class info
        # For now, assume config.name is the strategy class name
        return self.create_strategy(
            config.name,
            config_name=config.name,
            **config.parameters
        )
    
    def get_parameter_info(self, name: str) -> Dict[str, Any]:
        """
        Get parameter information for a strategy.
        
        Args:
            name: Strategy name
            
        Returns:
            Dictionary with parameter info
        """
        info = self.registry.get_strategy_info(name)
        if info is None:
            raise ValueError(f"Unknown strategy: {name}")
        
        return {
            'default_parameters': info.default_parameters,
            'required_parameters': info.required_parameters,
            'description': info.description
        }


# Global registry and factory instances
_global_registry = StrategyRegistry()
_global_factory = StrategyFactory(_global_registry)


def get_strategy_registry() -> StrategyRegistry:
    """Get the global strategy registry."""
    return _global_registry


def get_strategy_factory() -> StrategyFactory:
    """Get the global strategy factory."""
    return _global_factory


def register_strategy(strategy_class: Type[StrategyBase]) -> None:
    """Register a strategy class with the global registry."""
    _global_registry.register(strategy_class)


def create_strategy(name: str, **kwargs) -> StrategyBase:
    """Create a strategy using the global factory."""
    return _global_factory.create_strategy(name, **kwargs)


def list_available_strategies() -> List[str]:
    """List all available strategies."""
    return _global_registry.list_strategies()


def strategy_parameter_info(name: str) -> Dict[str, Any]:
    """Get parameter information for a strategy."""
    return _global_factory.get_parameter_info(name)