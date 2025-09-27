"""
Component registry for dependency injection and management.

This provides a central registry for all system components, enabling
dependency injection and component lifecycle management.
"""

from typing import Any, Dict, List, Optional, Type, TypeVar
from .interfaces import Component, ComponentConfig
import inspect


T = TypeVar('T', bound=Component)


class ComponentRegistry:
    """
    Central registry for system components.
    
    Manages component registration, dependency injection, and lifecycle.
    """
    
    def __init__(self):
        self._components: Dict[str, Component] = {}
        self._component_types: Dict[str, Type[Component]] = {}
        self._dependencies: Dict[str, List[str]] = {}
        self._initialized: Dict[str, bool] = {}
    
    def register_type(self, name: str, component_type: Type[Component]) -> None:
        """
        Register a component type.
        
        Args:
            name: Name to register the component type under
            component_type: The component class
        """
        if not issubclass(component_type, Component):
            raise ValueError(f"{component_type} must be a subclass of Component")
        self._component_types[name] = component_type
    
    def register_instance(self, name: str, component: Component) -> None:
        """
        Register a component instance.
        
        Args:
            name: Name to register the component under
            component: The component instance
        """
        if not isinstance(component, Component):
            raise ValueError(f"{component} must be an instance of Component")
        
        self._components[name] = component
        self._initialized[name] = False
        
        # Auto-detect dependencies from constructor
        self._detect_dependencies(name, component)
    
    def _detect_dependencies(self, name: str, component: Component) -> None:
        """Auto-detect component dependencies from constructor signature."""
        # Get the constructor signature
        sig = inspect.signature(component.__class__.__init__)
        deps = []
        
        for param_name, param in sig.parameters.items():
            if param_name == 'self' or param_name == 'config':
                continue
            
            # Check if parameter type is a Component subclass
            if param.annotation and inspect.isclass(param.annotation):
                if issubclass(param.annotation, Component):
                    deps.append(param_name)
        
        self._dependencies[name] = deps
    
    def get(self, name: str, component_type: Type[T] = None) -> T:
        """
        Get a component by name.
        
        Args:
            name: Name of the component
            component_type: Expected type (for type checking)
            
        Returns:
            The component instance
        """
        if name not in self._components:
            # Try to create from registered type
            if name in self._component_types:
                self._create_component(name)
            else:
                raise KeyError(f"Component '{name}' not found")
        
        component = self._components[name]
        
        # Initialize if needed
        if not self._initialized.get(name, False):
            self._initialize_component(name)
        
        # Type check if requested
        if component_type and not isinstance(component, component_type):
            raise TypeError(f"Component '{name}' is not of type {component_type}")
        
        return component
    
    def _create_component(self, name: str) -> None:
        """Create a component from registered type."""
        component_type = self._component_types[name]
        
        # Create default config
        config = ComponentConfig(name=name)
        
        # Resolve dependencies
        deps = {}
        sig = inspect.signature(component_type.__init__)
        for param_name, param in sig.parameters.items():
            if param_name in ['self', 'config']:
                continue
            
            # Try to resolve dependency
            if param_name in self._components or param_name in self._component_types:
                deps[param_name] = self.get(param_name)
        
        # Create instance
        if deps:
            component = component_type(config, **deps)
        else:
            component = component_type(config)
        
        self.register_instance(name, component)
    
    def _initialize_component(self, name: str) -> None:
        """Initialize a component and its dependencies."""
        # Initialize dependencies first
        for dep in self._dependencies.get(name, []):
            if dep in self._components and not self._initialized.get(dep, False):
                self._initialize_component(dep)
        
        # Initialize component
        component = self._components[name]
        component.initialize()
        self._initialized[name] = True
    
    def create(
        self, 
        name: str, 
        component_type: Type[T], 
        config: Optional[ComponentConfig] = None,
        **kwargs
    ) -> T:
        """
        Create and register a new component.
        
        Args:
            name: Name for the component
            component_type: Type of component to create
            config: Configuration for the component
            **kwargs: Additional arguments for component constructor
            
        Returns:
            The created component
        """
        if config is None:
            config = ComponentConfig(name=name)
        
        component = component_type(config, **kwargs)
        self.register_instance(name, component)
        return component
    
    def remove(self, name: str) -> None:
        """
        Remove a component from the registry.
        
        Args:
            name: Name of component to remove
        """
        if name in self._components:
            # Shutdown if initialized
            if self._initialized.get(name, False):
                self._components[name].shutdown()
            
            # Remove from registry
            del self._components[name]
            if name in self._initialized:
                del self._initialized[name]
            if name in self._dependencies:
                del self._dependencies[name]
    
    def list_components(self) -> List[str]:
        """Get list of registered component names."""
        return list(self._components.keys())
    
    def list_types(self) -> List[str]:
        """Get list of registered component type names."""
        return list(self._component_types.keys())
    
    def get_dependencies(self, name: str) -> List[str]:
        """Get dependencies for a component."""
        return self._dependencies.get(name, [])
    
    def is_initialized(self, name: str) -> bool:
        """Check if a component is initialized."""
        return self._initialized.get(name, False)
    
    def initialize_all(self) -> None:
        """Initialize all registered components."""
        for name in self._components:
            if not self._initialized.get(name, False):
                self._initialize_component(name)
    
    def shutdown_all(self) -> None:
        """Shutdown all initialized components."""
        # Shutdown in reverse order of initialization
        for name in reversed(list(self._components.keys())):
            if self._initialized.get(name, False):
                self._components[name].shutdown()
                self._initialized[name] = False
    
    def clear(self) -> None:
        """Clear all components from registry."""
        self.shutdown_all()
        self._components.clear()
        self._component_types.clear()
        self._dependencies.clear()
        self._initialized.clear()


# Global registry instance
_registry = ComponentRegistry()


def get_component(name: str, component_type: Type[T] = None) -> T:
    """Get a component from the global registry."""
    return _registry.get(name, component_type)


def register_component(name: str, component: Component) -> None:
    """Register a component in the global registry."""
    _registry.register_instance(name, component)


def register_component_type(name: str, component_type: Type[Component]) -> None:
    """Register a component type in the global registry."""
    _registry.register_type(name, component_type)


def list_components() -> List[str]:
    """List all registered components."""
    return _registry.list_components()


def get_registry() -> ComponentRegistry:
    """Get the global component registry."""
    return _registry