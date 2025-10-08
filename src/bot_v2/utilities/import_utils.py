"""Utilities for organizing and optimizing imports."""

from __future__ import annotations

import importlib
import sys
import time
from typing import Any, Callable, TypeVar

T = TypeVar("T")


class LazyImport:
    """Lazy import wrapper that defers import until first access."""
    
    def __init__(self, module_path: str, attribute: str | None = None) -> None:
        """Initialize lazy import.
        
        Args:
            module_path: Full module path (e.g., 'package.module')
            attribute: Attribute to import from module (optional)
        """
        self.module_path = module_path
        self.attribute = attribute
        self._module: Any = None
        self._loaded = False
        
    def _load(self) -> Any:
        """Load the module/attribute on first access."""
        if not self._loaded:
            start_time = time.time()
            self._module = importlib.import_module(self.module_path)
            
            if self.attribute:
                self._module = getattr(self._module, self.attribute)
                
            self._loaded = True
            load_time = time.time() - start_time
            
            # Log slow imports
            if load_time > 0.1:  # 100ms threshold
                import logging
                logger = logging.getLogger(__name__)
                logger.debug(f"Slow import: {self.module_path} took {load_time:.3f}s")
                
        return self._module
        
    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to loaded module."""
        module = self._load()
        return getattr(module, name)
        
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Allow calling if imported object is callable."""
        module = self._load()
        return module(*args, **kwargs)
        
    def __repr__(self) -> str:
        """String representation showing import status."""
        status = "loaded" if self._loaded else "lazy"
        attr_part = f".{self.attribute}" if self.attribute else ""
        return f"<LazyImport {self.module_path}{attr_part} [{status}]>"


def lazy_import(module_path: str, attribute: str | None = None) -> LazyImport:
    """Create a lazy import wrapper.
    
    Args:
        module_path: Full module path
        attribute: Attribute to import from module
        
    Returns:
        LazyImport wrapper
    """
    return LazyImport(module_path, attribute)


class OptionalImport:
    """Optional import that gracefully handles missing dependencies."""
    
    def __init__(self, module_path: str, attribute: str | None = None) -> None:
        """Initialize optional import.
        
        Args:
            module_path: Full module path
            attribute: Attribute to import from module
        """
        self.module_path = module_path
        self.attribute = attribute
        self._module: Any = None
        self._available = False
        self._attempted = False
        
    def _try_load(self) -> Any:
        """Attempt to load the module/attribute."""
        if not self._attempted:
            try:
                self._module = importlib.import_module(self.module_path)
                
                if self.attribute:
                    self._module = getattr(self._module, self.attribute)
                    
                self._available = True
            except ImportError:
                self._available = False
                
            self._attempted = True
            
        return self._module if self._available else None
        
    def is_available(self) -> bool:
        """Check if the optional dependency is available."""
        self._try_load()
        return self._available
        
    def get(self, default: Any = None) -> Any:
        """Get the imported object or default if not available.
        
        Args:
            default: Default value if import fails
            
        Returns:
            Imported object or default
        """
        result = self._try_load()
        return result if result is not None else default
        
    def require(self, error_message: str | None = None) -> Any:
        """Get the imported object or raise ImportError if not available.
        
        Args:
            error_message: Custom error message
            
        Returns:
            Imported object
            
        Raises:
            ImportError: If the optional dependency is not available
        """
        if not self.is_available():
            msg = error_message or f"Optional dependency {self.module_path} is required but not available"
            raise ImportError(msg)
            
        return self._module  # type: ignore[return-value]
        
    def __getattr__(self, name: str) -> Any:
        """Forward attribute access if available."""
        module = self._try_load()
        if module is None:
            raise AttributeError(f"Optional import {self.module_path} is not available")
            
        return getattr(module, name)
        
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Allow calling if imported object is callable."""
        module = self.require()
        return module(*args, **kwargs)
        
    def __repr__(self) -> str:
        """String representation showing availability status."""
        status = "available" if self.is_available() else "unavailable"
        attr_part = f".{self.attribute}" if self.attribute else ""
        return f"<OptionalImport {self.module_path}{attr_part} [{status}]>"


def optional_import(module_path: str, attribute: str | None = None) -> OptionalImport:
    """Create an optional import wrapper.
    
    Args:
        module_path: Full module path
        attribute: Attribute to import from module
        
    Returns:
        OptionalImport wrapper
    """
    return OptionalImport(module_path, attribute)


def conditional_import(
    condition: bool,
    module_path: str,
    attribute: str | None = None,
) -> LazyImport | OptionalImport:
    """Create conditional import based on a condition.
    
    Args:
        condition: Whether to create lazy or optional import
        module_path: Full module path
        attribute: Attribute to import from module
        
    Returns:
        LazyImport if condition is True, OptionalImport otherwise
    """
    if condition:
        return lazy_import(module_path, attribute)
    else:
        return optional_import(module_path, attribute)


# Common optional dependencies for the trading system
pandas = optional_import("pandas")
numpy = optional_import("numpy")
matplotlib = optional_import("matplotlib")
plotly = optional_import("plotly")
scipy = optional_import("scipy")
sklearn = optional_import("sklearn")

# Heavy imports that should be lazy
tensorflow = lazy_import("tensorflow")
torch = lazy_import("torch")
cvxpy = lazy_import("cvxpy")


def get_import_stats() -> dict[str, Any]:
    """Get statistics about loaded modules.
    
    Returns:
        Dictionary with import statistics
    """
    loaded_modules = list(sys.modules.keys())
    
    # Categorize modules
    stdlib_modules = []
    third_party_modules = []
    local_modules = []
    
    for module_name in loaded_modules:
        if module_name.startswith("_") or "." not in module_name:
            if module_name in sys.builtin_module_names:
                stdlib_modules.append(module_name)
        elif module_name.startswith(("bot_v2", "tests")):
            local_modules.append(module_name)
        else:
            third_party_modules.append(module_name)
            
    return {
        "total_modules": len(loaded_modules),
        "stdlib_modules": len(stdlib_modules),
        "third_party_modules": len(third_party_modules),
        "local_modules": len(local_modules),
        "memory_usage": sys.getsizeof(sys.modules),
    }


def optimize_imports() -> None:
    """Optimize imports by cleaning up unused modules."""
    # This is a placeholder for future optimization logic
    # Could include:
    # - Detecting unused modules
    # - Unloading heavy modules that are no longer needed
    # - Compiling .pyc files for faster startup
    pass


class ImportProfiler:
    """Profile import times to identify slow imports."""
    
    def __init__(self) -> None:
        self.import_times: dict[str, float] = {}
        self.original_import = __builtins__['__import__']
        
    def start_profiling(self) -> None:
        """Start profiling imports."""
        __builtins__['__import__'] = self._profiled_import
        
    def stop_profiling(self) -> None:
        """Stop profiling imports."""
        __builtins__['__import__'] = self.original_import
        
    def _profiled_import(self, name: str, globals: Any = None, locals: Any = None, fromlist: Any = None, level: int = 0) -> Any:
        """Profile import function."""
        start_time = time.time()
        try:
            result = self.original_import(name, globals, locals, fromlist, level)
            return result
        finally:
            import_time = time.time() - start_time
            self.import_times[name] = import_time
            
    def get_slow_imports(self, threshold: float = 0.1) -> list[tuple[str, float]]:
        """Get imports slower than threshold.
        
        Args:
            threshold: Time threshold in seconds
            
        Returns:
            List of (module_name, import_time) tuples
        """
        return [
            (name, time_taken)
            for name, time_taken in self.import_times.items()
            if time_taken > threshold
        ]
        
    def print_report(self, threshold: float = 0.1) -> None:
        """Print import performance report.
        
        Args:
            threshold: Time threshold for highlighting slow imports
        """
        print("Import Performance Report")
        print("=" * 40)
        
        slow_imports = self.get_slow_imports(threshold)
        
        if slow_imports:
            print(f"Slow imports (> {threshold}s):")
            for name, time_taken in sorted(slow_imports, key=lambda x: x[1], reverse=True):
                print(f"  {name}: {time_taken:.3f}s")
        else:
            print(f"No imports slower than {threshold}s")
            
        print(f"\nTotal imports profiled: {len(self.import_times)}")
        if self.import_times:
            total_time = sum(self.import_times.values())
            avg_time = total_time / len(self.import_times)
            print(f"Total import time: {total_time:.3f}s")
            print(f"Average import time: {avg_time:.3f}s")


# Decorator for functions with heavy imports
def with_lazy_imports(**imports: LazyImport) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator that provides lazy imports as keyword arguments.
    
    Args:
        **imports: Mapping of parameter names to LazyImport objects
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Inject lazy imports into kwargs
            lazy_kwargs = {}
            for param_name, lazy_import in imports.items():
                lazy_kwargs[param_name] = lazy_import._load()
                
            # Merge with existing kwargs (lazy imports take precedence)
            merged_kwargs = {**kwargs, **lazy_kwargs}
            
            return func(*args, **merged_kwargs)
            
        return wrapper
    return decorator


# Utility function to check if we're in a test environment
def is_test_environment() -> bool:
    """Check if we're running in a test environment.
    
    Returns:
        True if in test environment
    """
    import sys
    return "pytest" in sys.modules or "unittest" in sys.modules


# Conditional imports based on environment
test_utils = optional_import("pytest") if is_test_environment() else None
dev_utils = optional_import("IPython") if not is_test_environment() else None
