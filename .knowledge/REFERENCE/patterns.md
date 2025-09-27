# Code Patterns

## Import Pattern
```python
# Always use this pattern
from src.bot_v2.features.[slice] import function_name

# Never do cross-slice imports
# ❌ WRONG
from src.bot_v2.features.backtest import BacktestEngine
from src.bot_v2.features.analyze import indicators

# ✅ RIGHT - Import from one slice only
from src.bot_v2.features.backtest import BacktestEngine, calculate_indicators
```

## Slice Structure Pattern
```python
# features/[slice]/__init__.py
from .core import main_function, helper_function
from .models import DataModel
from .utils import utility_function

__all__ = ['main_function', 'helper_function', 'DataModel', 'utility_function']

# features/[slice]/core.py
def main_function():
    """Main business logic."""
    pass

# features/[slice]/models.py
from dataclasses import dataclass

@dataclass
class DataModel:
    """Data structures for this slice."""
    field: str

# features/[slice]/utils.py
def utility_function():
    """Helper functions for this slice."""
    pass
```

## Test Pattern
```python
# src/bot_v2/test_[slice].py
from src.bot_v2.features.[slice] import main_function

def test_basic_functionality():
    """Test the slice works."""
    result = main_function()
    assert result is not None

def test_isolation():
    """Verify no cross-slice imports."""
    import subprocess
    result = subprocess.run(
        ["grep", "-r", "from bot_v2.features", f"src/bot_v2/features/[slice]/"],
        capture_output=True
    )
    assert result.returncode != 0  # No matches found

if __name__ == "__main__":
    test_basic_functionality()
    test_isolation()
    print("✅ All tests pass")
```

## Local Implementation Pattern
```python
# Instead of importing from another slice
# ❌ WRONG
from src.bot_v2.features.analyze import calculate_sma

# ✅ RIGHT - Implement locally
def calculate_sma(data, period):
    """Local SMA calculation for this slice."""
    return data.rolling(period).mean()
```

## Configuration Pattern
```python
# features/[slice]/config.py
from dataclasses import dataclass

@dataclass
class SliceConfig:
    """Configuration for this slice only."""
    param1: str = "default"
    param2: int = 10
    
# Use in slice
from .config import SliceConfig

config = SliceConfig()
```

## Error Handling Pattern
```python
# Define slice-specific exceptions
class SliceError(Exception):
    """Base exception for this slice."""
    pass

class ValidationError(SliceError):
    """Validation failed in this slice."""
    pass

# Use in slice
def validate_input(data):
    if not data:
        raise ValidationError("Data cannot be empty")
```

## Public API Pattern
```python
# Only expose what's needed in __init__.py
# Keep internal functions private with underscore

# features/[slice]/core.py
def public_function():
    """This will be exposed."""
    return _private_helper()

def _private_helper():
    """This stays internal to slice."""
    pass

# features/[slice]/__init__.py
from .core import public_function  # Don't import _private_helper
__all__ = ['public_function']
```