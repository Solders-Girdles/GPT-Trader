# How to Add a New Feature

## Steps

### 1. Create New Slice
```bash
mkdir -p src/bot_v2/features/new_feature
```

### 2. Implement Core Functionality
```python
# src/bot_v2/features/new_feature/__init__.py
from .core import main_function
__all__ = ['main_function']

# src/bot_v2/features/new_feature/core.py
def main_function():
    """Your implementation here."""
    pass
```

### 3. Add Required Files
Each slice should have:
- `__init__.py` - Public API exports
- `core.py` - Main business logic
- `models.py` - Data structures (if needed)
- `utils.py` - Helper functions (if needed)
- `config.py` - Configuration (if needed)

### 4. Create Test File
```python
# src/bot_v2/test_new_feature.py
from src.bot_v2.features.new_feature import main_function

def test_new_feature():
    result = main_function()
    assert result is not None

if __name__ == "__main__":
    test_new_feature()
    print("✅ Tests pass")
```

### 5. Verify Isolation
```bash
# Should return nothing
grep -r "from bot_v2.features" src/bot_v2/features/new_feature/
```

### 6. Update Knowledge
- Add slice to STATE.json
- Update navigation docs if needed

## Remember
- Complete isolation - no imports from other slices
- Self-contained - include everything needed locally
- Test independently - slice must work alone