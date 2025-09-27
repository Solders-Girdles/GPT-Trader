# ⚠️ DEPRECATED KNOWLEDGE LAYER

**This directory contains outdated information from August 2024.**

The project has migrated from Alpaca/Equities to Coinbase/Perpetuals.

**For current documentation, see: [docs/README.md](../docs/README.md)**

---

# 🚨 CRITICAL RULES - NEVER VIOLATE

## Rule 1: Complete Slice Isolation

### ❌ NEVER DO THIS
```python
# Cross-slice imports are FORBIDDEN
from src.bot_v2.features.backtest import BacktestEngine
from src.bot_v2.features.analyze import indicators  # VIOLATION!
```

### ✅ ALWAYS DO THIS
```python
# Import from ONE slice only
from src.bot_v2.features.backtest import BacktestEngine, calculate_indicators
```

### Why?
Each slice must be deployable independently. Cross-slice imports break this.

## Rule 2: Local Implementation

### ❌ NEVER DO THIS
```python
# Don't reach into other slices for utilities
from src.bot_v2.features.analyze.utils import helper_function
```

### ✅ ALWAYS DO THIS
```python
# Implement what you need locally
def helper_function():
    """Local implementation for this slice."""
    pass
```

### Why?
Duplication is better than dependencies. Each slice = ~500 tokens.

## Rule 3: No Shared Code

### ❌ NEVER DO THIS
```python
# Don't create shared/common directories
src/bot_v2/common/  # FORBIDDEN
src/bot_v2/shared/  # FORBIDDEN
src/bot_v2/utils/   # FORBIDDEN
```

### ✅ ALWAYS DO THIS
```python
# Each slice has its own utilities
src/bot_v2/features/[slice]/utils.py  # Local to slice
```

### Why?
Shared code creates coupling. We want zero coupling.

## Rule 4: Update STATE.json

### When to Update
- Adding new slice
- Changing slice status
- Modifying system metrics
- Fixing major bugs

### How to Update
```json
{
  "slices": {
    "new_slice": {
      "status": "operational",
      "test": "passing"
    }
  }
}
```

## Rule 5: Minimal File Creation

### ❌ NEVER CREATE
- Reports (use STATE.json)
- Analysis files (use STATE.json)
- Documentation (update .knowledge/)
- Temporary files

### ✅ ONLY CREATE
- Code in slices
- Tests for slices
- Updates to existing knowledge

## Rule 6: Test in Isolation

### Always Run
```bash
# After any change
poetry run python src/bot_v2/test_[slice].py

# Verify isolation
grep -r "from bot_v2.features" src/bot_v2/features/[slice]/
```

### Must Pass
- Slice imports work
- No cross-slice imports found
- Tests execute successfully

## Enforcement

### How to Check Violations
```bash
# Check for cross-slice imports
for slice in $(ls src/bot_v2/features/); do
  echo "Checking $slice..."
  grep -r "from bot_v2.features" src/bot_v2/features/$slice/ || echo "✅ Clean"
done
```

### Consequences of Violations
- System becomes untestable
- Token efficiency destroyed
- Maintenance becomes impossible
- Other slices break

## Remember

**These rules are NOT guidelines - they are ABSOLUTE.**

Breaking these rules breaks the entire architecture.

When in doubt:
1. Keep it in the slice
2. Duplicate if needed
3. Test in isolation
4. Update STATE.json
5. Check WHERE_TO_PUT.md for file placement