# How to Run Tests

## Prerequisites
Make sure you have:
1. Virtual environment activated: `source .venv/bin/activate`
2. Dependencies installed: `cd config && poetry install && cd ..`

## Test Commands

### Test All Slices
```bash
poetry run python src/bot_v2/test_all_slices.py
```

### Test Specific Slice
```bash
poetry run python src/bot_v2/test_[slice_name].py

# Examples:
poetry run python src/bot_v2/test_backtest.py
poetry run python src/bot_v2/test_ml_strategy.py
```

### Quick Import Test
```bash
python -c "from src.bot_v2.features.backtest import *; print('✅')"
```

### Verify Slice Isolation
```bash
# Should return nothing if properly isolated
grep -r "from bot_v2.features" src/bot_v2/features/[slice_name]/
```

### Run All Tests with pytest
```bash
poetry run pytest src/bot_v2/ -v
```

## Testing Workflow

1. **Before Changes**
   ```bash
   poetry run python src/bot_v2/test_[slice].py
   ```

2. **After Changes**
   ```bash
   # Test the modified slice
   poetry run python src/bot_v2/test_[slice].py
   
   # Test all slices for side effects
   poetry run python src/bot_v2/test_all_slices.py
   ```

3. **Verify Isolation**
   ```bash
   grep -r "import" src/bot_v2/features/[slice]/ | grep features
   ```

## Expected Results

✅ **Good Test Output:**
```
Testing backtest slice...
✅ backtest imports successfully
✅ backtest isolation verified
✅ backtest tests pass
```

❌ **Bad Test Output:**
```
Testing backtest slice...
❌ backtest has isolation violations
```

## Troubleshooting

- **Import errors**: Check slice has all needed code locally
- **Isolation violations**: Remove cross-slice imports
- **Test failures**: Fix within slice boundaries only