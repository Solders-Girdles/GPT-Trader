# Config Matrix Test Fixes

## Summary
Fixed import paths and environment variable handling in `tests/test_config_matrix.py` to ensure tests pass both locally and in CI.

## Issues Fixed

### 1. Import Path Issue
**Problem**: Tests were importing from `src.bot_v2.*` which doesn't exist in Poetry's package resolution.

**Fix**: Changed all imports and patch targets from:
```python
from src.bot_v2.orchestration.broker_factory import create_brokerage
@patch('src.bot_v2.orchestration.broker_factory.CoinbaseBrokerage')
```

To:
```python
from bot_v2.orchestration.broker_factory import create_brokerage
@patch('bot_v2.orchestration.broker_factory.CoinbaseBrokerage')
```

### 2. Environment Variable Interference
**Problem**: Existing environment variables (COINBASE_API_MODE, COINBASE_API_BASE, COINBASE_WS_URL) were interfering with test assertions about auto-detection.

**Fix**: Explicitly clear conflicting environment variables in all tests:
```python
env_vars = {
    # ... test variables ...
    "COINBASE_API_MODE": "",  # Clear to test auto-detection
    "COINBASE_API_BASE": "",  # Clear to test URL auto-selection
    "COINBASE_WS_URL": "",    # Clear to test WS URL auto-selection
}
```

## Test Coverage

The config matrix tests now properly validate:

1. **Production with specific keys**: Uses COINBASE_PROD_API_KEY/SECRET
2. **Sandbox with specific keys**: Uses COINBASE_SANDBOX_API_KEY/SECRET
3. **Production with fallback**: Falls back to COINBASE_API_KEY/SECRET
4. **Sandbox with fallback**: Falls back to COINBASE_API_KEY/SECRET

Each test correctly verifies:
- Appropriate key selection based on environment
- Correct base URL for sandbox vs production
- Sandbox mode auto-selects "exchange" API mode

## CI Integration

The test runs in CI with a matrix strategy covering all four scenarios:
- `prod`: Production with specific keys
- `sandbox`: Sandbox with specific keys
- `prod_fallback`: Production with fallback keys
- `sandbox_fallback`: Sandbox with fallback keys

## Verification

All tests now pass locally:
```bash
poetry run pytest -q tests/test_config_matrix.py
# Result: 4 passed in 0.45s
```

## Additional Improvements Suggested

1. **Add passphrase testing**: Verify COINBASE_API_PASSPHRASE selection for both prod/sandbox
2. **Add CDP key testing**: Test COINBASE_PROD_CDP_API_KEY/PRIVATE_KEY fallback behavior
3. **Move to main workflow**: Consider integrating into bot-v2.yml for single source of truth
4. **Add as required check**: Make this a required status check for PRs

## Status

✅ **COMPLETE** - All tests passing with proper import paths and environment isolation.