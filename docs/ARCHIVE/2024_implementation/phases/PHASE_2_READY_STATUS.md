# Phase 2 Exchange Sandbox Testing - Ready Status

## Current Status ✅

### Environment Check
- **Current Mode**: CDP/Advanced Trade (needs to switch to Exchange Sandbox)
- **Current Egress IPs**:
  - IPv4: `72.208.131.101`
  - IPv6: `2600:8800:2800:cd00:ac87:63ad:30c9:d62a`

### WebSocket Connectivity ✅
- Exchange Sandbox Feed: **WORKING**
  - URL: `wss://ws-feed-public.sandbox.exchange.coinbase.com`
  - Latency: 371ms
  - Public ticker data flowing

### Test Scripts Ready ✅
1. **Permission Probe**: `scripts/exchange_sandbox_order_test.py`
2. **Simple Test**: `scripts/test_exchange_sandbox_simple.py`
3. **WebSocket Probe**: `scripts/ws_probe.py`

## Required Actions Before Phase 2 Execution

### 1. Set Exchange Sandbox Credentials
```bash
# Clear CDP settings
unset COINBASE_CDP_API_KEY COINBASE_CDP_PRIVATE_KEY COINBASE_AUTH_TYPE

# Set Exchange Sandbox credentials
export COINBASE_API_KEY="your-exchange-sandbox-key"
export COINBASE_API_SECRET="your-exchange-sandbox-secret"
export COINBASE_API_PASSPHRASE="your-exchange-sandbox-passphrase"
export COINBASE_SANDBOX=1
export COINBASE_API_MODE=exchange
```

### 2. Verify IP Whitelisting
Ensure both IPs are whitelisted in your Exchange Sandbox API key settings:
- IPv4: `72.208.131.101`
- IPv6: `2600:8800:2800:cd00:ac87:63ad:30c9:d62a`

### 3. Verify Setup
```bash
# Check environment
echo "API_MODE: $COINBASE_API_MODE"  # Should show: exchange
echo "SANDBOX: $COINBASE_SANDBOX"     # Should show: 1

# Test connectivity
poetry run python scripts/ws_probe.py --sandbox
```

## Phase 2 Test Commands

### Step 1: Permission Probe
```bash
poetry run python scripts/exchange_sandbox_order_test.py --quick-check
```

Expected results:
- VIEW permission: ✅
- TRADE permission: ✅
- Order lifecycle: OPEN → CANCELLED
- Latencies: <500ms for place/cancel

### Step 2: Full Order Test
```bash
poetry run python scripts/exchange_sandbox_order_test.py
```

### Step 3: Alternative CLI
```bash
poetry run python scripts/test_exchange_sandbox_simple.py
```

## Code Fixes Completed ✅

1. **Exception Handling Fixed**:
   - `place_order` now properly re-raises ValidationError, TypeError, etc.
   - Only returns None for ExecutionError/NetworkError

2. **TimeInForce Enum Handling**:
   - Fixed in `execution.py` and `adapters.py`
   - Handles both string and enum types

3. **Risk Manager Integration**:
   - Removed invalid `validate_order` call
   - Validation happens in execution engine

4. **Test Improvements**:
   - Fixed module import paths
   - All 11 live_trade_error_handling tests passing
   - All 10 type_consolidation tests passing

## Ready for Phase 2 ✅

The codebase is fully prepared for Phase 2 Exchange Sandbox testing. Once credentials are configured and IPs are whitelisted, the tests can be executed immediately.

### Security Reminders
- Use sandbox-only credentials
- Keep credentials in local `.env` file
- Never commit credentials
- Rotate keys after testing