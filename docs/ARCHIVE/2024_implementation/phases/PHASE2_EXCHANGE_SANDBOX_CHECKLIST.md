# Phase 2: Exchange Sandbox Order Lifecycle Validation

## Pre-Flight Checklist

### 🔐 Credentials Setup
- [ ] Create Exchange Sandbox API key at https://public.sandbox.exchange.coinbase.com/
- [ ] Enable **View** permission
- [ ] Enable **Trade** permission
- [ ] Add IPv4 whitelist: `72.208.131.101`
- [ ] Add IPv6 whitelist: `2600:8800:2800:cd00:ac87:63ad:30c9:d62a`

### 📝 Environment Configuration (.env)
```bash
COINBASE_API_KEY=your-exchange-sandbox-key
COINBASE_API_SECRET=your-base64-secret
COINBASE_API_PASSPHRASE=your-passphrase
COINBASE_SANDBOX=1
COINBASE_API_MODE=exchange
```

## Validation Steps

### Step 1: Quick Permission Probe
```bash
poetry run python scripts/exchange_sandbox_order_test.py --quick-check
```

**Expected Output:**
- ✅ "VIEW permission confirmed"
- ✅ "TRADE permission confirmed"
- ✅ Order placed and cancelled successfully

**Troubleshooting:**
| Error | Cause | Fix |
|-------|-------|-----|
| 401 Unauthorized | Wrong credentials | Verify key/secret/passphrase |
| 403 Forbidden | Missing trade permission | Enable "Trade" on API key |
| Connection refused | IP not whitelisted | Add IPs to whitelist |
| No accounts | Wrong mode | Set COINBASE_API_MODE=exchange |

### Step 2: Full Order Lifecycle Test
```bash
poetry run python scripts/exchange_sandbox_order_test.py
```

**Success Criteria:**
- [x] Order placed with status OPEN
- [x] Order cancelled within 5 seconds
- [x] Zero fills (order at $10 shouldn't execute)
- [x] Place latency < 500ms
- [x] Cancel latency < 500ms

### Step 3: Alternative CLI Test
```bash
poetry run python scripts/test_exchange_sandbox_simple.py
```

Uses the simple_cli broker infrastructure with same success criteria.

## Test Results Template

```yaml
test_date: YYYY-MM-DD HH:MM:SS
environment: exchange_sandbox
test_type: order_lifecycle

credentials:
  api_mode: exchange
  sandbox: true
  permissions: [view, trade]

results:
  permission_probe: PASS/FAIL
  order_placement: PASS/FAIL
  order_cancellation: PASS/FAIL
  
metrics:
  place_latency_ms: XXX
  cancel_latency_ms: XXX
  total_duration_s: X.X
  fills_count: 0
  
order_details:
  order_id: xxx-xxx-xxx
  symbol: BTC-USD
  side: buy
  type: limit
  price: 10.00
  size: 0.0001
  status_transitions: [placed, open, cancelled]
```

## Common Issues & Solutions

### 1. Environment Variables Not Loading
```bash
# Verify variables are set
echo $COINBASE_API_MODE
echo $COINBASE_SANDBOX

# Source .env if needed
source .env
# or
export $(cat .env | xargs)
```

### 2. Mode Mismatch
```bash
# Ensure exchange mode
export COINBASE_API_MODE=exchange
export COINBASE_SANDBOX=1
```

### 3. Permission Delays
- Wait 60 seconds after changing API permissions
- Some UI changes have propagation delay

### 4. Whitelist Issues
- Verify your current IP: `curl ifconfig.me`
- Add both IPv4 and IPv6 if dual-stack
- Whitelist changes take effect immediately

## Security Checklist

### During Testing
- [ ] Use only sandbox credentials
- [ ] Never log full API secrets
- [ ] Keep .env in .gitignore

### After Testing
- [ ] Rotate sandbox API key
- [ ] Clear shell history if credentials were typed
- [ ] Verify no credentials in logs/artifacts

## Phase 2 Sign-Off

### Validation Complete
- [ ] Permission probe passed
- [ ] Order lifecycle test passed
- [ ] Latency within targets
- [ ] No unexpected fills
- [ ] Results documented

### Ready for Next Phase
- [ ] Exchange sandbox validated
- [ ] Order flow confirmed working
- [ ] Can proceed to production canary with limits

## Quick Reference Commands

```bash
# Check current setup
grep COINBASE .env | grep -v SECRET

# Quick permission test
poetry run python scripts/exchange_sandbox_order_test.py --quick-check

# Full lifecycle test
poetry run python scripts/exchange_sandbox_order_test.py

# Alternative CLI test
poetry run python scripts/test_exchange_sandbox_simple.py

# Test with custom parameters
poetry run python scripts/exchange_sandbox_order_test.py \
  --symbol ETH-USD \
  --limit-price 5.0 \
  --qty 0.001
```

## Success Metrics

| Metric | Target | Actual |
|--------|--------|--------|
| Permission Validation | ✅ | [ ] |
| Order Placement | ✅ | [ ] |
| Order Cancellation | ✅ | [ ] |
| Place Latency | < 500ms | [ ] |
| Cancel Latency | < 500ms | [ ] |
| Fills at $10 | 0 | [ ] |
| Total Test Duration | < 10s | [ ] |

---

**Phase 2 Status:** ⏳ Awaiting Validation

Once all checkboxes are complete and tests pass, Phase 2 is validated and ready for production canary deployment with strict limits.