# Stage 3 Quick Reference Card

## 🔧 Environment Setup (Copy & Paste)

```bash
# Core Settings
export COINBASE_SANDBOX=1
export COINBASE_API_MODE=advanced
export MAX_IMPACT_BPS=50

# CDP JWT Authentication
export COINBASE_CDP_API_KEY='organizations/YOUR_ORG_ID/apiKeys/YOUR_KEY_ID'
export COINBASE_CDP_PRIVATE_KEY_PATH="$HOME/.coinbase/cdp-private-key.pem"

# For Corporate Networks (if using proxy)
export NO_PROXY='api.sandbox.coinbase.com,advanced-trade-ws.sandbox.coinbase.com,*.coinbase.com'

# Fix PEM Permissions
chmod 400 "$COINBASE_CDP_PRIVATE_KEY_PATH"
```

## 🚀 Launch Sequence

```bash
# 1. Preflight Check (2 seconds)
bash scripts/preflight/run_preflight.sh

# 2. TIF Validation (1 second)
python scripts/validation/validate_tif_simple.py

# 3. Auth Test with Live Probe (5 seconds)
python scripts/test_stop_limit_hardened.py --symbol BTC-USD

# 4. Launch Stage 3 (24 hours)
python scripts/stage3_runner.py
```

## 📍 Correct Endpoints

### Sandbox (COINBASE_SANDBOX=1)
- REST: `https://api.sandbox.coinbase.com`
- WS: `wss://advanced-trade-ws.sandbox.coinbase.com`

### Production (COINBASE_SANDBOX=0)
- REST: `https://api.coinbase.com`
- WS: `wss://advanced-trade-ws.coinbase.com`

## 🔍 Quick Diagnostics

### Check Auth is Working
```bash
# Should show accounts and confirm endpoints
python scripts/test_stop_limit_hardened.py --symbol BTC-USD
```

Expected output:
```
🔌 TESTING LIVE AUTHENTICATION
Endpoints:
  REST: https://api.sandbox.coinbase.com
  WS: wss://advanced-trade-ws.sandbox.coinbase.com
  Mode: SANDBOX

Testing authentication with list_accounts()...
✅ Authentication successful - found 3 accounts
   First account: USD
```

### Check Preflight Report
```bash
# Should show jwt_auth=true, derivatives_enabled=true
cat docs/ops/preflight/preflight_report_*.json | jq '.environment'
```

### Monitor Stage 3
```bash
# In separate terminal
tail -f logs/stage3_run.log

# Check artifacts
ls -la artifacts/stage3/
```

## 🛑 Emergency Controls

```bash
# Set reduce-only mode
curl -X POST localhost:8080/api/reduce-only

# Close all positions
python scripts/emergency_close_all.py

# Stop Stage 3
# Press Ctrl+C in the terminal running stage3_runner.py
```

## 📊 Stage 3 Settings

| Parameter | Value |
|-----------|-------|
| Symbols | BTC-USD, ETH-USD, SOL-USD, XRP-USD |
| Max Impact | 50 bps |
| Position Limits | BTC: $500, ETH: $300, SOL/XRP: $100 |
| Stop Distance | 2% |
| Duration | 24 hours |
| Monitoring | Every 1 minute |
| Artifacts | Every 5 minutes |

## ✅ Success Indicators

- **Auth Test**: "Authentication successful - found X accounts"
- **Preflight**: "ALL PREFLIGHT CHECKS PASSED"
- **TIF**: "GTD Fully Gated: ✅"
- **Stage 3**: Artifacts appearing in `/artifacts/stage3/`

## ❌ Common Issues & Fixes

### Auth Fails with 401
```bash
# Check CDP key format
echo $COINBASE_CDP_API_KEY
# Should be: organizations/xxx/apiKeys/xxx

# Verify PEM is valid
head -1 "$COINBASE_CDP_PRIVATE_KEY_PATH"
# Should show: -----BEGIN EC PRIVATE KEY-----
```

### Connection Timeout
```bash
# Add Coinbase to NO_PROXY
export NO_PROXY='api.sandbox.coinbase.com,advanced-trade-ws.sandbox.coinbase.com,*.coinbase.com'

# Test connectivity
curl -I https://api.sandbox.coinbase.com
```

### Permission Denied on PEM
```bash
chmod 400 "$COINBASE_CDP_PRIVATE_KEY_PATH"
ls -la "$COINBASE_CDP_PRIVATE_KEY_PATH"
# Should show: -r--------
```

---

**Remember**: Always run in SANDBOX mode first!