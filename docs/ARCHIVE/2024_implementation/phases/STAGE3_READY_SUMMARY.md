# Stage 3 Production Deployment - Ready Summary

## ✅ All Components Verified

### Authentication & Validation
- **Hardened Auth Test**: `scripts/test_stop_limit_hardened.py`
  - CDP JWT validation with PEM permission checks
  - **NEW**: Live auth probe via `list_accounts()` to verify connectivity
  - Fail-fast with clear error messages
  - Supports `--no-live-probe` to skip live test

- **Preflight Script**: `scripts/preflight/run_preflight.sh`
  - Comprehensive environment validation
  - Generates JSON report with `jwt_auth` and `derivatives_enabled` status
  - Creates all required directories
  - Validates component imports

### Core Components
- **PortfolioValuationService**: Reconciliation verified ✅
- **FeesEngine**: Tier calculations working ✅
- **MarginStateMonitor**: Window transitions detected ✅
- **LiquidityService**: SIZED_DOWN events captured ✅
- **OrderPolicyMatrix**: GTD properly gated ✅

### Verification Artifacts
- `/verification_reports/financial_reconciliation.json` ✅
- `/verification_reports/sized_down_event.json` ✅
- `/docs/ops/preflight/tif_validation.json` ✅

## 🚀 Quick Start Commands

### 1. Environment Setup
```bash
export COINBASE_SANDBOX=1
export COINBASE_API_MODE=advanced
export MAX_IMPACT_BPS=50
export COINBASE_CDP_API_KEY='your-cdp-api-key'
export COINBASE_CDP_PRIVATE_KEY_PATH='/path/to/private-key.pem'
export NO_PROXY='.coinbase.com,*.coinbase.com'  # if using proxy

# Fix PEM permissions
chmod 400 "$COINBASE_CDP_PRIVATE_KEY_PATH"
```

### 2. Run Preflight Checks
```bash
# Comprehensive preflight validation
bash scripts/preflight/run_preflight.sh

# Check the report
cat docs/ops/preflight/preflight_report_*.json | jq .
```

### 3. Validate Authentication (with Live Probe)
```bash
# Test CDP JWT auth with live API call
python scripts/test_stop_limit_hardened.py --symbol BTC-USD

# Skip live probe if needed
python scripts/test_stop_limit_hardened.py --symbol BTC-USD --no-live-probe
```

### 4. Launch Stage 3
```bash
# Full 24-hour run with monitoring
python scripts/stage3_runner.py

# Monitor in separate terminal
tail -f logs/stage3_run.log
```

## 📊 What Stage 3 Will Do

1. **Multi-Asset Trading**: BTC, ETH, SOL, XRP with conservative sizing
2. **Stop-Limit Testing**: Micro orders with 2% stop distance
3. **Continuous Monitoring**: Portfolio reconciliation every minute
4. **Artifact Collection**: Every 5 minutes to `/artifacts/stage3/`
5. **Impact Control**: 50 bps cap across all components

## 🔍 Live Auth Probe Details

The enhanced `test_stop_limit_hardened.py` now includes:
- **Live connectivity test** using `client.get_accounts()`
- **Specific error handling** for 401/403/connection errors
- **Helpful diagnostics** based on error type
- **Success confirmation** showing account count

Example successful output:
```
🔌 TESTING LIVE AUTHENTICATION
============================================================
Testing authentication with list_accounts()...
✅ Authentication successful - found 3 accounts
   First account: USD
```

## 📈 Success Criteria

### Must Pass
- [x] Auth validation with live probe
- [x] Financial reconciliation < $0.01 drift
- [x] GTD orders remain gated
- [ ] 24h run without margin calls
- [ ] Acceptance rate > 95%

### Should Capture
- [ ] At least 1 SIZED_DOWN event
- [ ] Margin window transitions
- [ ] Stop-limit order acceptance
- [ ] Multi-asset position tracking

## 🛡️ Risk Controls

- **Conservative Sizing**: Max $500 per symbol
- **Impact Cap**: 50 bps enforced
- **Reduce-Only Mode**: Available on demand
- **Emergency Close**: `python scripts/emergency_close_all.py`

## 📁 Output Locations

- **Artifacts**: `/artifacts/stage3/`
- **Logs**: `/logs/stage3_run.log`
- **Reports**: `/docs/ops/preflight/`
- **Verification**: `/verification_reports/`

## ✅ Final Checklist

- [x] CDP JWT auth configured and tested
- [x] Live connectivity verified
- [x] All components validated
- [x] Verification artifacts present
- [x] TIF support confirmed (GTD gated)
- [x] Impact cap set to 50 bps
- [x] Preflight report shows `jwt_auth=true`
- [x] Ready for 24h Stage 3 run

---

**Status**: READY FOR STAGE 3 DEPLOYMENT
**Confidence**: HIGH (all validations passed)
**Next Step**: `python scripts/stage3_runner.py`