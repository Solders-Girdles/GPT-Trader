# Phase 2 Execution Checklist - Demo Trading

## Pre-Flight Requirements ✅

### Environment Setup
- [ ] `COINBASE_SANDBOX=1` - Sandbox mode enabled
- [ ] `COINBASE_API_MODE=advanced` - Advanced Trade API
- [ ] `COINBASE_AUTH_TYPE=JWT` - JWT authentication
- [ ] `COINBASE_ENABLE_DERIVATIVES=1` - Derivatives enabled
- [ ] CDP API key configured
- [ ] Private key secured (file, not env)

### Safety Configuration
- [ ] `CONSERVATIVE` sizing mode
- [ ] `max_impact_bps <= 15`
- [ ] `--rsi-confirm` flag enabled
- [ ] Daily loss cap: $100 (demo)
- [ ] Pre-funding quiet: 30 minutes
- [ ] Kill switch tested

### Validation Scripts
```bash
# Run all pre-flight checks
python scripts/preflight_check.py

# Validate API capabilities
python scripts/capability_probe.py

# Test with mock adapter first
COINBASE_SANDBOX=1 RUN_SANDBOX_VALIDATIONS=1 \
  python scripts/validate_perps_client_week1.py
```

## Phase 2 Execution Steps 🚀

### Step 1: Dry Run Pulse (5 minutes)
```bash
# Single cycle with real adapter
python -m src.bot_v2.features.brokerages.coinbase.demo \
  --mode dry-run \
  --symbol BTC-PERP \
  --cycles 1 \
  --validate-only

# Verify:
✅ WebSocket connected
✅ Filters/guards active
✅ No actual orders placed
```

### Step 2: Post-Only Limits (10 minutes)
```bash
# Place non-crossing post-only order
python scripts/place_test_order.py \
  --symbol BTC-PERP \
  --type limit \
  --side buy \
  --size 0.0001 \
  --price 40000 \
  --post-only \
  --auto-cancel 30

# Expected:
✅ Order accepted (non-crossing)
✅ Auto-cancelled after 30s
✅ POST_ONLY_WOULD_CROSS rejections for crossing prices
```

### Step 3: Market Entry/Exit (10 minutes)
```bash
# Tiny market order entry and reduce-only exit
python scripts/market_test.py \
  --symbol BTC-PERP \
  --entry-size 0.0001 \
  --reduce-only-exit

# Verify:
✅ Market entry filled
✅ Position tracked correctly
✅ Reduce-only exit executed
✅ PnL calculated and logged
```

### Step 4: Stop Orders (Optional, 5 minutes)
```bash
# Far stop-limit order (won't trigger)
python scripts/stop_test.py \
  --symbol BTC-PERP \
  --type stop-limit \
  --stop-price 35000 \
  --limit-price 34900 \
  --size 0.0001 \
  --cancel-after 60

# Verify:
✅ Stop order created
✅ Successfully cancelled
```

## Monitoring During Demo (30 minutes) 📊

### Launch Dashboard
```bash
# Terminal 1: Run dashboard
python scripts/dashboard_lite.py --refresh 5

# Terminal 2: Run validator
python scripts/demo_run_validator.py --duration 300
```

### Key Metrics to Watch

#### Engine Metrics
- [ ] Orders placed/cancelled/rejected
- [ ] Post-only rejections observed
- [ ] Stop orders triggered (if any)
- [ ] Latency < 200ms p50
- [ ] Cancel/replace operations

#### Strategy Metrics
- [ ] Acceptance rate > 90%
- [ ] Rejection reasons logged
- [ ] SIZED_DOWN events for oversized orders
- [ ] STRICT mode rejections clear
- [ ] Spread/depth/volatility filters working

#### PnL & Funding
- [ ] PositionState updates correct
- [ ] Funding accrual on interval boundaries
- [ ] Funding sign correct (long pays in contango)
- [ ] Daily PnL metrics generated

#### Storage & Observability
- [ ] Logs in correct location (`EVENT_STORE_ROOT`)
- [ ] Real-time metrics visible
- [ ] No noisy error logs
- [ ] Health checks passing

## Acceptance Criteria ✅

### Must Pass (All Required)
- [ ] At least one entry/exit completed cleanly
- [ ] No unexpected retries or rate-limit faults
- [ ] Post-only crossing rejections observed
- [ ] PnLTracker coherent and accurate
- [ ] Health/metrics clean and readable

### Should Pass (3 of 5)
- [ ] Non-crossing post-only accepted
- [ ] SIZED_DOWN events for oversized notional
- [ ] Funding accrual only on new intervals
- [ ] WebSocket stable (< 3 reconnects)
- [ ] Latency consistently < 300ms p95

### Nice to Have
- [ ] Stop order create/cancel successful
- [ ] Multiple symbols tested (ETH-PERP)
- [ ] Slippage tracking accurate
- [ ] Volume-weighted metrics

## Post-Demo Analysis 📈

### Generate Reports
```bash
# Export metrics
python scripts/dashboard_lite.py --export demo_metrics.json

# Generate validation report
python scripts/demo_run_validator.py --duration 0 --export

# Analyze rejection reasons
grep "rejected" /tmp/trading_orders.log | \
  jq -r .reason | sort | uniq -c > rejections.txt
```

### Review Checklist
- [ ] Order acceptance rate
- [ ] Rejection reason breakdown
- [ ] Latency distribution
- [ ] PnL accuracy
- [ ] Funding calculations
- [ ] Error log analysis

## Troubleshooting Common Issues 🔧

### WebSocket Disconnects
```bash
# Check JWT expiry
python scripts/check_jwt_expiry.py

# Test WebSocket stability
python scripts/test_websocket_stability.py --duration 300
```

### High Rejection Rate
```bash
# Analyze rejections
python scripts/analyze_rejections.py --last-hour

# Check quantization
python scripts/test_quantization.py --symbol BTC-PERP
```

### PnL Discrepancies
```bash
# Reconcile positions
python scripts/reconcile_positions.py

# Verify funding calculations
python scripts/verify_funding.py --symbol BTC-PERP
```

## Promotion to Canary 🎯

### Go/No-Go Decision
```yaml
Criteria:
  Technical:
    - All must-pass criteria met: YES/NO
    - At least 3 should-pass met: YES/NO
    - No critical issues: YES/NO
  
  Business:
    - Risk team approval: YES/NO
    - Compliance check: YES/NO
    - Management sign-off: YES/NO

Decision: PROCEED/HOLD/ABORT
```

### Canary Configuration
```bash
# Update for canary (real funds, tiny size)
export COINBASE_SANDBOX=0
export COINBASE_MAX_POSITION_SIZE=0.001
export COINBASE_DAILY_LOSS_LIMIT=100
export COINBASE_SYMBOLS="BTC-PERP"
```

## Documentation & Handoff 📄

### Required Artifacts
- [ ] Demo validation report
- [ ] Metrics dashboard snapshot
- [ ] Rejection analysis
- [ ] Latency distribution
- [ ] PnL reconciliation
- [ ] Issue log with resolutions

### Handoff Package
```bash
# Create handoff archive
tar -czf demo_handoff_$(date +%Y%m%d).tar.gz \
  demo_validation_*.json \
  demo_metrics.json \
  rejections.txt \
  /tmp/trading_orders.log \
  preflight_output.txt
```

## Sign-Off ✍️

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Engineering | | | |
| Risk | | | |
| Operations | | | |
| Management | | | |

---

*Checklist Version: 1.0*
*Phase: Demo Trading*
*Next Phase: Production Canary*
*Last Updated: 2025-08-30*