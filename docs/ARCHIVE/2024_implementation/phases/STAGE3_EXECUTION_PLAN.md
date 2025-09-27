# Stage 3 Execution Plan - Multi-Asset Micro Test

## Overview

Stage 3 implements controlled multi-asset trading with conservative sizing, stop-limit order testing, and comprehensive monitoring over 24 hours.

## Configuration

### Symbols
- **BTC-USD**: Primary test symbol (most liquid)
- **ETH-USD**: Secondary large cap
- **SOL-USD**: Mid-cap test
- **XRP-USD**: Small-cap test

### Position Limits (Conservative)
```json
{
  "BTC-USD": {"max_size": 0.01, "max_notional": 500},
  "ETH-USD": {"max_size": 0.1, "max_notional": 300},
  "SOL-USD": {"max_size": 10, "max_notional": 100},
  "XRP-USD": {"max_size": 200, "max_notional": 100}
}
```

### Risk Parameters
- **Max Impact**: 50 bps (aligned across all components)
- **Stop Distance**: 2% from entry
- **Sizing Mode**: Conservative
- **Leverage Limit**: 1x initially

### Order Types
- **Supported**: LIMIT (GTC, IOC), STOP_LIMIT (GTC)
- **Gated**: LIMIT (GTD) - remains gated
- **Testing**: Micro stop-limit orders

## Pre-Flight Checklist

### 1. TIF Validation ✅
```bash
python scripts/validation/validate_tif_simple.py
```
**Result**: GTD properly gated, GTC/IOC supported

### 2. Component Verification ✅
- Portfolio valuation reconciliation: PASSED
- SIZED_DOWN event generation: PASSED
- Fee tier awareness: VERIFIED
- Margin window detection: WORKING

### 3. Environment Setup
```bash
# Create artifact directories
mkdir -p artifacts/stage3
mkdir -p logs
mkdir -p docs/ops/preflight

# Verify sandbox credentials
export COINBASE_SANDBOX=1
export MAX_IMPACT_BPS=50
```

## Execution Commands

### Option 1: Full Stage 3 Runner (Recommended)
```bash
python scripts/stage3_runner.py
```

This will:
- Run preflight checks
- Initialize orchestrator with Stage 3 settings
- Test stop-limit orders on each symbol
- Monitor portfolio for 24 hours
- Collect all artifacts

### Option 2: Manual Execution
```bash
# 1. Run preflight
bash scripts/run_preflight.sh

# 2. Start orchestrator with Stage 3 config
python scripts/run_perps_bot_v2_week3.py \
  --profile demo \
  --symbols BTC-USD,ETH-USD,SOL-USD,XRP-USD \
  --sizing-mode conservative \
  --max-impact-bps 50 \
  --order-type stop_limit \
  --stop-pct 2 \
  --run-24h

# 3. Monitor in separate terminal
python scripts/dashboard_lite.py
```

## Monitoring During Run

### Real-Time Dashboards
- Portfolio equity and PnL
- Per-symbol positions and metrics
- Margin utilization and windows
- Liquidity scores and spreads

### Key Metrics to Track
1. **Acceptance Rate**: Target >95% (excluding enforcement rejections)
2. **SIZED_DOWN Events**: Capture all with original/adjusted sizes
3. **Margin Utilization**: Should stay <10% with conservative sizing
4. **Reconciliation**: Equity = Realized + Unrealized - Fees - Funding

### Alert Thresholds
- Margin utilization >20%: WARNING
- Margin utilization >50%: CRITICAL
- Mark staleness >30s: WARNING
- Reconciliation drift >$1: ERROR

## Artifacts to Collect

### Required Artifacts
1. **reconciliation_history.json** - Multi-asset PnL breakdown
2. **sized_down_events.json** - Liquidity-based reductions
3. **margin_snapshots.json** - Window transitions and utilization
4. **execution_logs.json** - Order lifecycle events
5. **rejection_breakdown.json** - Rejection reasons and counts

### Storage Location
All artifacts stored in: `/artifacts/stage3/`

### Collection Schedule
- Portfolio snapshots: Every 1 minute
- Artifact saves: Every 5 minutes
- Final summary: At completion

## Stop-Limit Test Protocol

### Test Configuration
```python
stop_tests = {
    'BTC-USD': {'size': 0.001, 'stop_offset': 2%},
    'ETH-USD': {'size': 0.01, 'stop_offset': 2%},
    'SOL-USD': {'size': 1, 'stop_offset': 2%},
    'XRP-USD': {'size': 20, 'stop_offset': 2%}
}
```

### Expected Behavior
1. Place non-crossing stop-limit orders
2. Monitor for trigger conditions
3. Capture execution if triggered
4. Log order IDs and statuses

## Success Criteria

### Must Pass
- [x] Financial reconciliation within $0.01
- [x] GTD orders remain gated
- [ ] No margin calls or liquidations
- [ ] Acceptance rate >95%
- [ ] All artifacts generated

### Should Pass
- [ ] At least 1 SIZED_DOWN event captured
- [ ] Stop-limit orders placed successfully
- [ ] Margin window transitions detected
- [ ] Fee tier correctly applied

### Nice to Have
- [ ] Cross-asset correlation metrics
- [ ] Funding rate accruals
- [ ] WebSocket reconnection handling

## Post-Run Analysis

### Generate Summary Report
```bash
python scripts/analyze_stage3_artifacts.py
```

### Key Questions to Answer
1. Did multi-asset reconciliation balance?
2. Were SIZED_DOWN events triggered appropriately?
3. Did margin windows transition correctly?
4. Were stop-limit orders accepted?
5. What was the effective acceptance rate?

## Rollback Plan

### If Issues Arise
1. **Immediate**: Set reduce-only mode
2. **5 minutes**: Close all positions at market
3. **10 minutes**: Disable trading for all symbols
4. **Document**: Save all logs and artifacts

### Emergency Commands
```bash
# Set reduce-only mode
curl -X POST localhost:8080/api/reduce-only

# Close all positions
python scripts/emergency_close_all.py

# Disable trading
curl -X POST localhost:8080/api/disable-trading
```

## Next Steps After Success

### Stage 4: Size Increase
- Increase to 0.1 BTC equivalent per symbol
- Enable bracket orders
- Test reduce-only mode transitions
- Add more aggressive strategies

### Stage 5: Full Production
- Target position sizes
- All order types (except GTD)
- Automated strategy selection
- 24/7 operation

## Timeline

- **T+0h**: Start Stage 3 runner
- **T+1h**: Verify initial trades
- **T+4h**: Check margin window transition (if applicable)
- **T+8h**: Mid-run checkpoint
- **T+24h**: Complete run and analyze
- **T+25h**: Generate reports and decision

---

**Status**: READY TO EXECUTE
**Command**: `python scripts/stage3_runner.py`
**Duration**: 24 hours
**Risk Level**: LOW (conservative sizing)