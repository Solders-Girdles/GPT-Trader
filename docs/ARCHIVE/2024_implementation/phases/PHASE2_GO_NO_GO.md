# Phase 2: Go/No-Go Decision Matrix

## Current Status: READY (Pending Credentials)

### Mock Verification ✅ COMPLETE

| Feature | Expected | Actual | Status |
|---------|----------|--------|--------|
| Post-only non-crossing | Accept order | Accepted | ✅ |
| Post-only crossing | Reject order | Rejected | ✅ |
| SIZED_DOWN logging | Show adjustment | Logged correctly | ✅ |
| Stop order triggers | Track locally | Tracked | ✅ |
| Impact sizing | Reduce notional | Reduced to 59% | ✅ |
| PnL calculation | Accurate | Verified | ✅ |

### Real Broker Requirements

| Requirement | Status | Action |
|-------------|--------|--------|
| CDP API Key | ⏳ Pending | Set COINBASE_CDP_API_KEY |
| CDP Private Key | ⏳ Pending | Set COINBASE_CDP_PRIVATE_KEY |
| Clock sync (<2s offset) | ✅ Ready | Verified |
| DNS/TLS connectivity | ✅ Ready | Tested |
| EventStore directory | ✅ Ready | Created |
| Safety config | ✅ Ready | Configured |

## Execution Sequence

### Step 1: Environment Setup
```bash
export COINBASE_API_MODE=advanced
export COINBASE_AUTH_TYPE=JWT
export COINBASE_ENABLE_DERIVATIVES=1
export COINBASE_CDP_API_KEY='your_key'
export COINBASE_CDP_PRIVATE_KEY='your_private_key'
export EVENT_STORE_ROOT=/tmp/phase2_eventstore
```

### Step 2: Pre-Flight Verification
```bash
bash scripts/phase2_real_broker_checklist.sh
```

### Step 3: Capability Probe
```bash
python scripts/probe_capabilities.py --live --symbol BTC-PERP
```

### Step 4: Dry-Run Pulse (1 cycle)
```bash
python scripts/phase2_demo_runner.py --dry-run
```
**Verify**: WS snapshots, filters, staleness blocking

### Step 5: Live Demo Trading
```bash
# Terminal 1: Run demo
python scripts/phase2_demo_runner.py

# Terminal 2: Monitor metrics
python scripts/phase2_metrics_monitor.py
```

## Key Metrics to Monitor

### Execution Metrics (5-10 min window)
- `placed`: > 0 (orders attempted)
- `cancelled`: > 0 (auto-cancel working)
- `rejected`: < 20% (acceptable rejection rate)
- `post_only_rejected`: > 0 (crossing protection working)
- `stop_triggered`: Increments if stops hit

### Strategy Metrics
- Acceptance rate: > 20% good
- Rejection breakdown:
  - `spread_too_wide`: < 50%
  - `insufficient_depth`: < 30%
  - `stale_data`: < 10%

### PnL & Funding
- Realized PnL: Updates on exits
- Unrealized PnL: Updates with mark price
- Funding: Accrues only at intervals (8h typically)

### Latency & Health
- `place_order_latency`: < 500ms
- `ws_staleness`: < 10s
- `retry_count`: < 3 per order
- No rate limit errors

## Safety Guardrails Active

| Guardrail | Setting | Purpose |
|-----------|---------|---------|
| Sizing mode | CONSERVATIVE | Reduce impact |
| Target notional | $25-100 | Tiny positions |
| Max impact | 10 bps | Limit slippage |
| Daily loss limit | $100 | Auto-shutdown |
| Pre-funding quiet | 30 min | Avoid funding spikes |
| Auto-cancel | 30s | Prevent stale orders |
| Post-only | Enabled | Maker only |
| Kill switch | Ctrl+C | Emergency stop |

## Success Criteria

Phase 2 succeeds if ALL of the following occur:

- [ ] At least 1 post-only limit placed (non-crossing)
- [ ] At least 1 crossing attempt rejected
- [ ] SIZED_DOWN log appears for large notional
- [ ] Order auto-cancels after timeout
- [ ] PnL updates correctly on position changes
- [ ] No unexpected errors or retries
- [ ] Kill switch test switches to reduce-only
- [ ] Metrics readable in EventStore

## Known Issues & Mitigations

| Issue | Mitigation |
|-------|------------|
| Post-only fills immediately | Increase offset to 15-20 bps |
| Stop orders unsupported | Engine gates; use local triggers |
| WS staleness spikes | Increase threshold to 15s |
| Min size rejection | Adjust to exchange minimums |
| Funding time null | Skip quiet period with log |

## Go/No-Go Decision

### GO Criteria ✅
- All mock tests passed
- Credentials configured
- Clock synchronized
- Network connectivity verified
- Safety limits configured
- Kill switch tested

### NO-GO Criteria ❌
- Missing credentials
- Clock offset > 5s
- Network issues
- Rate limit errors
- Unexpected exceptions

## Phase 3 Promotion Criteria

Promote to Phase 3 (Production Canary) when:

1. **Execution**: ≥1 clean entry/exit with no errors
2. **Rejection**: Post-only rejections working
3. **Sizing**: SIZED_DOWN events logged
4. **PnL**: Accurate tracking verified
5. **Safety**: Kill switch tested successfully
6. **Metrics**: Clean logs, no excessive errors
7. **Stability**: 10+ minutes without issues

---

## Current Decision: **READY** ✅

**Status**: All systems verified in mock. Awaiting CDP credentials for real broker execution.

**Recommendation**: Proceed with Phase 2 immediately upon credential configuration.