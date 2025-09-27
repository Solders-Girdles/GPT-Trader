# Production Rollout Plan - Week 3 Advanced Orders

## Current Status ✅
- **AdvancedExecutionEngine**: Integrated with full parameter mapping
- **Order Types**: Market, Limit (w/ post-only), Stop, Stop-Limit
- **Impact Sizing**: Conservative/Strict/Aggressive modes implemented
- **PnL/Funding**: Full tracking with daily metrics
- **Validation**: All mock tests passing

## Pre-Production Checklist

### Technical Readiness
- [ ] Stop order support verification in real Coinbase adapter
- [ ] Engine→broker shim tests aligned with actual parameter names
- [ ] Sized-down logging includes original vs adjusted notional
- [ ] Capability probe function for feature detection
- [ ] Metrics appended to EventStore with bot_id
- [ ] Health file includes order metrics summary

### Operational Readiness
- [ ] Kill switch tested (reduce-only mode toggle)
- [ ] Runbooks updated with new CLI flags
- [ ] Dashboard or log aggregation for metrics
- [ ] Alert thresholds configured
- [ ] Rollback procedure documented

## Rollout Phases

### Phase 1: Sandbox Rehearsal (1-2 days)
**Objective**: Validate all order types and flows in sandbox environment

**Configuration**:
```bash
COINBASE_SANDBOX=1 python scripts/run_perps_bot_v2_week3.py \
  --profile dev \
  --symbols BTC-PERP ETH-PERP \
  --dry-run \
  --max-spread-bps 10 \
  --min-depth-l1 50000 \
  --rsi-confirm
```

**Tests**:
1. Market orders with strict filters
2. Limit post-only (non-crossing validation)
3. Stop/stop-limit placement and cancellation
4. Cancel/replace flow
5. Impact-aware sizing adjustments
6. Reduce-only exits

**Success Criteria**:
- [ ] All order types place successfully
- [ ] Post-only rejects crossing orders
- [ ] Cancel/replace maintains idempotency
- [ ] Metrics properly logged to EventStore
- [ ] No unexpected errors in 24h run

### Phase 2: Demo Profile (1-2 days)
**Objective**: Test with tiny real positions

**Configuration**:
```bash
python scripts/run_perps_bot_v2_week3.py \
  --profile demo \
  --symbols BTC-PERP \
  --order-type limit \
  --post-only \
  --sizing-mode conservative \
  --max-impact-bps 10 \
  --max-spread-bps 5 \
  --min-depth-l10 200000 \
  --rsi-confirm
```

**Tests**:
1. Real entries with $100-500 notional
2. Post-only maker orders
3. Stop-loss placement (far from market)
4. Position tracking and PnL updates
5. Funding accrual timing

**Success Criteria**:
- [ ] Positions open/close correctly
- [ ] PnL tracking matches exchange
- [ ] Funding accrues at correct intervals
- [ ] No slippage beyond configured limits
- [ ] Metrics dashboard functional

### Phase 3: Production Canary (2-3 days)
**Objective**: Limited production deployment

**Configuration**:
```bash
python scripts/run_perps_bot_v2_week3.py \
  --profile prod \
  --symbols BTC-PERP \
  --order-type market \
  --sizing-mode conservative \
  --max-impact-bps 5 \
  --max-spread-bps 3 \
  --min-depth-l1 100000 \
  --min-depth-l10 500000 \
  --min-vol-1m 100000 \
  --rsi-confirm \
  --liq-buffer-pct 25 \
  --max-slippage-bps 10
```

**Monitoring**:
- Order acceptance rate > 80%
- Post-only rejection rate < 10%
- Slippage < max_slippage_bps
- Daily PnL tracking
- Max drawdown < 2%

**Expansion Criteria**:
- [ ] 24h stable operation
- [ ] Metrics within bounds
- [ ] No manual interventions required
- [ ] Add ETH-PERP after 48h
- [ ] Increase notional gradually

## Go/No-Go Decision Matrix

| Component | Green (Go) | Yellow (Caution) | Red (No-Go) |
|-----------|------------|------------------|-------------|
| Order Success Rate | > 95% | 90-95% | < 90% |
| Post-Only Rejects | < 5% | 5-10% | > 10% |
| Slippage | < 5 bps | 5-10 bps | > 10 bps |
| System Latency | < 100ms | 100-500ms | > 500ms |
| Error Rate | < 0.1% | 0.1-1% | > 1% |
| PnL Tracking Drift | < 0.1% | 0.1-0.5% | > 0.5% |

## Kill Switch Procedures

### Immediate Shutdown
```bash
# Send SIGTERM to gracefully shutdown
kill -TERM <pid>

# Or use reduce-only mode
curl -X POST localhost:8080/reduce-only
```

### Reduce-Only Mode
```bash
# Switch to reduce-only (no new entries)
python scripts/run_perps_bot_v2_week3.py --reduce-only
```

### Emergency Position Close
```bash
# Close all positions immediately
python scripts/emergency_close_all.py --confirm
```

## Rollback Plan

If issues arise:
1. Enable reduce-only mode immediately
2. Monitor position closures
3. Revert to Week 2 runner if necessary:
   ```bash
   python scripts/run_perps_bot_v2.py --profile prod
   ```
4. Investigate logs and metrics
5. Fix issues in dev/sandbox
6. Re-attempt rollout with fixes

## Metrics Collection

### Key Metrics to Track
- **Strategy**: Entries accepted/rejected, rejection breakdown
- **Execution**: Orders placed/filled/cancelled/rejected
- **PnL**: Realized/unrealized, funding paid/received
- **Market**: Spread, depth, volume at decision time
- **System**: Latency, error rate, restarts

### Dashboard Query Examples
```sql
-- Order success rate
SELECT 
  COUNT(*) as total_orders,
  SUM(CASE WHEN status = 'filled' THEN 1 ELSE 0 END) as filled,
  SUM(CASE WHEN status = 'rejected' THEN 1 ELSE 0 END) as rejected,
  AVG(CASE WHEN status = 'filled' THEN 1 ELSE 0 END) * 100 as success_rate
FROM orders
WHERE timestamp > NOW() - INTERVAL '1 day';

-- PnL summary
SELECT 
  symbol,
  SUM(realized_pnl) as total_realized,
  SUM(funding_paid) as total_funding,
  COUNT(DISTINCT DATE(timestamp)) as days_active,
  SUM(realized_pnl) / COUNT(DISTINCT DATE(timestamp)) as daily_avg
FROM positions
GROUP BY symbol;
```

## Contact & Escalation

- **Primary**: Engineering team slack channel
- **Escalation**: On-call engineer
- **Emergency**: Trading desk hotline
- **Logs**: `/data/perps_bot_v3/logs/`
- **Metrics**: EventStore + Grafana dashboard

## Sign-off

- [ ] Engineering approval
- [ ] Risk management review
- [ ] Trading desk acknowledgment
- [ ] Operations readiness confirmed

---

*Last Updated: 2025-08-30*
*Version: Week 3 Advanced Orders*