# Canary Deployment Decision Template

## Pre-Canary Checklist

### Demo Validation ✅
- [ ] Demo ran for minimum 30 minutes
- [ ] Acceptance rate ≥ 90%
- [ ] No critical errors
- [ ] PnL tracking accurate
- [ ] Funding calculations correct

### System Metrics ✅
- [ ] p50 latency < 200ms
- [ ] p95 latency < 500ms
- [ ] WebSocket reconnects < 3 per 30 min
- [ ] Staleness events < 3 per hour
- [ ] Memory usage stable

### Risk Controls ✅
- [ ] Kill switch tested and working
- [ ] Daily loss cap configured ($100)
- [ ] Position limits enforced
- [ ] Reduce-only mode functional
- [ ] Pre-funding quiet window active

## Canary Configuration

```yaml
Environment: Production
Duration: 30 minutes
Symbol: BTC-PERP
Position Size: 0.0001 BTC (minimal)
Max Loss: $50
Mode: CONSERVATIVE
Filters:
  - RSI confirmation: ON
  - Volatility filter: STRICT
  - Spread filter: < 20 bps
  - Depth requirement: > $100k
```

## Metrics Snapshot (Pre-Canary)

| Metric | Demo Value | Target | Status |
|--------|------------|--------|--------|
| Acceptance Rate | _____% | ≥ 90% | ⬜ |
| p50 Latency | _____ms | < 200ms | ⬜ |
| p95 Latency | _____ms | < 500ms | ⬜ |
| Order Success | _____% | > 95% | ⬜ |
| WS Uptime | _____% | > 99% | ⬜ |
| Error Rate | _____% | < 1% | ⬜ |

## Sign-off Requirements

### Technical Approval
- **Engineer**: _________________________ Date: __________ ⬜
  - Demo validation complete
  - System metrics within SLOs
  - No blocking issues

### Risk Approval
- **Risk Manager**: _____________________ Date: __________ ⬜
  - Risk controls verified
  - Loss limits appropriate
  - Kill switch tested

### Business Approval
- **Product Owner**: ____________________ Date: __________ ⬜
  - Business case validated
  - Success criteria defined
  - Rollback plan approved

### Final Authorization
- **CTO/Director**: _____________________ Date: __________ ⬜
  - All approvals obtained
  - Go/No-Go decision: ⬜ GO / ⬜ NO-GO

## Canary Execution Plan

### T-0: Pre-Launch (5 minutes before)
```bash
# 1. Final preflight
python scripts/preflight_check.py

# 2. Start monitoring
python scripts/dashboard_lite.py

# 3. Verify kill switch
python scripts/test_kill_switch.py

# 4. Check market conditions
python scripts/check_market_conditions.py --symbol BTC-PERP
```

### T+0: Launch Canary
```bash
# Launch with strict parameters
python -m src.bot_v2.canary_trader \
  --symbol BTC-PERP \
  --size 0.0001 \
  --max-loss 50 \
  --duration 1800 \
  --mode CONSERVATIVE \
  --rsi-confirm \
  --monitor
```

### T+5: Initial Validation (5 minutes)
- [ ] Orders being placed
- [ ] No immediate errors
- [ ] Metrics collecting
- [ ] Position tracking working

### T+15: Mid-Canary Check (15 minutes)
- [ ] Acceptance rate check
- [ ] Latency within bounds
- [ ] No risk limit breaches
- [ ] PnL reasonable

### T+30: Completion
- [ ] All positions closed
- [ ] Final metrics collected
- [ ] Logs archived
- [ ] Report generated

## Rollback Triggers

**IMMEDIATE ROLLBACK IF:**
- [ ] Loss exceeds $50
- [ ] Error rate > 5%
- [ ] Latency p95 > 1000ms
- [ ] WebSocket down > 2 minutes
- [ ] Unauthorized behavior detected

### Rollback Procedure
```bash
# 1. Activate kill switch
bash scripts/kill_switch.sh --close-all

# 2. Stop canary
systemctl stop canary-trader

# 3. Archive logs
python scripts/archive_canary_logs.py

# 4. Generate incident report
python scripts/generate_incident_report.py
```

## Success Criteria

### Minimum Requirements (Must Have All)
- ✅ No critical errors
- ✅ Loss < $50
- ✅ Acceptance rate > 85%
- ✅ All positions properly closed
- ✅ Audit trail complete

### Target Metrics (Should Have 3+)
- ⬜ Profitable or break-even
- ⬜ Acceptance rate > 95%
- ⬜ Zero WebSocket disconnects
- ⬜ p95 latency < 300ms
- ⬜ All risk filters working as expected

## Post-Canary Actions

### If Successful ✅
1. Archive canary data
2. Generate success report
3. Schedule production rollout
4. Update documentation
5. Team celebration 🎉

### If Failed ❌
1. Complete incident report
2. Root cause analysis
3. Fix identified issues
4. Schedule retry (minimum 48 hours)
5. Additional demo validation

## Monitoring Dashboard URLs

- Trading Dashboard: http://localhost:8080/dashboard
- Metrics: http://localhost:9090/metrics
- Logs: /tmp/trading_logs/canary.log
- Reports: docs/ops/preflight/canary/

## Emergency Contacts

| Role | Name | Phone | Email |
|------|------|-------|-------|
| On-Call Engineer | | | |
| Risk Manager | | | |
| Product Owner | | | |
| Escalation | | | |

## Notes Section

```
Pre-canary observations:
_________________________________________
_________________________________________
_________________________________________

Market conditions:
_________________________________________
_________________________________________

Special considerations:
_________________________________________
_________________________________________
```

---

**Document Status**: ⬜ DRAFT / ⬜ APPROVED / ⬜ EXECUTED
**Version**: 1.0
**Date**: _______________
**Next Review**: After canary completion