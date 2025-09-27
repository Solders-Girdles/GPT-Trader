# Production Launch Checklist

## Pre-Launch Requirements

### ✅ Code Readiness
- [ ] All tests passing (pytest with 0 errors)
- [ ] No hardcoded credentials in codebase
- [ ] MockBroker interface aligned with IBrokerage
- [ ] All `from src.` imports corrected to `from bot_v2`
- [ ] Root directory cleaned (no loose Python files)

### ✅ Configuration
- [ ] Canary profile created and reviewed
- [ ] Environment variables documented
- [ ] API keys secured in vault/secrets manager
- [ ] Database connection strings configured
- [ ] Logging configuration tested

### ✅ Risk Controls
- [ ] Daily loss limits configured ($10 for canary)
- [ ] Position size limits set (0.01 BTC max)
- [ ] Circuit breakers configured
- [ ] Reduce-only mode verified
- [ ] Kill switch tested and accessible

## Day 0: Pre-Production Validation

### Morning (Before Market Open)
- [ ] Run full test suite: `pytest tests/ -v`
- [ ] Verify canary profile: `poetry run perps-bot --profile canary --dry-run`
- [ ] Check API connectivity: `python scripts/validate_perps_client_week1.py`
- [ ] Validate WebSocket connection: `python scripts/validate_ws_week1.py`
- [ ] Review risk parameters one final time

### Infrastructure Check
- [ ] Database is accessible and has space
- [ ] Monitoring dashboards are live
- [ ] Alerting channels tested (Slack/PagerDuty)
- [ ] Log aggregation working
- [ ] Backup systems verified

### Team Readiness
- [ ] On-call schedule confirmed
- [ ] Emergency contacts updated
- [ ] Runbook location shared
- [ ] Kill switch procedure documented
- [ ] Rollback plan reviewed

## Day 1: Canary Launch

### Pre-Market (30 minutes before)
```bash
# 1. Set environment
export COINBASE_PROFILE=canary
export LOG_LEVEL=DEBUG
export DRY_RUN=true

# 2. Preflight check
poetry run perps-bot --profile canary --preflight-only

# 3. Verify outputs
tail -f logs/preflight_$(date +%Y%m%d).log
```

### Market Open (First Hour)
```bash
# 1. Launch in dry-run mode first
poetry run perps-bot --profile canary --dry-run --duration 300

# 2. Review dry-run logs
grep -E "ERROR|WARNING" logs/canary_dryrun_*.log

# 3. If clean, launch canary (reduce-only mode)
poetry run perps-bot --profile canary --reduce-only

# 4. Monitor in real-time
tail -f logs/canary_$(date +%Y%m%d)_*.log | grep -E "ORDER|FILL|ERROR"
```

### Continuous Monitoring
- [ ] Watch order flow (should be minimal)
- [ ] Verify position sizes stay under limits
- [ ] Check PnL calculations match expectations
- [ ] Monitor error rate (< 1%)
- [ ] Validate event recording to database

### End of Day 1
- [ ] Review all trades executed
- [ ] Verify risk limits were respected
- [ ] Check for any unexpected behaviors
- [ ] Calculate actual vs expected PnL
- [ ] Document any issues or observations

## Day 2-5: Canary Observation

### Daily Checklist
- [ ] Morning: Review previous day's performance
- [ ] Check error logs for patterns
- [ ] Verify database records are complete
- [ ] Confirm risk metrics are within bounds
- [ ] Test kill switch (during low activity)

### Metrics to Track
```python
# Key metrics query
SELECT 
    DATE(created_at) as trade_date,
    COUNT(*) as total_trades,
    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
    SUM(pnl) as total_pnl,
    MAX(ABS(pnl)) as max_single_loss,
    AVG(execution_time_ms) as avg_latency
FROM trades
WHERE profile = 'canary'
GROUP BY DATE(created_at);
```

### Progressive Testing
- Day 2: Maintain reduce-only mode, observe behavior
- Day 3: Test emergency shutdown procedure
- Day 4: Test recovery from disconnection
- Day 5: Review for promotion decision

## Promotion to Production

### Promotion Criteria (All Must Pass)
- [ ] 5 days of stable canary operation
- [ ] Zero critical errors
- [ ] All risk limits properly enforced
- [ ] PnL calculations verified accurate
- [ ] Latency within acceptable range (< 100ms)
- [ ] Event recording 100% complete

### Production Profile Adjustments
```yaml
# Gradual increase from canary limits
Week 1: 
  - max_position_size: 0.05 BTC
  - daily_loss_limit: $50
  - max_concurrent_positions: 2

Week 2:
  - max_position_size: 0.10 BTC
  - daily_loss_limit: $100
  - max_concurrent_positions: 3

Week 3+:
  - max_position_size: 0.25 BTC
  - daily_loss_limit: $250
  - max_concurrent_positions: 5
```

## Emergency Procedures

### Kill Switch Activation
```bash
# Method 1: Signal file
touch /tmp/KILL_SWITCH_ACTIVATED

# Method 2: API call
curl -X POST https://api.yourbot.com/emergency/shutdown \
  -H "Authorization: Bearer ${KILL_SWITCH_KEY}"

# Method 3: Direct database
UPDATE bot_control SET shutdown = true WHERE profile = 'canary';
```

### Rollback Procedure
```bash
# 1. Stop current version
systemctl stop perps-bot

# 2. Revert to previous version
git checkout tags/last-stable-release

# 3. Restore configuration
cp config/profiles/canary.yaml.backup config/profiles/canary.yaml

# 4. Restart with safe mode
poetry run perps-bot --profile canary --safe-mode --reduce-only
```

### Incident Response
1. **Identify**: Alert received or issue observed
2. **Assess**: Determine severity and impact
3. **Contain**: Activate reduce-only or kill switch if needed
4. **Communicate**: Notify team via Slack/PagerDuty
5. **Remediate**: Fix issue or rollback
6. **Document**: Create incident report

## Post-Launch Review

### Week 1 Review
- [ ] Performance metrics analysis
- [ ] Error pattern identification
- [ ] Risk limit effectiveness
- [ ] Infrastructure bottlenecks
- [ ] Cost analysis (AWS, API calls, etc.)

### Optimization Opportunities
- [ ] Strategy parameter tuning
- [ ] Latency improvements
- [ ] Risk limit adjustments
- [ ] Infrastructure scaling needs
- [ ] Code optimizations identified

### Documentation Updates
- [ ] Update runbook with learnings
- [ ] Document any workarounds needed
- [ ] Update monitoring thresholds
- [ ] Revise emergency procedures
- [ ] Create knowledge base entries

## Sign-offs

### Technical Approval
- [ ] Lead Developer: _________________ Date: _______
- [ ] DevOps Lead: ___________________ Date: _______
- [ ] Security Review: _______________ Date: _______

### Business Approval  
- [ ] Risk Manager: __________________ Date: _______
- [ ] Product Owner: _________________ Date: _______
- [ ] Compliance: ____________________ Date: _______

## Notes Section

```
Launch Notes:
- 
- 
- 

Issues Encountered:
- 
- 
- 

Lessons Learned:
- 
- 
- 
```

---

**Remember**: Start small, monitor everything, and scale gradually. The canary profile is designed to fail safely - let it protect you during the learning phase.