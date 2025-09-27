# Incident Response Standard Operating Procedures

## Quick Reference

| Incident Type | Severity | Response Time | First Action |
|--------------|----------|---------------|--------------|
| Key Compromise | P1 | < 5 min | Revoke key |
| Auth Failure | P2 | < 15 min | Check JWT expiry |
| WS Disconnect | P3 | < 30 min | Check backoff |
| High Latency | P3 | < 1 hour | Scale down |
| Loss Breach | P1 | Immediate | Kill switch |

## 1. Authentication/Authorization Failures

### Symptoms
- 401/403 errors from API
- JWT token rejection
- "Unauthorized" in logs

### Immediate Actions
```bash
# 1. Check token expiry
python -c "
import jwt, os
token = os.getenv('CURRENT_JWT')
decoded = jwt.decode(token, options={'verify_signature': False})
print(f'Expires: {decoded.get(\"exp\")}')
"

# 2. Regenerate JWT
python scripts/generate_jwt.py

# 3. Test authentication
curl -H "Authorization: Bearer $NEW_TOKEN" \
  https://api.coinbase.com/api/v3/brokerage/accounts

# 4. If persistent, switch to backup key
export COINBASE_CDP_API_KEY=$BACKUP_KEY
export COINBASE_CDP_PRIVATE_KEY_PATH=$BACKUP_KEY_PATH
```

### Backoff Strategy
- Initial retry: 1 second
- Exponential backoff: 2^n seconds (max 60s)
- Max retries: 5
- Alert ops after 3 failures

### Escalation
1. Check clock drift (> 30s will fail JWT)
2. Verify key permissions (400/600)
3. Check IP allowlist in CDP
4. Contact Coinbase support if persistent

## 2. WebSocket Disconnections

### Symptoms
- "WebSocket closed" messages
- Missing market data
- Stale price warnings

### Immediate Actions
```bash
# 1. Check current connections
python scripts/ws_probe.py --auth

# 2. Monitor reconnect attempts
tail -f logs/websocket.log | grep -E "connect|disconnect|error"

# 3. Test connectivity
ping -c 5 advanced-trade-ws.coinbase.com
```

### Recovery Procedure
1. **Pause entries** - Block new positions
2. **Allow exits** - Reduce-only mode
3. **Wait for reconnect** - Auto-backoff active
4. **Verify data flow** - Check staleness
5. **Resume trading** - After 2 clean minutes

### Alert Thresholds
- > 3 disconnects/hour: Warning
- > 10 disconnects/hour: Critical
- > 5 min downtime: Page on-call

## 3. Key Compromise

### IMMEDIATE ACTIONS (< 5 minutes)

```bash
#!/bin/bash
# EMERGENCY KEY REVOCATION

# 1. Kill all trading immediately
pkill -f trading_bot
systemctl stop trading-bot-perpetuals

# 2. Block network access
iptables -A OUTPUT -d api.coinbase.com -j DROP

# 3. Revoke compromised key (via UI or API)
# Via CDP Console: Settings → API Keys → Revoke

# 4. Switch to standby key
export COINBASE_CDP_API_KEY=$STANDBY_KEY
export COINBASE_CDP_PRIVATE_KEY_PATH=$STANDBY_KEY_PATH

# 5. Audit recent activity
python scripts/audit_trades.py --hours 24 > compromise_audit.log

# 6. Resume with new key
iptables -D OUTPUT -d api.coinbase.com -j DROP
systemctl start trading-bot-perpetuals
```

### Post-Incident
1. **Audit all trades** in compromise window
2. **Check for unauthorized** withdrawals/transfers
3. **Generate new standby** key immediately
4. **Update documentation** with incident details
5. **Security review** of access logs
6. **Notify compliance** if required

## 4. Daily Loss Breach

### Automatic Actions
- Trading halted automatically at limit
- All positions marked reduce-only
- Open orders cancelled
- Alert sent to team

### Manual Recovery
```bash
# 1. Verify positions
python scripts/get_positions.py

# 2. Check PnL calculation
python scripts/verify_pnl.py --detailed

# 3. Review trades that caused loss
python scripts/analyze_losing_trades.py --today

# 4. Reset for next day (after review)
python scripts/reset_daily_limits.py --confirm
```

### Root Cause Analysis
- Market conditions (volatility spike?)
- Strategy malfunction
- Data quality issues
- Execution problems

## 5. High Latency

### Symptoms
- p95 latency > 500ms
- Order timeouts
- Slow dashboard updates

### Immediate Actions
```bash
# 1. Check current latency
python scripts/measure_latency.py --samples 100

# 2. Scale down activity
export MAX_ORDERS_PER_SECOND=5
export BATCH_SIZE=10

# 3. Monitor improvement
watch -n 5 'python scripts/check_latency.py'
```

### Mitigation
1. Reduce order frequency
2. Batch operations
3. Increase cache TTL
4. Disable non-critical features
5. Switch to closer region if available

## 6. Funding Rate Spike

### Alert Threshold
- Funding rate > 0.1% (8-hour)
- Funding rate < -0.1% (8-hour)

### Actions
```bash
# 1. Check current funding
python scripts/get_funding_rates.py

# 2. Calculate impact
python scripts/funding_impact.py --position-size $SIZE

# 3. Decision tree
if [ $FUNDING_IMPACT -gt $DAILY_TARGET ]; then
    # Reduce or close position
    python scripts/reduce_position.py --symbol BTC-PERP --percent 50
fi
```

## 7. Kill Switch Activation

### Triggers
- Manual activation
- Daily loss breach
- Critical error count > threshold
- System compromise detected

### Kill Switch Procedure
```bash
#!/bin/bash
# kill_switch.sh

echo "🚨 KILL SWITCH ACTIVATED"
timestamp=$(date -u +%Y%m%d_%H%M%S)

# 1. Set reduce-only mode
echo "REDUCE_ONLY" > /tmp/trading_mode

# 2. Cancel all open orders
python scripts/cancel_all_orders.py --force > logs/kill_switch_$timestamp.log

# 3. Stop order placement
systemctl stop trading-bot-orders

# 4. Close positions if critical
if [ "$1" == "--close-all" ]; then
    python scripts/close_all_positions.py --confirm
fi

# 5. Alert team
python scripts/send_alert.py \
  --severity CRITICAL \
  --message "Kill switch activated at $timestamp"

echo "✅ Trading halted - Manual intervention required"
```

## 8. Staleness Events

### Detection
- No updates for > 5 seconds
- Timestamp drift > threshold
- Sequence gaps in feed

### Response
```python
# Staleness handler
def handle_staleness(symbol, last_update):
    age = time.time() - last_update
    
    if age > 5:  # 5 second threshold
        logger.warning(f"Stale data for {symbol}: {age:.1f}s old")
        
        # Block entries
        self.trading_enabled = False
        
        # Allow exits only
        self.reduce_only_mode = True
        
        # Try reconnect
        self.reconnect_websocket()
        
    if age > 30:  # Critical staleness
        # Activate kill switch
        self.activate_kill_switch()
```

## Escalation Matrix

### L1 Response (< 15 min)
- On-call engineer
- Access to all systems
- Can activate kill switch
- Handles P2-P4 incidents

### L2 Response (< 30 min)
- Senior engineer
- Key rotation authority
- Handles P1-P2 incidents
- Approves position closures

### L3 Response (< 1 hour)
- Engineering manager
- Business continuity decisions
- External communication
- Regulatory notifications

## Communication Templates

### Internal Alert
```
INCIDENT: [Type]
SEVERITY: [P1-P4]
TIME: [UTC timestamp]
IMPACT: [Positions/PnL affected]
STATUS: [Investigating/Mitigating/Resolved]
ACTIONS: [Current steps being taken]
ETA: [Resolution estimate]
```

### External Communication
```
We are currently experiencing [issue type].
Impact: [customer impact if any]
We are actively working on resolution.
ETA: [time estimate]
Updates: [update frequency]
```

## Post-Incident Review

### Required Documentation
1. Timeline of events
2. Root cause analysis
3. Impact assessment (PnL, positions)
4. Actions taken
5. Lessons learned
6. Prevention measures

### Review Meeting (within 48 hours)
- Incident commander
- On-call engineer
- Engineering manager
- Risk management
- Compliance (if required)

## Prevention Checklist

### Daily
- [ ] Review error logs
- [ ] Check latency metrics
- [ ] Verify position limits
- [ ] Test kill switch

### Weekly
- [ ] Rotate logs
- [ ] Update runbooks
- [ ] Review incidents
- [ ] Test backup systems

### Monthly
- [ ] Key rotation
- [ ] Security audit
- [ ] Disaster recovery drill
- [ ] Update contact list

---

*Version: 1.0*
*Last Updated: 2025-08-30*
*Review: Monthly*
*Owner: Operations Team*