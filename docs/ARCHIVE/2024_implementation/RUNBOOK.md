# Operational Runbook - Coinbase Perpetuals Trading

## Table of Contents
1. [System Overview](#system-overview)
2. [Startup Procedures](#startup-procedures)
3. [Shutdown Procedures](#shutdown-procedures)
4. [Monitoring](#monitoring)
5. [Incident Response](#incident-response)
6. [Common Issues](#common-issues)
7. [Emergency Procedures](#emergency-procedures)
8. [Maintenance](#maintenance)

## System Overview

### Architecture
```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Trading   │────▶│   Coinbase   │────▶│  Perpetuals │
│     Bot     │     │   CDP API    │     │   Markets   │
└─────────────┘     └──────────────┘     └─────────────┘
       │                    │                     │
       ▼                    ▼                     ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Monitoring │     │   WebSocket  │     │   Funding   │
│  Dashboard  │     │   Channels   │     │   Tracker   │
└─────────────┘     └──────────────┘     └─────────────┘
```

### Key Components
- **Trading Engine**: Order placement and management
- **Risk Manager**: Position limits and safety checks
- **Data Pipeline**: Market data and order flow
- **Monitor Service**: Health checks and alerts

## Startup Procedures

### Pre-Startup Checklist
```bash
# 1. Run preflight checks
python scripts/preflight_check.py

# 2. Verify API connectivity
python scripts/capability_probe.py

# 3. Check market conditions
python scripts/check_market_status.py

# 4. Validate configuration
python scripts/validate_config.py
```

### Startup Sequence

#### 1. Initialize Core Services
```bash
# Start in order
systemctl start redis
systemctl start postgres
systemctl start monitoring
```

#### 2. Start Trading Bot
```bash
# Development/Testing
python -m src.bot_v2.features.brokerages.coinbase.demo \
  --mode paper \
  --symbols BTC-PERP \
  --risk-level conservative

# Production
systemctl start trading-bot-perpetuals
```

#### 3. Verify Startup
```bash
# Check logs
tail -f /var/log/trading/startup.log

# Verify connections
curl http://localhost:8080/health

# Check WebSocket
python scripts/test_websocket.py
```

### Post-Startup Validation
- [ ] API authentication successful
- [ ] WebSocket channels connected
- [ ] Risk limits loaded
- [ ] Positions synchronized
- [ ] Monitoring active

## Shutdown Procedures

### Graceful Shutdown

#### 1. Prepare for Shutdown
```bash
# Set to reduce-only mode
curl -X POST http://localhost:8080/api/mode \
  -d '{"mode": "reduce_only"}'

# Wait for positions to close
python scripts/wait_for_flat.py --timeout 300
```

#### 2. Stop Trading
```bash
# Cancel all open orders
python scripts/cancel_all_orders.py

# Stop order placement
systemctl stop trading-bot-perpetuals
```

#### 3. Final Checks
```bash
# Verify no open orders
python scripts/verify_no_orders.py

# Check final positions
python scripts/get_positions.py

# Save state
python scripts/save_state.py
```

### Emergency Shutdown
```bash
# IMMEDIATE STOP - Kill Switch
python scripts/kill_switch.py

# Force cancel all orders
python scripts/force_cancel_all.py

# Block network access
iptables -A OUTPUT -d api.coinbase.com -j DROP

# Stop all services
systemctl stop trading-bot-perpetuals --force
```

## Monitoring

### Health Checks

#### API Health
```bash
# Check every 30 seconds
while true; do
  curl -s http://localhost:8080/health | jq .
  sleep 30
done
```

#### WebSocket Status
```python
# Monitor WebSocket connection
python scripts/monitor_websocket.py --interval 5
```

#### Position Monitoring
```bash
# Real-time position tracking
watch -n 1 'python scripts/get_positions.py --format table'
```

### Key Metrics

| Metric | Warning | Critical | Action |
|--------|---------|----------|--------|
| API Latency | > 500ms | > 1000ms | Check network, scale down |
| Order Success Rate | < 95% | < 90% | Review rejections |
| WebSocket Disconnects | > 3/hour | > 10/hour | Check connectivity |
| Position Size | > 75% limit | > 90% limit | Reduce exposure |
| Daily PnL | < -1% | < -2% | Trigger stop loss |

### Alerting Rules

```yaml
alerts:
  - name: high_api_latency
    condition: api_latency_p95 > 500
    action: notify_ops
    
  - name: position_limit_breach
    condition: position_size > max_position * 0.9
    action: reduce_positions
    
  - name: funding_rate_spike
    condition: abs(funding_rate) > 0.001
    action: alert_trader
```

## Incident Response

### Incident Levels

#### Level 1: Information
- Degraded performance
- Non-critical warnings
- **Response**: Monitor, document

#### Level 2: Warning
- Intermittent failures
- Approaching limits
- **Response**: Investigate, prepare fixes

#### Level 3: Critical
- Service disruption
- Position at risk
- **Response**: Immediate action required

#### Level 4: Emergency
- System compromise
- Major losses
- **Response**: Kill switch, escalate

### Response Procedures

#### Authentication Failures
```bash
# 1. Check token expiration
python scripts/check_jwt_expiry.py

# 2. Regenerate if needed
python scripts/refresh_jwt.py

# 3. Test authentication
python scripts/test_auth.py

# 4. If persistent, switch to backup key
export COINBASE_CDP_API_KEY=$BACKUP_KEY
systemctl restart trading-bot
```

#### Rate Limit Errors
```bash
# 1. Check current usage
python scripts/check_rate_limits.py

# 2. Implement backoff
export RATE_LIMIT_BACKOFF=exponential
export INITIAL_RETRY_DELAY=1

# 3. Reduce request rate
python scripts/throttle_requests.py --max-rps 10

# 4. Monitor recovery
tail -f /var/log/trading/rate_limits.log
```

#### Position Sync Issues
```bash
# 1. Get positions from API
python scripts/get_api_positions.py > api_positions.json

# 2. Compare with local state
python scripts/compare_positions.py \
  --api api_positions.json \
  --local local_state.json

# 3. Reconcile differences
python scripts/reconcile_positions.py

# 4. Verify sync
python scripts/verify_position_sync.py
```

## Common Issues

### Issue: WebSocket Keeps Disconnecting

**Symptoms**: Frequent reconnections, missed data
**Diagnosis**:
```bash
# Check connection logs
grep "websocket" /var/log/trading/connections.log | tail -50

# Test connectivity
python scripts/test_websocket_stability.py --duration 300
```

**Resolution**:
1. Check network stability
2. Verify JWT token fresh
3. Reduce subscription channels
4. Implement reconnection backoff

### Issue: Orders Rejected

**Symptoms**: High rejection rate, error messages
**Diagnosis**:
```bash
# Analyze rejections
python scripts/analyze_rejections.py --last-hour

# Common rejection reasons
grep "order_rejected" /var/log/trading/orders.log | \
  jq -r .reason | sort | uniq -c
```

**Resolution**:
1. Check minimum size requirements
2. Verify price/size quantization
3. Confirm sufficient margin
4. Review rate limits

### Issue: Funding Payments Unexpected

**Symptoms**: Large funding charges, timing issues
**Diagnosis**:
```bash
# Check funding rates
python scripts/get_funding_rates.py

# Review payment history
python scripts/funding_history.py --days 7
```

**Resolution**:
1. Monitor funding rates before entry
2. Adjust position timing
3. Implement funding rate limits
4. Consider hedging strategies

## Emergency Procedures

### Kill Switch Activation

```bash
#!/bin/bash
# kill_switch.sh

echo "🚨 ACTIVATING KILL SWITCH"

# 1. Cancel all orders immediately
python scripts/cancel_all_orders.py --force

# 2. Close all positions
python scripts/close_all_positions.py --market

# 3. Disable trading
touch /tmp/TRADING_DISABLED

# 4. Stop bot
systemctl stop trading-bot-perpetuals

# 5. Notify team
python scripts/send_alert.py \
  --severity CRITICAL \
  --message "Kill switch activated"

echo "✅ Kill switch activated - Trading stopped"
```

### Data Recovery

```bash
# 1. Stop services
systemctl stop trading-bot

# 2. Backup current state
tar -czf backup_$(date +%Y%m%d_%H%M%S).tar.gz \
  /var/lib/trading/state \
  /var/log/trading

# 3. Restore from backup
tar -xzf backup_YYYYMMDD_HHMMSS.tar.gz -C /

# 4. Verify data integrity
python scripts/verify_state_integrity.py

# 5. Restart with recovered data
systemctl start trading-bot
```

### Key Compromise Response

```bash
# IMMEDIATE ACTION REQUIRED

# 1. Revoke compromised key (< 2 minutes)
python scripts/emergency_key_revoke.py

# 2. Switch to backup key
export COINBASE_CDP_API_KEY=$EMERGENCY_KEY
export COINBASE_CDP_PRIVATE_KEY_PATH=$EMERGENCY_KEY_PATH

# 3. Audit recent activity
python scripts/audit_api_activity.py --hours 24

# 4. Check for unauthorized trades
python scripts/detect_unauthorized_trades.py

# 5. File security incident
python scripts/file_security_incident.py
```

## Maintenance

### Daily Tasks

```bash
# Morning checks (9:00 AM)
python scripts/daily_health_check.py
python scripts/reconcile_positions.py
python scripts/check_funding_schedule.py

# End of day (5:00 PM)
python scripts/generate_daily_report.py
python scripts/backup_state.py
python scripts/cleanup_logs.py --keep-days 7
```

### Weekly Tasks

```bash
# Monday
python scripts/analyze_weekly_performance.py
python scripts/optimize_parameters.py --dry-run

# Wednesday
python scripts/test_disaster_recovery.py
python scripts/verify_backups.py

# Friday
python scripts/security_audit.py
python scripts/update_documentation.py
```

### Monthly Tasks

```bash
# First Monday
python scripts/rotate_logs.py
python scripts/update_dependencies.py
python scripts/performance_review.py

# Mid-month
python scripts/capacity_planning.py
python scripts/cost_analysis.py

# End of month
python scripts/generate_monthly_report.py
python scripts/archive_old_data.py
```

## Performance Tuning

### Latency Optimization

```python
# Measure current latency
python scripts/measure_latency.py --samples 1000

# Optimize connection pooling
export CONNECTION_POOL_SIZE=10
export CONNECTION_TIMEOUT=5

# Enable request batching
export ENABLE_BATCH_REQUESTS=true
export BATCH_SIZE=50
export BATCH_TIMEOUT_MS=100
```

### Memory Management

```bash
# Monitor memory usage
python scripts/monitor_memory.py --interval 60

# Adjust cache sizes
export ORDER_CACHE_SIZE=1000
export PRICE_CACHE_TTL=5

# Enable garbage collection tuning
export PYTHONGC="100:10:10"
```

## Debugging

### Enable Debug Logging

```bash
# Set debug level
export LOG_LEVEL=DEBUG
export DEBUG_API_CALLS=true
export DEBUG_WEBSOCKET=true

# Restart with debugging
systemctl restart trading-bot

# Watch debug logs
tail -f /var/log/trading/debug.log | grep -E "ERROR|WARNING"
```

### Trace Specific Issues

```python
# Trace order lifecycle
python scripts/trace_order.py --order-id xxx-yyy-zzz

# Debug position calculation
python scripts/debug_position.py --symbol BTC-PERP

# Analyze WebSocket messages
python scripts/capture_websocket.py --duration 60 --save ws_dump.json
```

## Contacts

### Internal Escalation
1. **L1 Support**: ops-team@company.com
2. **L2 Engineering**: dev-team@company.com
3. **L3 Management**: cto@company.com

### External Support
- **Coinbase API**: api-support@coinbase.com
- **Emergency**: security@coinbase.com

### On-Call Schedule
- **Primary**: Check PagerDuty
- **Secondary**: Check OpsGenie
- **Manager**: Always available

---

*Runbook Version: 1.0*
*Last Updated: 2025-08-30*
*Next Review: Monthly*