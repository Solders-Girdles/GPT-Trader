# Operations Runbook

---
status: current
last-updated: 2025-09-27
consolidates:
  - RUNBOOK_PERPS.md
  - LIVE_MONITORING_GUIDE.md
  - ADVANCED_ORDERS_GUIDE.md
---

## Overview

Complete operational procedures for GPT-Trader. The bot operates **spot-first**; enable perpetuals only after Coinbase grants INTX access and `COINBASE_ENABLE_DERIVATIVES=1` is set. Notes below call out derivatives-specific steps where relevant.

## Daily Operations

### Morning Checklist
- [ ] Run preflight checks: `poetry run python scripts/production_preflight.py --profile canary`
- [ ] Confirm streaming telemetry is updating (latest timestamps in `var/logs/perps_bot.log`)
- [ ] Review overnight PnL and positions via `poetry run perps-bot --account-snapshot`
- [ ] Scan `var/logs/perps_bot.log` for new ERROR/CRITICAL entries
- [ ] Verify recent market heartbeat metrics in `scripts/perps_dashboard.py`

### System Monitoring

#### Key Metrics to Monitor
- **Position Count**: Current open positions
- **Daily PnL**: Running profit/loss
- **Error Rate**: Percentage of failed operations
- **WebSocket Status**: Connection stability
- **API Rate Limits**: Usage vs. limits

#### Monitoring Commands
```bash
# System health check
poetry run python scripts/production_preflight.py --profile canary

# Single-cycle smoke of the bot loop
poetry run perps-bot --profile dev --dev-fast

# Account snapshot (balances, permissions, fee schedule)
poetry run perps-bot --account-snapshot

# Check recent errors
tail -n 100 var/logs/perps_bot.log | grep ERROR

# Live telemetry dashboard
poetry run python scripts/perps_dashboard.py --profile dev --refresh 5
```

## Advanced Order Management

### Order Types Supported
- **Market Orders**: Immediate execution at current price
- **Limit Orders**: Execute only at specified price or better
- **Stop Orders**: Trigger market order when price reached
- **Good Till Date (GTD)**: Orders with expiration
- **Reduce-Only Orders**: Can only reduce position size

### Order Tooling Examples (Spot)
```bash
# Preview a market buy
poetry run perps-bot --preview-order \
  --order-symbol BTC-USD --order-side buy --order-type market --order-qty 0.01

# Preview a limit sell with a client ID
poetry run perps-bot --preview-order \
  --order-symbol BTC-USD --order-side sell --order-type limit \
  --order-qty 0.01 --order-price 65000 --order-client-id demo-001

# Apply an edit using the preview id returned above
poetry run perps-bot --apply-order-edit "ORDER_ID:PREVIEW_ID"

# Preview a reduce-only stop exit
poetry run perps-bot --preview-order \
  --order-symbol BTC-USD --order-side sell --order-type stop_limit \
  --order-qty 0.01 --order-price 60000 --order-stop 59500 --order-reduce-only
```

> **Derivatives:** Swap the symbol to `*-PERP` only after INTX approval and derivatives are enabled.

## Live Monitoring

### Real-time Dashboard
```bash
# Launch terminal metrics dashboard (reads EventStore + health.json)
poetry run python scripts/perps_dashboard.py --profile dev --refresh 5 --window-min 15

# Access metrics via the console; Prometheus scraping remains unchanged.
```

### Alert Thresholds
| Metric | Warning | Critical | Action |
|--------|---------|----------|---------|
| Daily Loss | -1% | -2% | Enable reduce-only |
| Error Rate | 10% | 25% | Halt trading |
| WebSocket Down | 5 min | 15 min | Restart bot |
| Position Count | 10 | 20 | Review exposure |

### Log Analysis
```bash
# Real-time error monitoring
tail -f var/logs/perps_bot.log | grep -E "ERROR|CRITICAL"

# PnL tracking
grep "PnL" var/logs/perps_bot.log | tail -20

# Order execution analysis
grep "order.*filled" var/logs/perps_bot.log | tail -10
```

## Incident Response

### Emergency Procedures

#### 1. Emergency Stop
```bash
# Immediate halt of all trading
export RISK_KILL_SWITCH_ENABLED=1
pkill -f perps-bot
```

#### 2. Reduce-Only Mode
```bash
# Enable reduce-only (can only close positions)
export RISK_REDUCE_ONLY_MODE=1
```

#### 3. Position Emergency Close
```bash
# Close all positions immediately
export RISK_REDUCE_ONLY_MODE=1
# Submit closing orders via Coinbase UI or manual CLI previews.
echo "Issue market exits to flatten exposure while reduce-only is active."
```

### Common Issues

#### WebSocket Disconnections
**Symptoms**: No real-time data, stale prices
**Solutions**:
1. Check network connectivity
2. Restart bot to reconnect
3. Verify WebSocket URL in config

#### Order Rejections
**Symptoms**: Orders fail to place
**Solutions**:
1. Check position limits
2. Verify margin requirements
3. Review size increments
4. Check reduce-only constraints

#### High Error Rates
**Symptoms**: Many failed API calls
**Solutions**:
1. Check API rate limits
2. Verify authentication
3. Review network stability
4. Check Coinbase API status

## Maintenance Procedures

### Weekly Tasks
- [ ] Review performance metrics
- [ ] Analyze strategy effectiveness
- [ ] Update risk parameters if needed
- [ ] Archive old log files
- [ ] Review and optimize positions

### Monthly Tasks
- [ ] Full system backup
- [ ] Review and update documentation
- [ ] Analyze cost vs. returns
- [ ] Review security settings
- [ ] Update dependencies if needed

### Quarterly Tasks
- [ ] Comprehensive performance review
- [ ] Risk management assessment
- [ ] Strategy parameter optimization
- [ ] Infrastructure capacity planning
- [ ] Documentation consolidation review

## Performance Optimization

### Latency Optimization
- Use WebSocket for real-time data
- Enable HTTP keep-alive connections
- Minimize unnecessary API calls
- Implement efficient order batching

### Resource Management
- Monitor memory usage
- Archive old log files
- Clean up temporary data
- Optimize database queries

### Cost Management
- Monitor API usage costs
- Optimize trading frequency
- Review spread costs
- Analyze funding costs

## Backup and Recovery

### Daily Backups
```bash
# Backup configuration
cp .env /backup/env_$(date +%Y%m%d)

# Backup logs
tar -czf /backup/logs_$(date +%Y%m%d).tar.gz var/logs/

# Backup results
cp -r var/data/perps_bot/ /backup/perps_data_$(date +%Y%m%d)/
```

### Disaster Recovery
1. Stop all trading immediately
2. Assess data integrity
3. Restore from latest backup
4. Verify configuration
5. Test with paper trading first
6. Gradually resume live trading

## Contact Information

### Emergency Contacts
- System Admin: [Contact info]
- API Support: Coinbase Advanced Trade Support
- Development Team: [Contact info]

### Resources
- API Documentation: https://docs.cdp.coinbase.com/
- System Logs: `var/logs/perps_bot.log`
- Configuration: `.env`
- Status Page: [Internal status page]
