# Operations Runbook

---
status: current
last-updated: 2025-01-01
consolidates:
  - RUNBOOK_PERPS.md
  - LIVE_MONITORING_GUIDE.md
  - ADVANCED_ORDERS_GUIDE.md
---

## Overview

Complete operational procedures for GPT-Trader. The bot operates **spot-first**; enable perpetuals only after Coinbase grants INTX access and `COINBASE_ENABLE_DERIVATIVES=1` is set. Notes below call out derivatives-specific steps where relevant.

## Daily Operations

### Morning Checklist
- [ ] Check system health: `python scripts/preflight_check.py`
- [ ] Verify WebSocket connections active
- [ ] Review overnight PnL and positions
- [ ] Check error logs for issues
- [ ] Validate market data freshness

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
python scripts/preflight_check.py

# Check WebSocket connectivity
python scripts/ws_probe.py

# View current positions
python -c "from bot_v2.cli import show_positions; show_positions()"

# Check recent errors
tail -n 100 logs/perps_bot.log | grep ERROR

# Monitor PnL in real-time
tail -f logs/perps_bot.log | grep "PnL"
```

## Advanced Order Management

### Order Types Supported
- **Market Orders**: Immediate execution at current price
- **Limit Orders**: Execute only at specified price or better
- **Stop Orders**: Trigger market order when price reached
- **Good Till Date (GTD)**: Orders with expiration
- **Reduce-Only Orders**: Can only reduce position size

### Order Placement Examples (Spot)
```bash
# Market buy order
poetry run perps-bot place-order --symbol BTC-USD --side buy --type market --size 0.01

# Limit sell order
poetry run perps-bot place-order --symbol BTC-USD --side sell --type limit --size 0.01 --price 65000

# Stop-loss order
poetry run perps-bot place-order --symbol BTC-USD --side sell --type stop --size 0.01 --stop-price 60000 --reduce-only

# Good-till-date order
poetry run perps-bot place-order --symbol BTC-USD --side buy --type limit --size 0.01 --price 63000 --gtd "2025-12-31T23:59:59Z"
```

> **Derivatives:** Swap the symbol to `*-PERP` only after INTX approval and derivatives are enabled.

## Live Monitoring

### Real-time Dashboard
```bash
# Launch monitoring dashboard
python scripts/dashboard_server.py --port 8080

# Access at: http://localhost:8080
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
tail -f logs/perps_bot.log | grep -E "ERROR|CRITICAL"

# PnL tracking
grep "PnL" logs/perps_bot.log | tail -20

# Order execution analysis
grep "order.*filled" logs/perps_bot.log | tail -10
```

## Incident Response

### Emergency Procedures

#### 1. Emergency Stop
```bash
# Immediate halt of all trading
export RISK_KILL_SWITCH=1
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
python scripts/emergency_close_positions.py --confirm
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
tar -czf /backup/logs_$(date +%Y%m%d).tar.gz logs/

# Backup results
cp -r results/ /backup/results_$(date +%Y%m%d)/
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
- System Logs: `logs/perps_bot.log`
- Configuration: `.env`
- Status Page: [Internal status page]
