# Operations Runbook

---
status: current
last-updated: 2025-10-07
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
- [ ] Review overnight PnL and positions via `poetry run perps-bot account snapshot`
- [ ] Scan `var/logs/perps_bot.log` for new ERROR/CRITICAL entries
- [ ] Verify recent market heartbeat metrics in `scripts/perps_dashboard.py`

### Spot Profile Daily Checklist
- **Creds & Env** – Ensure the deployment environment exports the correct Coinbase credentials and keeps `COINBASE_ENABLE_DERIVATIVES` unset unless INTX access is confirmed.
- **Dry-run smoke** – `poetry run perps-bot run --profile dev --dev-fast` should complete without errors on staging infrastructure.
- **Metric scrape** – Verify `/metrics` is reachable from the Prometheus exporter and confirms recent timestamps.
- **Risk guard lookback** – Review guard counters (drawdown, staleness, volatility) for unexpected spikes.

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
poetry run perps-bot run --profile dev --dev-fast

# Account snapshot (balances, permissions, fee schedule)
poetry run perps-bot account snapshot

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
poetry run perps-bot orders preview \
  --symbol BTC-USD --side buy --type market --quantity 0.01

# Preview a limit sell with a client ID
poetry run perps-bot orders preview \
  --symbol BTC-USD --side sell --type limit \
  --quantity 0.01 --price 65000 --client-id demo-001

# Apply an edit using the preview id returned above
poetry run perps-bot orders apply-edit --order-id ORDER_ID --preview-id PREVIEW_ID

# Preview a reduce-only stop exit
poetry run perps-bot orders preview \
  --symbol BTC-USD --side sell --type stop_limit \
  --quantity 0.01 --price 60000 --stop 59500 --reduce-only
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

### Severity Matrix
| Severity | Criteria | First Actions |
|----------|----------|---------------|
| `SEV-1` | Bot offline, or guard-triggered halt with open exposure | Halt trading if not already stopped, contact on-call, collect logs, confirm balances |
| `SEV-2` | Partial functionality loss (e.g., account telemetry stalled) | Investigate recent deploys, restart affected service, monitor recovery |
| `SEV-3` | Degraded performance without financial impact | Create ticket, plan fix, monitor |

### Generic Recovery Flow
1. Identify failing component (bot loop, exporter, network).
2. Capture logs (`journalctl -u perps-bot` or container logs).
3. Restart the component using the supervisor used in your environment.
4. Confirm recovery via metrics and a controlled `--dev-fast` run.
5. File or update the incident record.

### Common Playbooks

#### Exchange/API Degradation
1. Guard will raise `market_data_staleness`. Switch to reduce-only mode with `poetry run perps-bot run --profile spot --reduce-only`.
2. Confirm Coinbase status and rate limits.
3. Once restored, re-enable full trading and watch the next five cycles.

#### Risk Guard Triggered
1. Check logs for the specific guard (daily loss, volatility, correlation).
2. Ensure the guard forced reduce-only or flat positions; if not, issue manual market orders using the preview tooling.
3. Document root cause and required config tweaks.

#### Metrics Exporter Gap
1. Validate the path passed to `--metrics-file` still updates.
2. Restart the exporter process.
3. Use `curl http://localhost:9102/metrics` to confirm Prometheus scrape readiness.

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

## Operational Commands

- `poetry run perps-bot run --profile canary --dry-run` – Protective canary run before deployments.
- `poetry run perps-bot account snapshot` – On-demand permissions audit.
- `poetry run python scripts/monitoring/export_metrics.py --metrics-file ...` – Start exporter locally.
- `poetry run pytest -q` – Full regression suite; run pre-deploy when code changes land.
- `poetry run python scripts/validation/verify_core.py --check all` – Quick health sweep of core orchestration surfaces.

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

## Change Management
1. Stage changes in feature branches; ensure documentation and tests update together.
2. CI must include `poetry run pytest -q` and any per-profile smoke tests.
3. Deploy via canary → prod progression with live monitoring at each stage.

## Knowledge Base
- `docs/ARCHITECTURE.md` – High-level system overview.
- `docs/MONITORING_PLAYBOOK.md` – Metrics and alert details.
- Repository history – Contains deprecated runbooks if historical context is needed.

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
