# Operations Runbook

---
status: current
last-updated: 2025-10-02
consolidates:
  - RUNBOOK_PERPS.md
  - LIVE_MONITORING_GUIDE.md
  - ADVANCED_ORDERS_GUIDE.md
updates:
  - 2025-10-02: Added State Recovery & Failover Procedures with batch operations (247x speedup)
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

### Spot Profile Daily Checklist
- **Creds & Env** – Ensure the deployment environment exports the correct Coinbase credentials and keeps `COINBASE_ENABLE_DERIVATIVES` unset unless INTX access is confirmed.
- **Dry-run smoke** – `poetry run perps-bot --profile dev --dev-fast` should complete without errors on staging infrastructure.
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
  --order-symbol BTC-USD --order-side buy --order-type market --order-quantity 0.01

# Preview a limit sell with a client ID
poetry run perps-bot --preview-order \
  --order-symbol BTC-USD --order-side sell --order-type limit \
  --order-quantity 0.01 --order-price 65000 --order-client-id demo-001

# Apply an edit using the preview id returned above
poetry run perps-bot --apply-order-edit "ORDER_ID:PREVIEW_ID"

# Preview a reduce-only stop exit
poetry run perps-bot --preview-order \
  --order-symbol BTC-USD --order-side sell --order-type stop_limit \
  --order-quantity 0.01 --order-price 60000 --order-stop 59500 --order-reduce-only
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
1. Guard will raise `market_data_staleness`. Switch to reduce-only mode with `poetry run perps-bot --profile spot --reduce-only`.
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

## State Recovery & Failover Procedures

### Failover Performance Expectations

The system uses **batch operations** for state recovery, dramatically reducing recovery time during infrastructure failures. Operators should expect these recovery timings:

| Failure Type | Recovery Time | Workload Size | Notes |
|--------------|---------------|---------------|-------|
| **Redis Outage** | ~8-20ms | 350-1000 keys | Restores from PostgreSQL WARM tier |
| **Trading Engine Crash** | ~30-45ms | 80-200 orders | Cancels pending orders, validates positions |
| **PostgreSQL Outage** | ~50ms | Varies | Restores from checkpoint (if available) |
| **Checkpoint Restoration** | 10-100x faster | Large snapshots | Batch writes eliminate individual RPCs |

**Comparison to Legacy Sequential Recovery:**
- Redis recovery: 2+ seconds → 8ms (**247x speedup**)
- Order cancellation: 46ms → 33ms (**1.4x speedup**)
- Checkpoint restore: Seconds to sub-second for typical workloads

### Automated Recovery Handlers

The system automatically attempts recovery for these failure scenarios:

#### 1. Redis Outage (HOT Tier Failure)
**Automatic Actions:**
- Detects Redis unavailability via connection timeout or error rate
- Triggers `storage.py:recover_redis()` handler
- Batch restores recent state from PostgreSQL WARM tier
- Repopulates cache with restored data
- Updates metadata (checksums, timestamps, versions)

**Verification:**
```bash
# Check Redis recovery logs
grep "Redis recovery" var/logs/perps_bot.log | tail -20

# Verify cache coherence after recovery
grep "Recovered.*keys to Redis" var/logs/perps_bot.log
```

**Expected Timeline:**
- Detection: 1-2 seconds (connection timeout)
- Recovery: ~8-20ms for typical workload
- Cache warmup: Automatic during batch restore
- **Total downtime: 1-2 seconds**

#### 2. Trading Engine Crash
**Automatic Actions:**
- Cancels all pending orders in batch operation
- Validates open positions for data integrity
- Removes invalid/corrupted positions
- Restores portfolio state from latest checkpoint
- Updates system status to "recovered"

**Verification:**
```bash
# Check trading engine recovery status
grep "Trading engine recovery" var/logs/perps_bot.log

# Verify order cancellations
grep "Cancelled.*pending orders" var/logs/perps_bot.log

# Check position validation
grep "position validation" var/logs/perps_bot.log
```

**Expected Timeline:**
- Detection: Immediate (crash/exception)
- Order cancellation: ~30-45ms for 80-200 orders
- Position validation: ~10-20ms
- **Total recovery: <100ms**

#### 3. PostgreSQL Outage (WARM Tier Failure)
**Automatic Actions:**
- Detects PostgreSQL unavailability
- Restores from latest checkpoint (if available)
- Falls back to backup manager if no checkpoint
- Updates data loss estimate based on checkpoint age
- Continues operation with HOT tier (Redis) only

**Verification:**
```bash
# Check PostgreSQL recovery
grep "PostgreSQL recovery" var/logs/perps_bot.log

# Check checkpoint restoration
grep "Restored from checkpoint" var/logs/perps_bot.log

# Verify data loss estimate
grep "data_loss_estimate" var/logs/perps_bot.log
```

#### 4. S3 Outage (COLD Tier Failure)
**Automatic Actions:**
- Configures local disk fallback for cold storage
- Updates system state: `system:s3_available = false`
- Continues operation with HOT/WARM tiers
- Queues data for S3 sync when service recovers

**Verification:**
```bash
# Check S3 recovery status
grep "S3 recovery" var/logs/perps_bot.log

# Verify local fallback
grep "local storage fallback" var/logs/perps_bot.log
```

### Manual Recovery Operations

For scenarios requiring manual intervention:

#### Force Checkpoint Restoration
```bash
# Restore from specific checkpoint
poetry run python -c "
from bot_v2.state.checkpoint.restoration import CheckpointRestoration
from bot_v2.state.state_manager import StateManager
import asyncio

async def restore():
    state_mgr = StateManager.create()
    restorer = CheckpointRestoration(state_mgr)
    success = await restorer.restore_from_checkpoint('checkpoint_id')
    print(f'Restoration: {\"SUCCESS\" if success else \"FAILED\"}')

asyncio.run(restore())
"
```

#### Verify Cache Coherence
```bash
# Run failover drill to validate recovery paths
poetry run python scripts/benchmarks/redis_failover_drill.py

# Expected output: Cache coherence validation should show "Match: True"
```

#### Batch State Operations (Advanced)

When manual state management is required:

```python
# Batch delete stale keys
from bot_v2.state.state_manager import StateManager
import asyncio

async def cleanup_stale_keys():
    state_mgr = StateManager.create()
    stale_keys = await state_mgr.get_keys_by_pattern("expired:*")
    deleted = await state_mgr.batch_delete_state(stale_keys)
    print(f"Deleted {deleted} stale keys")

asyncio.run(cleanup_stale_keys())
```

### Cache Coherence Guarantees

All batch operations maintain **perfect cache/storage consistency**:

✅ **Atomic Updates** - Cache updated only after successful storage
✅ **Partial Failure Handling** - Only successfully stored keys are cached
✅ **Metadata Consistency** - Checksums, timestamps, and versions stay synchronized
✅ **No Divergence** - Cache always reflects persisted state (verified via failover drill)

**Verification:**
- Sequential and batch recovery paths produce byte-identical cache states
- Failover drill confirms coherence under real-world conditions
- 574 tests validate batch operations, partial failures, and error handling

### Performance Tuning

#### Checkpoint Frequency
- **Default**: Every 5 minutes
- **High-frequency trading**: Every 1-2 minutes (enabled by batch performance)
- **Low-activity**: Every 10-15 minutes

More frequent checkpoints reduce data loss window with minimal performance penalty due to batch operations.

#### Recovery Optimization
```bash
# Enable aggressive checkpoint retention
export CHECKPOINT_RETENTION_COUNT=10

# Enable checkpoint compression (for large snapshots)
export CHECKPOINT_ENABLE_COMPRESSION=1

# Tune batch operation sizes
export STATE_BATCH_SIZE=1000  # Keys per batch operation
```

### Monitoring Recovery Operations

Key metrics to track during/after recovery:

```bash
# Recovery operation success rate
grep "recovery.*completed" var/logs/perps_bot.log | wc -l

# Average recovery time
grep "recovery.*elapsed" var/logs/perps_bot.log | awk '{sum+=$NF; count++} END {print sum/count "ms"}'

# Cache hit rate after recovery
grep "cache_hit_rate" var/logs/perps_bot.log | tail -10

# Data loss estimates
grep "data_loss_estimate" var/logs/perps_bot.log
```

## Operational Commands

- `poetry run perps-bot --profile canary --dry-run` – Protective canary run before deployments.
- `poetry run perps-bot --account-snapshot` – On-demand permissions audit.
- `poetry run python scripts/monitoring/export_metrics.py --metrics-file ...` – Start exporter locally.
- `poetry run pytest -q` – Full regression suite; run pre-deploy when code changes land.
- `poetry run python scripts/validation/verify_core.py --check all` – Quick health sweep of core orchestration surfaces.
- `poetry run python scripts/benchmarks/redis_failover_drill.py` – Validate failover performance and cache coherence.

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
- **Leverage batch operations for state management** (247x speedup for recovery)
- **Use batch order operations** when cancelling multiple orders
- Enable checkpoint compression for faster state restoration

### Batch Operations Benefits
The system uses batch operations internally for critical paths:

**State Recovery:**
- Redis failover: 2+ seconds → 8ms (247x speedup)
- Checkpoint restore: 10-100x faster than sequential
- Trading engine recovery: 1.4x faster

**Cache Coherence:**
- All batch operations maintain perfect cache/storage consistency
- Partial failures handled gracefully (only successful writes cached)
- No manual cache invalidation required

**When to Use Batch Operations:**
- Manual state cleanup (>10 keys)
- Emergency order cancellation (>5 orders)
- Bulk position validation
- Cold tier housekeeping

**Example - Batch State Cleanup:**
```python
# Efficient bulk cleanup using batch operations
from bot_v2.state.state_manager import StateManager
import asyncio

async def cleanup():
    state_mgr = StateManager.create()
    stale = await state_mgr.get_keys_by_pattern("expired:*")
    deleted = await state_mgr.batch_delete_state(stale)
    print(f"Cleaned {deleted} stale keys")

asyncio.run(cleanup())
```

### Resource Management
- Monitor memory usage
- Archive old log files
- Clean up temporary data
- **Use batch delete for bulk cleanup** (faster, maintains cache coherence)
- Optimize database queries with batch reads

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
