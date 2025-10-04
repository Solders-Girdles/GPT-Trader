# Continuous Mode Configuration for 48H Drift Review

**Configuration Date**: 2025-10-04 22:29 UTC
**Purpose**: Enable continuous bot operation for meaningful 48H drift monitoring

---

## Changes Applied

### 1. Docker Compose Configuration

**File**: `deploy/bot_v2/docker/docker-compose.yaml`

**Command Updated:**
```yaml
# BEFORE (single-cycle mode):
command: ["python", "-m", "bot_v2", "--profile", "dev", "--dry-run", "--interval", "60"]

# AFTER (continuous mode):
command: ["python", "-m", "bot_v2", "--profile", "dev", "--interval", "60"]
```

**Key Changes:**
- ❌ Removed `--dry-run` flag → enables background tasks and mark updates
- ✅ Kept `--interval 60` → bot runs continuously with 60s cycle interval
- ✅ Kept `--profile dev` → uses mock broker (safe for staging)

**Environment Variables Added:**
```yaml
environment:
  - BROKER=mock  # Explicitly use mock broker instead of real Coinbase
```

### 2. Operational Verification

**Bot Status After Deployment:**
```
Container: bot_v2_main
Status: Up 105+ seconds (continuous)
Ports: 8080 (web), 8443 (secure), 9090 (metrics)
Uptime: Increasing (not restarting)
```

**Metrics Observed:**
- Uptime: 105.6 seconds (continuously increasing)
- Threads: 23 (up from 13 - background tasks active)
- Memory: 2.76 GB (stable)
- CPU: 0.7% (low, stable)
- Cycle interval: ~5 seconds per symbol group

---

## Why These Changes Enable Drift Monitoring

### Without --dry-run flag:

1. **Background Tasks Active**
   - Runtime guards enabled
   - Position reconciliation enabled (if applicable)
   - Account telemetry enabled (if broker supports)
   - Mark update loops active

2. **State Persistence**
   - Mark windows accumulate over time
   - Risk manager state updates continuously
   - Uptime tracking accurate

3. **Realistic Metrics**
   - Continuous uptime tracking
   - Background task health monitoring
   - Memory/CPU trends over 48H
   - Network activity patterns

### With Mock Broker:

1. **Safe for Staging**
   - No real API calls to Coinbase
   - No real orders placed
   - No rate limit concerns
   - No credential requirements

2. **Deterministic Behavior**
   - Predictable mark prices
   - Consistent response times
   - Reproducible test scenarios

3. **Limitations**
   - No WebSocket streaming metrics (needs real broker)
   - No real API latency data
   - No actual order execution metrics

---

## Metrics Now Available for 48H Review

### From JSON Logs (Continuous):
- ✅ Uptime tracking
- ✅ Memory usage trends
- ✅ CPU usage patterns
- ✅ Thread count (background task health)
- ✅ Cycle execution time
- ✅ Strategy performance timing
- ✅ Network activity

### Still Missing (Requires Prometheus Endpoint):
- ❌ Prometheus metrics scraping
- ❌ Grafana dashboard visualization
- ❌ Historical metric queries
- ❌ Alert rule evaluation

### Not Available (Requires Real Broker):
- ❌ WebSocket streaming lag
- ❌ Real API latency (p99)
- ❌ Order execution metrics
- ❌ Rate limit tracking

---

## Next Steps for T+24H and T+48H Metrics

### Metric Collection Strategy:

**From Container Logs:**
```bash
# Extract uptime, memory, CPU from JSON logs
docker logs bot_v2_main 2>&1 | grep "metrics_update" | jq '.uptime_seconds, .system.memory_used_mb, .system.cpu_percent'

# Get thread count trend
docker logs bot_v2_main 2>&1 | grep "metrics_update" | jq '.system.threads'

# Calculate cycle execution frequency
docker logs bot_v2_main 2>&1 | grep "metrics_update" | jq -r '.timestamp' | head -100
```

**From Docker Stats:**
```bash
# Real-time resource usage
docker stats bot_v2_main --no-stream
```

**From Service Logs:**
```bash
# Check for errors/warnings
docker logs bot_v2_main 2>&1 | grep -E "ERROR|WARNING" | tail -50
```

### T+24H Snapshot (2025-10-05 22:29 UTC):

Capture:
- Uptime (should be ~86,400 seconds)
- Memory usage (check for leaks)
- CPU usage (check for spikes)
- Thread count (check for growth)
- Error/warning counts
- Cycle execution time trends

### T+48H Snapshot (2025-10-06 22:29 UTC):

Capture:
- Final uptime (should be ~172,800 seconds)
- Memory drift analysis
- CPU stability analysis
- Background task health
- Error frequency analysis
- Performance degradation check

---

## Troubleshooting

### If Bot Restarts During 48H Window:

1. Check logs for error before restart:
   ```bash
   docker logs bot_v2_main --tail 200 2>&1 | grep -B 20 "ERROR\|CRITICAL"
   ```

2. Check Docker restart count:
   ```bash
   docker inspect bot_v2_main | jq '.[0].RestartCount'
   ```

3. Check container exit code:
   ```bash
   docker inspect bot_v2_main | jq '.[0].State.ExitCode'
   ```

### If Memory Grows Continuously:

1. Calculate leak rate:
   ```bash
   # Compare T0, T+24H, T+48H memory values
   # Growth > 10% may indicate leak
   ```

2. Check thread count growth:
   ```bash
   # Threads should stabilize after initial startup
   # Continuous growth indicates thread leak
   ```

---

## Success Criteria for Continuous Mode

- ✅ Bot uptime > 24 hours without restart (for T+24H)
- ✅ Bot uptime > 48 hours without restart (for T+48H)
- ✅ Memory usage stable (< 10% growth over 48H)
- ✅ CPU usage stable (no sustained spikes)
- ✅ Thread count stable (no continuous growth)
- ✅ No critical errors in logs
- ✅ Background tasks remain healthy

---

## Current Status

**Deployment**: ✅ Complete
**Continuous Operation**: ✅ Verified (105+ seconds uptime)
**Background Tasks**: ✅ Active (23 threads)
**Metrics Collection**: ✅ JSON logs available
**Monitoring Period Start**: 2025-10-04 22:29 UTC
**T+24H Checkpoint**: 2025-10-05 22:29 UTC
**T+48H Checkpoint**: 2025-10-06 22:29 UTC

**Ready for 48H drift review** ✅
