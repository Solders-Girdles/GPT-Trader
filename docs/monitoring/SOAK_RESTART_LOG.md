# Short Soak Restart Log

**Issue**: Health check configuration incompatible with CLI interval mode
**Resolution**: Disabled health check, restarted soak
**Impact**: Soak timer reset, new T+2H target established

---

## Timeline

**Original Soak Start**: 2025-10-04 22:29 UTC
**Preliminary Capture**: 2025-10-04 22:35 UTC (~13 min, too early)
**Health Check Issue Identified**: Container reported "unhealthy"
**First Fix Attempt**: Changed health check from curl to pgrep (22:44 UTC)
**Second Fix**: Disabled health check (pgrep not available in slim image) (22:45 UTC)
**Soak Restart**: 2025-10-04 22:45 UTC

**New T+2H Target**: 2025-10-05 00:45 UTC (approximately)

---

## Issue Details

### Problem:
Docker Compose health check configured for HTTP endpoint:
```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
```

Bot running in CLI interval mode (no HTTP server) → health check fails → container marked "unhealthy"

### Fix Attempts:

1. **Attempt 1 - Use pgrep**:
   ```yaml
   healthcheck:
     test: ["CMD", "pgrep", "-f", "python -m bot_v2"]
   ```
   **Result**: Failed - `pgrep` not available in `python:3.12-slim` image

2. **Attempt 2 - Disable health check**:
   ```yaml
   # Health check disabled for CLI interval mode (no HTTP server)
   # healthcheck:
   #   test: ...
   ```
   **Result**: ✅ Success - container runs without false "unhealthy" status

---

## Preliminary Results (T+0.22H / 13 min)

From early capture before restart:

**Positive Indicators**:
- ✅ Memory stable: 2.75 GB, fluctuation < 2%
- ✅ CPU stable: 1-2%
- ✅ No restarts (before health check fix)
- ✅ No critical errors

**Observations**:
- ⚠️ Threads grew from ~17 → 29 (may settle during longer soak)
- ⚠️ Container "unhealthy" (health check issue, now resolved)

---

## Current Status (Post-Restart)

**Soak Start Time**: 2025-10-04 22:45 UTC
**Current State**:
- Container: Running (no health status)
- Uptime: 35 seconds (will grow)
- Memory: 2765 MB
- CPU: 1.9%
- Threads: 17 (initial)
- Status: ✅ Stable

**Next Capture**: 2025-10-05 00:45 UTC (~2 hours from restart)

---

## Recommendations

### For Future Deployments:

1. **CLI Mode**: Don't configure HTTP health checks for CLI interval mode
   - Use process check (if procps available)
   - Or disable health check entirely
   - Container restart policy handles crashes

2. **Web Mode**: HTTP health check appropriate when bot runs web server
   - Implement `/health` endpoint
   - Use curl-based health check

3. **Metrics Endpoint**: If implementing Prometheus metrics:
   - Run HTTP server on port 9090
   - Health check can ping `/metrics` or `/health`
   - Update compose file accordingly

---

## Impact on Phase 1 Review

**Minimal Impact**:
- Soak restart necessary to fix configuration
- New 2-hour soak from 22:45 UTC
- Preliminary results showed positive trends
- Full T+2H capture will occur at 00:45 UTC

**No Change to Validation Plan**:
- Still running 2-4 hour short soak
- Still marking Phase 1 provisionally complete after T+2H
- Full 48H review still deferred to post-Phase 2

---

## Updated Capture Instructions

**New T+2H Time**: 2025-10-05 00:45 UTC (approximately)

**Capture Command** (same as before):
```bash
cd /Users/rj/PycharmProjects/GPT-Trader/deploy/bot_v2/docker
./capture_t2h_metrics.sh > T2H_METRICS_FINAL.txt
cat T2H_METRICS_FINAL.txt
```

**Expected Results**:
- Uptime: ~7200 seconds (2 hours)
- Memory: < 20% growth from ~2765 MB baseline
- CPU: < 5% average
- Threads: Stable (monitor for growth beyond normal initialization)
- Restarts: 0
- Errors: 0

---

**Restart Logged By**: Phase 1 Drift Review Team
**Next Action**: Capture at T+2H (00:45 UTC)
