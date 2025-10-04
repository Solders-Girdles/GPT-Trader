# 48-Hour Drift Review - T0 Baseline Snapshot

**Deployment Context**
- Deployment SHA: `084055d` (Phase 1 Extended - 100% characterization coverage)
- Deployment date/time (UTC): 2025-10-04 22:18:00
- Environment: `staging (dev profile, dry-run mode)`
- Primary owner: Phase 1 Validation
- Rollback window: Open (48H)

---

## 1. System Configuration

**Bot Configuration:**
- Profile: `dev`
- Mode: `dry-run` (no real orders)
- Interval: 60 seconds (single cycle + exit + Docker restart)
- Symbols: BTC, ETH, SOL, XRP, LTC, ADA, DOGE, BCH, AVAX, LINK (10 total)
- Strategy: BaselinePerpsStrategy (short_ma=5, long_ma=20)
- Leverage: 1x (spot-like)

**Infrastructure:**
- Docker Compose stack (development target)
- Services: Postgres, Redis, RabbitMQ, Vault, Prometheus, Grafana, Elasticsearch, Kibana, Jaeger
- All services: Healthy and running

---

## 2. T0 Baseline Metrics (2025-10-04 22:25 UTC)

### Bot Performance Metrics (from logs)

| Metric | Value | Source |
|--------|-------|--------|
| Cycle execution time | 12-13ms | JSON metrics logs |
| Memory usage (bot process) | 2670-2705 MB | JSON metrics logs |
| CPU usage (bot process) | 4.5-7.3% | JSON metrics logs |
| Disk usage | 22.64 GB (2.4%) | JSON metrics logs |
| Network sent | 0 MB | JSON metrics logs |
| Network received | ~0.0003 MB | JSON metrics logs |
| Open files | 4 | JSON metrics logs |
| Threads | 13 | JSON metrics logs |

**Note:** Bot currently configured for single-cycle execution (completes cycle → exits clean → Docker restarts). For continuous 48H monitoring, bot needs to run in interval mode without exit.

### Strategy Execution Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| Symbols processed per cycle | 10 | All symbols |
| Strategy execution time (avg) | 0.007-0.020ms | Per symbol |
| Decision outcome | 100% hold | No signals (expected - insufficient MA data) |
| Order attempts | 0 | Dry-run mode |
| Order failures | 0 | N/A |

### Infrastructure Resource Usage

| Service | CPU % | Memory | Status |
|---------|-------|--------|--------|
| bot_v2_main | Restarting | Restarting | Single-cycle mode |
| Postgres | 0.00% | 19.25 MiB | Healthy |
| Redis | 0.85% | 9.49 MiB | Healthy |
| RabbitMQ | 45.18% | 126.6 MiB | Healthy |
| Elasticsearch | 0.83% | 1004 MiB | Healthy |
| Prometheus | 0.00% | 75.44 MiB | Running |
| Grafana | 0.31% | 296.4 MiB | Running |
| Kibana | 0.56% | 473.3 MiB | Running |
| Jaeger | 0.01% | 14.81 MiB | Running |
| Vault | 0.56% | 149.8 MiB | Running |

**Total Infrastructure Memory:** ~2.1 GB

---

## 3. Feature Flag Status

✅ **Confirmed Active:**
- Market data service delegation (flag retired, always on)
- Streaming service delegation (flag retired, always on)
- Builder pattern construction (mandatory, no legacy path)
- Modular test suite (100% active, 0 skipped)

✅ **Expected Behavior:**
- No streaming active (dry-run mode, dev profile)
- Background tasks disabled (dry-run mode)
- Reconciliation loops disabled (dry-run mode)
- Account telemetry disabled (mock broker, no endpoints)

---

## 4. Observability Status

### Metrics Endpoints
- ❌ Bot metrics endpoint: Not exposed (port 9090 not serving metrics)
- ✅ Prometheus: Running on port 9091
- ✅ Grafana: Running on port 3000
- ✅ Jaeger: Running on port 16686

### Logging
- ✅ Structured JSON logs active
- ✅ Correlation IDs present
- ✅ Event type tracking active
- ✅ Component tagging active
- ✅ Performance timing logs active

### Current Gaps
- No bot metrics exposed (needs /metrics endpoint implementation)
- No streaming metrics (streaming disabled in dry-run)
- No order execution metrics (dry-run mode)
- No mark update frequency metrics (REST-only, no continuous updates)

---

## 5. Known Limitations for 48H Monitoring

**Current Deployment Configuration Issues:**

1. **Single-Cycle Mode**: Bot exits after each cycle instead of running continuously
   - **Impact**: Cannot capture continuous metrics or track drift over 24-48H
   - **Fix Required**: Update command to use interval mode without exit

2. **No Metrics Endpoint**: Bot doesn't expose Prometheus metrics on :9090
   - **Impact**: Cannot track bot metrics in Prometheus/Grafana
   - **Fix Required**: Implement /metrics endpoint or configure metrics export

3. **Dry-Run Mode**: No real trading activity
   - **Impact**: Cannot measure order latency, execution metrics, or risk manager updates
   - **Fix Required**: Switch to canary profile with small position sizes for realistic metrics

4. **Mock Broker**: Using deterministic broker (REST-first marks)
   - **Impact**: No WebSocket streaming metrics, no real API latency
   - **Fix Required**: Configure real Coinbase Advanced Trade broker for staging

---

## 6. Recommended Adjustments for 48H Drift Review

### Critical (Required for meaningful drift monitoring):

1. **Enable Continuous Operation**
   ```bash
   # Update docker-compose.yaml command:
   command: ["python", "-m", "bot_v2", "--profile", "canary", "--interval", "60"]
   # Remove "--dry-run" to enable mark updates and background tasks
   ```

2. **Implement Metrics Export**
   - Add Prometheus metrics endpoint to PerpsBot
   - Or configure structured log → metrics bridge
   - Expose on port 9090 as configured in Prometheus

3. **Switch to Canary Profile**
   - Use real Coinbase broker (not mock)
   - Enable WebSocket streaming
   - Enable background tasks (reconciliation, guards)
   - Use small position sizes for safety

### Nice-to-Have (Enhanced observability):

1. Configure Grafana dashboards for bot metrics
2. Set up alerting rules in Prometheus
3. Enable Jaeger tracing for request flows
4. Configure log aggregation in Elasticsearch

---

## 7. Next Steps

**Immediate (before T+24H snapshot):**
1. ⚠️ **Fix single-cycle restart loop** - enable continuous operation
2. ⚠️ **Add metrics endpoint** - enable Prometheus scraping
3. ⚠️ **Consider canary mode** - get realistic metrics (optional)

**24H Checkpoint:**
- Capture T+24H metrics from logs/Prometheus
- Compare against T0 baseline
- Identify any drift or anomalies
- Document findings

**48H Checkpoint:**
- Capture T+48H metrics
- Calculate final drift percentages
- Complete checklist validation
- Approve/reject Phase 1 for production

---

## 8. Initial Observations

### ✅ Positive Indicators:
- All infrastructure services healthy
- Bot cycles completing successfully (0 errors)
- Structured logging working correctly
- Clean shutdowns every cycle
- Memory usage stable (~2.7GB per cycle)

### ⚠️ Operational Concerns:
- Cannot run 48H continuous monitoring in current configuration
- No metrics endpoint for Prometheus integration
- Dry-run mode limits realistic metric capture
- Single-cycle restart pattern prevents baseline establishment

### 📋 Action Items:
1. Update deployment configuration for continuous operation
2. Implement or expose metrics endpoint
3. Re-baseline with continuous operation before 24H checkpoint
4. Consider switching to canary profile for production-like metrics

---

**Baseline Captured By**: Phase 1 Drift Review
**Next Review**: T+24H (2025-10-05 22:25 UTC)
**Final Review**: T+48H (2025-10-06 22:25 UTC)
