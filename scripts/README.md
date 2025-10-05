# Soak Test Deployment Scripts

Automated scripts for Phase 3.2/3.3 Sandbox Soak Test deployment and monitoring.

## Quick Start

1. **Configure Credentials**
   ```bash
   cp .env.sandbox.example .env
   # Edit .env with your Coinbase Sandbox API credentials
   ```

2. **Deploy Soak Test**
   ```bash
   ./scripts/deploy_sandbox_soak.sh
   ```
   This automated script will:
   - Validate environment and credentials
   - Start monitoring stack (Prometheus, Grafana, Alertmanager)
   - Validate bot configuration
   - Start the bot for soak testing

3. **Monitor Status**
   ```bash
   ./scripts/check_soak_status.sh
   ```

4. **Collect Metrics** (run in background)
   ```bash
   ./scripts/collect_soak_metrics.sh &
   ```

5. **Export Final Metrics** (after 24-48hr)
   ```bash
   # Use actual start/end times from your test
   ./scripts/export_final_metrics.sh "2025-10-04T00:00:00Z" "2025-10-06T00:00:00Z"
   ```

## Scripts Reference

### `deploy_sandbox_soak.sh`

**Purpose**: Fully automated soak test deployment

**Prerequisites**:
- Docker Desktop running
- Poetry installed
- `.env` file configured with Coinbase Sandbox credentials

**What it does**:
1. Validates prerequisites (Docker, Poetry, credentials)
2. Installs/syncs dependencies via Poetry
3. Starts monitoring stack (Prometheus, Grafana, Alertmanager)
4. Validates bot configuration with `--validate-only` dry run
5. Displays configuration summary and prompts for confirmation
6. Starts bot with canary profile and streams logs

**Usage**:
```bash
./scripts/deploy_sandbox_soak.sh
```

**Output**:
- Logs to `logs/sandbox_soak_YYYYMMDD_HHMMSS.log`
- Monitoring dashboards at:
  - Prometheus: http://localhost:9091
  - Grafana: http://localhost:3000 (admin/admin123)
  - Alertmanager: http://localhost:9093

---

### `check_soak_status.sh`

**Purpose**: Quick health check for running soak test

**Usage**:
```bash
./scripts/check_soak_status.sh
```

**Checks**:
- Monitoring stack status (Prometheus, Grafana, Alertmanager)
- Bot health endpoint status
- Key metrics snapshot (uptime, streaming, guards, orders)
- Active alerts
- Recent log entries

**Example output**:
```
Monitoring Stack Status:
✓ Prometheus:    http://localhost:9091
✓ Grafana:       http://localhost:3000
✓ Alertmanager:  http://localhost:9093

Bot Health:
✓ Health endpoint: http://localhost:9090/health
  Status: ok
  Uptime: 14523s
  Streaming: connected
  Fallback: false
  Active Guards: 0

Key Metrics Snapshot:
  Streaming State:  1
  Guard Trips:      5
  Order Attempts:   23
  Order Success:    21

Active Alerts:
✓ No alerts firing
```

---

### `collect_soak_metrics.sh`

**Purpose**: Hourly metrics snapshot collection during soak test

**Usage**:
```bash
# Run in background
./scripts/collect_soak_metrics.sh &

# Or in a separate terminal
./scripts/collect_soak_metrics.sh
```

**Behavior**:
- Runs continuously until Ctrl+C
- Collects snapshots every hour (3600s)
- Saves to `data/soak_TIMESTAMP_*.json`

**Metrics collected**:
- `bot_uptime_seconds`
- `bot_guard_active`
- `bot_streaming_connection_state`
- `bot_guard_daily_loss_usd`
- `bot_streaming_fallback_active`
- `/health` endpoint snapshot
- Active alerts

**Stop collection**:
```bash
# Find process ID
ps aux | grep collect_soak_metrics

# Kill process
kill <PID>
```

---

### `export_final_metrics.sh`

**Purpose**: Export full time-series data after soak test completion

**Usage**:
```bash
./scripts/export_final_metrics.sh START_TIME END_TIME
```

**Example**:
```bash
# Export last 24 hours (macOS)
./scripts/export_final_metrics.sh \
  "$(date -u -v-24H +%Y-%m-%dT%H:%M:%SZ)" \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Export specific range
./scripts/export_final_metrics.sh \
  "2025-10-04T00:00:00Z" \
  "2025-10-06T00:00:00Z"
```

**Exports**:
- All bot metrics (`bot_.*`) at 15s resolution
- Guardrail metrics at 60s resolution
- Streaming metrics at 15s resolution
- Daily loss tracking at 60s resolution
- Reconnect counts at 60s resolution
- Order statistics at 60s resolution
- Alert history snapshot

**Output files** (in `data/`):
- `soak_full_metrics_TIMESTAMP.json` - All bot metrics
- `soak_guards_TIMESTAMP.json` - Guardrail states
- `soak_streaming_TIMESTAMP.json` - Streaming connection
- `soak_daily_loss_TIMESTAMP.json` - Daily P&L tracking
- `soak_reconnects_TIMESTAMP.json` - Reconnection attempts
- `soak_orders_TIMESTAMP.json` - Order execution stats
- `soak_alerts_TIMESTAMP.json` - Alert history

---

## Workflow

### During Soak Test

1. **Start deployment**:
   ```bash
   ./scripts/deploy_sandbox_soak.sh
   ```

2. **In a separate terminal, start metrics collection**:
   ```bash
   ./scripts/collect_soak_metrics.sh &
   ```

3. **Periodically check status** (every few hours):
   ```bash
   ./scripts/check_soak_status.sh
   ```

4. **Monitor dashboards**:
   - Grafana: http://localhost:3000 → Bot Health Overview dashboard
   - Review active alerts in Alertmanager: http://localhost:9093

5. **Execute test scenarios** (see `docs/testing/sandbox_soak_test_plan.md`):
   - Trigger order caps by placing large orders
   - Simulate daily loss accumulation
   - Test streaming disconnect/fallback
   - Force reconnect storms
   - Monitor circuit breaker behavior

### After Soak Test

1. **Stop bot**: Press Ctrl+C in deployment terminal

2. **Stop metrics collection**:
   ```bash
   pkill -f collect_soak_metrics
   ```

3. **Export final metrics**:
   ```bash
   # Get actual start time from bot logs
   START_TIME="2025-10-04T14:23:15Z"  # When bot started
   END_TIME="$(date -u +%Y-%m-%dT%H:%M:%SZ)"  # Now

   ./scripts/export_final_metrics.sh "$START_TIME" "$END_TIME"
   ```

4. **Archive data**:
   ```bash
   mkdir -p archive
   tar -czf archive/soak_test_$(date +%Y%m%d).tar.gz data/ logs/
   ```

5. **Generate report**:
   - Use Prometheus queries from `docs/testing/prometheus_queries.md`
   - Analyze exported JSON data
   - Document findings in `docs/testing/soak_test_results.md`

---

## Troubleshooting

### Monitoring stack won't start

**Symptom**: Docker containers not starting

**Solution**:
```bash
cd monitoring
docker-compose down
docker-compose up -d
docker-compose logs
```

### Bot health endpoint not responding

**Symptom**: `check_soak_status.sh` shows bot not responding

**Check**:
```bash
# Is bot running?
ps aux | grep bot_v2

# Check logs
tail -100 logs/sandbox_soak_*.log

# Try starting manually
poetry run python -m bot_v2.cli --profile canary --max-trade-value 100
```

### Metrics not being scraped

**Symptom**: Prometheus shows no bot metrics

**Check**:
1. Verify bot metrics endpoint: `curl http://localhost:9090/metrics`
2. Check Prometheus targets: http://localhost:9091/targets
3. Verify prometheus.yml scrape config points to correct port
4. Check firewall isn't blocking port 9090

### Credentials invalid

**Symptom**: Bot fails to connect to Coinbase

**Solution**:
1. Verify `.env` has correct sandbox credentials
2. Test API connection:
   ```bash
   curl -H "CB-ACCESS-KEY: $COINBASE_API_KEY" \
        https://api-public.sandbox.exchange.coinbase.com/products/BTC-USD
   ```

---

## Manual Deployment (Alternative)

If you prefer manual steps instead of `deploy_sandbox_soak.sh`:

```bash
# 1. Configure environment
cp .env.sandbox.example .env
# Edit .env with your credentials

# 2. Start monitoring
cd monitoring
export ADMIN_PASSWORD=admin123
export DATABASE_PASSWORD=trader
docker-compose up -d
cd ..

# 3. Install dependencies
poetry install --sync

# 4. Validate config
poetry run python -m bot_v2.cli --profile canary --dry-run --validate-only

# 5. Start bot
poetry run python -m bot_v2.cli \
  --profile canary \
  --max-trade-value 100 \
  --streaming-rest-poll-interval 5.0 \
  2>&1 | tee logs/sandbox_soak_$(date +%Y%m%d_%H%M%S).log
```

---

## Related Documentation

- **Test Plan**: `docs/testing/sandbox_soak_test_plan.md`
- **Deployment Checklist**: `docs/testing/sandbox_deployment_checklist.md`
- **Prometheus Queries**: `docs/testing/prometheus_queries.md`
- **Canary Profile**: `config/profiles/canary.yaml`
