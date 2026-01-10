# Canary Deployment Runbook

This runbook provides concrete steps, commands, Prometheus queries, and go/no-go criteria for canary production deployments.

## Overview

The canary deployment validates a new release against live production traffic with minimal risk exposure before full promotion.

| Phase | Duration | Criteria |
|-------|----------|----------|
| Preflight | ~5 min | All checks pass |
| Phase A (reduce-only) | 2-6 hours or 10+ reduce-only orders | No CRIT signals, no global pause |
| Phase B (micro-open) | 24h minimum or 50+ orders | No CRIT signals, stable metrics |
| Promotion | Manual | Exit criteria met |

Policy: Phase A is reduce-only (current canary profile). Phase B allows micro-positions for realistic execution validation. Promotion requires Phase B and health/metrics visibility.

Health and metrics endpoints are opt-in, but required for canary promotion.

---

## 0. Health and Metrics Endpoints (Required for Promotion)

The health server is opt-in. For canary promotion, `/health` and `/metrics` must be reachable.

- Ensure your deployment starts the health server (`gpt_trader.app.health_server.start_health_server`) and exposes port 8080.
- Enable the metrics endpoint:
  ```bash
  export GPT_TRADER_METRICS_ENDPOINT_ENABLED=1
  ```

If `/health` or `/metrics` are unavailable, you may run Phase A but promotion is a No-Go.

## 1. Preflight Checks

### 1.1 Run Preflight Command

```bash
uv run python scripts/production_preflight.py --profile canary
```

**Expected output:**
```
======================================================================
GPT-TRADER PRODUCTION PREFLIGHT CHECK
Profile: canary
Time: 2025-XX-XX HH:MM:SS UTC
======================================================================

[1/12] Python Version ............................ PASS
[2/12] Dependencies .............................. PASS
[3/12] Environment Variables ..................... PASS
[4/12] API Connectivity .......................... PASS
[5/12] Key Permissions ........................... PASS
[6/12] Risk Configuration ........................ PASS
[7/12] Pre-trade Diagnostics ..................... PASS
[8/12] Test Suite ................................ PASS
[9/12] Profile Configuration ..................... PASS
[10/12] System Time .............................. PASS
[11/12] Disk Space ............................... PASS
[12/12] Dry Run Simulation ....................... PASS

======================================================================
PREFLIGHT COMPLETE: 12/12 checks passed
======================================================================
```

**Exit code:** `0` = pass, `1` = fail

**If any check fails:**
```bash
# Run with verbose for diagnostics
uv run python scripts/production_preflight.py --profile canary --verbose

# Downgrade diagnostics to warnings (use only if safe to proceed)
uv run python scripts/production_preflight.py --profile canary --warn-only
```

### 1.2 Verify Canary Profile (BotConfig)

The live bot uses `config/profiles/canary.yaml` via `ProfileLoader`. Phase A is reduce-only. Phase B requires a micro-open profile (copy canary and set `trading.mode: normal`).

**Phase A (reduce-only) targets:**

| Field | Expected | Rationale |
|-------|----------|-----------|
| `trading.mode` | `reduce_only` | No new positions in Phase A |
| `risk_management.max_leverage` | `1` | No leverage |
| `risk_management.position_fraction` | `<= 0.001` | Micro sizing |
| `risk_management.daily_loss_limit` | `<= 10` | Minimal exposure |

**Phase B (micro-open) targets:**

| Field | Expected |
|-------|----------|
| `trading.mode` | `normal` |
| `risk_management.max_leverage` | `1` |
| `risk_management.position_fraction` | `<= 0.001` |
| `risk_management.daily_loss_limit` | `<= 10` |

**Profile check:**
```python
from gpt_trader.app.config.profile_loader import ProfileLoader
from gpt_trader.config.types import Profile

schema = ProfileLoader().load(Profile.CANARY)
print("mode:", schema.trading.mode)
print("max_leverage:", schema.risk.max_leverage)
print("position_fraction:", schema.risk.position_fraction)
print("daily_loss_limit:", schema.risk.daily_loss_limit)
```

Note: `ProfileLoader` reads `risk_management.position_fraction`. Fields under `trading.position_sizing` are not mapped into `BotConfig`.

If Phase B is required, create a local `config/profiles/canary_open.yaml` by copying `config/profiles/canary.yaml` and setting `trading.mode: normal`.

### 1.3 Verify Preflight RiskConfig (Env)

Preflight uses `RiskConfig.from_env()` (RISK_* env vars). This does not read the profile YAML. Ensure env risk is conservative and consistent with canary limits.

```bash
uv run agent-risk --with-docs
```

**Key bounds (Phase A / Phase B):**
- `max_leverage <= 1`
- `max_position_pct_per_symbol <= 0.001`
- `daily_loss_limit_pct <= 0.01` or `daily_loss_limit <= 10`
- `reduce_only_mode = 1` in Phase A, `0` in Phase B

**Manual verification:**
```python
from gpt_trader.features.live_trade.risk.config import RiskConfig

config = RiskConfig.from_env()
assert config.max_leverage <= 1
assert config.max_position_pct_per_symbol <= 0.001
assert config.daily_loss_limit_pct <= 0.01 or config.daily_loss_limit <= 10
print("RiskConfig (env) OK")
```

---

## 2. Health Checks

### 2.1 Component Health Status

```bash
curl -s http://localhost:8080/health | jq
```

**Expected response:**
```json
{
  "status": "healthy",
  "live": true,
  "ready": true,
  "checks": {
    "broker": { "status": "pass", "details": { "latency_ms": 150 } },
    "websocket": { "status": "pass", "details": { "connected": true, "stale": false } },
    "degradation": {
      "status": "pass",
      "details": { "global_paused": false, "reduce_only_mode": true }
    }
  },
  "signals": { "status": "OK" }
}
```

**Go/No-Go criteria:**

| Check | GO | NO-GO |
|-------|-----|--------|
| `status` | `healthy` or `degraded` | `unhealthy` or `starting` |
| `checks.broker.status` | `pass` | `fail` |
| `checks.websocket.details.connected` | `true` | `false` |
| `checks.websocket.details.stale` | `false` | `true` |
| `checks.degradation.details.global_paused` | `false` | `true` |
| `checks.degradation.details.reduce_only_mode` | Phase A: `true` / Phase B: `false` | mismatch |

### 2.2 Quick Health Verification Script

```bash
#!/bin/bash
# Example: save locally as scripts/canary_health_check.sh (not tracked)

set -e

echo "=== Canary Health Check ==="

# 1. Check broker ping
BROKER_STATUS=$(curl -s http://localhost:8080/health | jq -r '.checks.broker.status')
if [ "$BROKER_STATUS" != "pass" ]; then
  echo "FAIL: Broker check failed"
  exit 1
fi
echo "OK: Broker healthy"

# 2. Check WebSocket
WS_STATUS=$(curl -s http://localhost:8080/health | jq -r '.checks.websocket.status')
if [ "$WS_STATUS" != "pass" ]; then
  echo "FAIL: WebSocket check failed"
  exit 1
fi
echo "OK: WebSocket healthy"

# 3. Check no global pause
PAUSED=$(curl -s http://localhost:8080/health | jq -r '.checks.degradation.details.global_paused')
if [ "$PAUSED" == "true" ]; then
  echo "FAIL: Global trading paused"
  exit 1
fi
echo "OK: Trading not paused"

echo "=== All health checks passed ==="
```

---

## 3. Metrics Gate

### 3.1 Prometheus Queries and Thresholds

| Signal | Prometheus Query | WARN | CRIT |
|--------|------------------|------|------|
| Order Error Rate | `sum(rate(gpt_trader_order_submission_total{result=~"failed\|rejected"}[5m])) / sum(rate(gpt_trader_order_submission_total[5m]))` | >= 0.05 | >= 0.15 |
| Order Retry Rate | `(sum(rate(gpt_trader_broker_call_latency_seconds_count{operation="submit"}[5m])) - sum(rate(gpt_trader_order_submission_total[5m]))) / sum(rate(gpt_trader_order_submission_total[5m]))` | >= 0.10 | >= 0.25 |
| Broker Latency p95 | `histogram_quantile(0.95, sum(rate(gpt_trader_broker_call_latency_seconds_bucket{operation="submit",outcome="success"}[5m])) by (le)) * 1000` | >= 1000ms | >= 3000ms |
| Guard Trip Count | `sum(increase(gpt_trader_guard_trips_total[1h]))` | >= 3 | >= 10 |

WS staleness is evaluated via `/health` (no built-in WS timestamp metric is exported).

### 3.2 Alertmanager Rules

```yaml
# prometheus/rules/canary_alerts.yml
groups:
  - name: canary_health
    rules:
      # Order Error Rate
      - alert: CanaryOrderErrorRateWarn
        expr: |
          sum(rate(gpt_trader_order_submission_total{result=~"failed|rejected"}[5m]))
          / sum(rate(gpt_trader_order_submission_total[5m])) >= 0.05
        for: 2m
        labels:
          severity: warning
          component: canary
        annotations:
          summary: "Canary order error rate elevated"
          description: "Order error rate is {{ $value | humanizePercentage }}"

      - alert: CanaryOrderErrorRateCrit
        expr: |
          sum(rate(gpt_trader_order_submission_total{result=~"failed|rejected"}[5m]))
          / sum(rate(gpt_trader_order_submission_total[5m])) >= 0.15
        for: 1m
        labels:
          severity: critical
          component: canary
        annotations:
          summary: "Canary order error rate critical"
          description: "Order error rate is {{ $value | humanizePercentage }}"

      # Order Retry Rate
      - alert: CanaryOrderRetryRateWarn
        expr: |
          (
            sum(rate(gpt_trader_broker_call_latency_seconds_count{operation="submit"}[5m]))
            - sum(rate(gpt_trader_order_submission_total[5m]))
          ) / sum(rate(gpt_trader_order_submission_total[5m])) >= 0.10
        for: 5m
        labels:
          severity: warning
          component: canary
        annotations:
          summary: "Canary order retry rate elevated"
          description: "Order retry rate is {{ $value | humanizePercentage }}"

      - alert: CanaryOrderRetryRateCrit
        expr: |
          (
            sum(rate(gpt_trader_broker_call_latency_seconds_count{operation="submit"}[5m]))
            - sum(rate(gpt_trader_order_submission_total[5m]))
          ) / sum(rate(gpt_trader_order_submission_total[5m])) >= 0.25
        for: 2m
        labels:
          severity: critical
          component: canary
        annotations:
          summary: "Canary order retry rate critical"
          description: "Order retry rate is {{ $value | humanizePercentage }}"

      # Broker Latency
      - alert: CanaryBrokerLatencyWarn
        expr: |
          histogram_quantile(0.95,
            sum(rate(gpt_trader_broker_call_latency_seconds_bucket{operation="submit",outcome="success"}[5m])) by (le)
          ) * 1000 >= 1000
        for: 5m
        labels:
          severity: warning
          component: canary
        annotations:
          summary: "Canary broker latency elevated"
          description: "p95 latency is {{ $value | humanizeDuration }}"

      - alert: CanaryBrokerLatencyCrit
        expr: |
          histogram_quantile(0.95,
            sum(rate(gpt_trader_broker_call_latency_seconds_bucket{operation="submit",outcome="success"}[5m])) by (le)
          ) * 1000 >= 3000
        for: 2m
        labels:
          severity: critical
          component: canary
        annotations:
          summary: "Canary broker latency critical"
          description: "p95 latency is {{ $value | humanizeDuration }}"

      # Guard Trips
      - alert: CanaryGuardTripsWarn
        expr: sum(increase(gpt_trader_guard_trips_total[1h])) >= 3
        for: 0m
        labels:
          severity: warning
          component: canary
        annotations:
          summary: "Canary guard trips elevated"
          description: "{{ $value }} guard trips in last hour"

      - alert: CanaryGuardTripsCrit
        expr: sum(increase(gpt_trader_guard_trips_total[1h])) >= 10
        for: 0m
        labels:
          severity: critical
          component: canary
        annotations:
          summary: "Canary guard trips critical"
          description: "{{ $value }} guard trips in last hour"
```

### 3.3 Ad-hoc Metrics Check (No Prometheus)

```bash
# Requires GPT_TRADER_METRICS_ENDPOINT_ENABLED=1
curl -s http://localhost:8080/metrics | grep -E "gpt_trader_order_submission_total|gpt_trader_guard_trips_total"
```

---

## 4. Exit Criteria and Decision Tree

### 4.1 Promotion Criteria

Phase A exit criteria (reduce-only, 2-6 hours or 10+ orders):
- CRIT signals: zero
- Global pause events: zero
- WebSocket check: connected and not stale
- Reduce-only mode: true

Phase B promotion criteria (micro-open, 24 hours or 50+ orders):
- CRIT signals: zero
- Global pause events: zero
- Non-recoverable guard trips: zero
- Order error rate: < 5% sustained
- Broker latency p95: < 1000ms sustained
- WebSocket check: connected and not stale

### 4.2 Decision Tree

```
START
  │
  ├─► Preflight passes?
  │     NO  ──► FIX issues, re-run preflight
  │     YES ──▼
  │
  ├─► Health checks clean?
  │     NO  ──► Investigate component, DO NOT PROCEED
  │     YES ──▼
  │
  ├─► Phase A reduce-only window (2-6h / 10+ orders)
  │     │
  │     ├─► Any CRIT signal or global pause?
  │     │     YES ──► STOP canary, investigate, rollback if needed
  │     │     NO  ──▼
  │     │
  │     ├─► Phase A complete?
  │     │     NO  ──► Continue monitoring
  │     │     YES ──► Switch to Phase B (micro-open)
  │     │            (use canary_open profile)
  │     │
  │     └─► Phase B window (24h / 50+ orders)
  │           │
  │           ├─► Any CRIT signal or global pause?
  │           │     YES ──► STOP canary, investigate, rollback
  │           │     NO  ──▼
  │           │
  │           ├─► Guard trips > 10/hr?
  │           │     YES ──► PAUSE, investigate guard config
  │           │            If infra issue: fix and reset window
  │           │            If code issue: ROLLBACK
  │           │     NO  ──▼
  │           │
  │           ├─► Phase B window complete?
  │           │     NO  ──► Continue monitoring
  │           │     YES ──▼
  │           │
  │           └─► All criteria met?
  │                 NO  ──► Extend window or investigate
  │                 YES ──► PROMOTE TO PRODUCTION
  │
  └─► END
```

### 4.3 Rollback Procedure

If canary fails:

```bash
# 1. Stop canary instance (use your process manager)
# systemctl stop gpt-trader-canary
# docker stop gpt-trader-canary

# 2. Close open positions
# Use TUI panic (P) or close positions via the broker UI.
# In-process only: TradingBot.flatten_and_stop()

# 3. Revert to previous known-good version
git checkout <previous-release-tag>
uv sync

# 4. Restart with production profile
uv run gpt-trader run --profile prod
```

### 4.4 Promotion Procedure

When all criteria met:

```bash
# 1. Document canary results
uv run gpt-trader report daily --profile canary --report-format json --output-dir reports

# 2. Update production profile with any validated config changes
cp config/profiles/canary.yaml config/profiles/prod.yaml.new  # local staging copy
diff config/profiles/prod.yaml config/profiles/prod.yaml.new

# 3. Stop canary (use your process manager)
# systemctl stop gpt-trader-canary
# docker stop gpt-trader-canary

# 4. Deploy to production
uv run gpt-trader run --profile prod

# 5. Monitor production for 1 hour post-promotion
# (Use same health checks as canary)
```

---

## 5. Quick Reference

### Commands

| Action | Command |
|--------|---------|
| Run preflight | `uv run python scripts/production_preflight.py --profile canary` |
| Check env risk config | `uv run agent-risk --with-docs` |
| Health check | `curl http://localhost:8080/health \| jq` |
| Metrics endpoint | `curl http://localhost:8080/metrics` |
| Start Phase A | `uv run gpt-trader run --profile canary` |
| Start Phase B | `uv run gpt-trader run --profile canary_open` |
| Stop canary | `systemctl stop gpt-trader-canary` |

Note: Phase B assumes you created a local `config/profiles/canary_open.yaml` with `trading.mode: normal`.

### Thresholds Summary

| Metric | OK | WARN | CRIT |
|--------|-----|------|------|
| Order Error Rate | < 5% | 5-15% | >= 15% |
| Order Retry Rate | < 10% | 10-25% | >= 25% |
| Broker Latency p95 | < 1s | 1-3s | >= 3s |
| Guard Trips (1h) | < 3 | 3-10 | >= 10 |
| WS Staleness (health) | < 30s | 30-60s | >= 60s |

### Key Files

| Purpose | Path |
|---------|------|
| Canary profile | `config/profiles/canary.yaml` |
| Risk config | `src/gpt_trader/features/live_trade/risk/config.py` |
| Health checks | `src/gpt_trader/monitoring/health_checks.py` |
| Health signals | `src/gpt_trader/monitoring/health_signals.py` |
| Preflight CLI | `src/gpt_trader/preflight/cli.py` |
