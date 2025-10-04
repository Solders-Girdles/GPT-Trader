# 48-Hour Drift Review Checklist

Operational checklist for verifying post-deploy drift. Ensures market data, streaming, and other telemetry remain aligned with expectations after refactors.

**📋 Phase 1 Review Completed**: See [48H_DRIFT_REVIEW_PHASE_1.md](./48H_DRIFT_REVIEW_PHASE_1.md) for code-level validation results (all checks passed). Use this template for production deployment validation.

## 1. Deployment Context

- [ ] Deployment ID / SHA: `__________`
- [ ] Deployment date/time (UTC): `__________`
- [ ] Environment(s): `dev / canary / prod`
- [ ] Primary owner / on-call: `__________`
- [ ] Rollback window still open (Y/N): `__`

## 2. Drift Detection Summary

| Signal | Baseline | Current | Delta | Notes |
|--------|----------|---------|-------|-------|
| Mark update frequency | ___ Hz | ___ Hz | ___% | |
| Streaming heartbeat lag (p95) | ___ ms | ___ ms | ___% | |
| Risk manager mark freshness | ___s | ___s | ___% | |
| Order placement failure rate | ___% | ___% | ___% | |

**Findings:**
```
[Highlight anomalies, regressions, or confirmations]
```

**Mitigations:**
```
[If needed]
```

## 3. Performance Baseline Comparison

| Metric | Pre-Deploy | Post-Deploy | Delta | Status |
|--------|------------|-------------|-------|--------|
| Order latency (p99) | ___ms | ___ms | ___% | ⬜ |
| WS message lag (p99) | ___ms | ___ms | ___% | ⬜ |
| Backup duration (avg) | ___s | ___s | ___% | ⬜ |
| Memory usage (avg) | ___MB | ___MB | ___% | ⬜ |
| CPU usage (p95) | ___% | ___% | ___% | ⬜ |

**Analysis:**
```
[Significant changes? Root cause?]
```

## 4. Feature Flag Verification

- [x] Market data service delegation always on (flag retired Oct 2025)
- [x] Streaming service delegation always on (flag retired Oct 2025)
- [ ] `USE_PERPS_BOT_BUILDER=true` active
- [ ] `USE_NEW_CLI_HANDLERS=true` active
- [ ] No deprecation warnings in production logs

**Flag Status:**
```
[All enabled as expected? Any rollbacks needed?]
```

## 5. Telemetry & Observability

### New Metrics Available
- [ ] Liquidity scorer metrics appearing in dashboards
- [ ] Advanced execution telemetry logging
- [ ] Strategy handler performance metrics
- [ ] CLI command execution tracking

**Dashboard Health:**
```
[Are new metrics showing expected patterns?]
```

### Alerting
- [ ] No new false-positive alerts
- [ ] Existing alerts still firing correctly
- [ ] Heartbeat monitoring functioning
- [ ] No gaps in telemetry coverage

**Alert Review:**
```
[Any tuning needed?]
```

## 6. Integration Points

### External APIs
- [ ] Coinbase Advanced Trade latency within baseline
- [ ] WebSocket reconnect rate within baseline
- [ ] Rate limit errors unchanged

**Notes:**
```
[Anything notable with external dependencies]
```

### Internal Services
- [ ] Event store ingestion rate normal
- [ ] Orders store write volume normal
- [ ] Backup jobs succeeding

**Notes:**
```
[Anything notable across internal services]
```

## 7. Incident Review

- [ ] No new incidents during the 48h window
- [ ] Existing incidents resolved / mitigated
- [ ] Tickets created for any follow-up actions

**Summary:**
```
[Call out key learnings, outstanding risks, next steps]
```
