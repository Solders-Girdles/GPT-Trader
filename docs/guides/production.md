# Production Deployment Guide

---
status: current
last-updated: 2025-09-27
consolidates:
  - PRODUCTION_LAUNCH_CHECKLIST.md
  - PRODUCTION_DEPLOYMENT_RUNBOOK.md
  - PRODUCTION_ROLLOUT_PLAN.md
  - PRODUCTION_READINESS_IMPLEMENTATION.md
---

## Overview

This guide consolidates all production deployment documentation for GPT-Trader with a **spot-first** posture. Perpetual futures support remains dormant until Coinbase grants INTX access and `COINBASE_ENABLE_DERIVATIVES=1` is set.

## Pre-Launch Checklist

### System Requirements
- [ ] Python 3.12+ installed
- [ ] Poetry dependency manager configured
- [ ] Git repository cloned
- [ ] Environment variables configured (.env)

### API Configuration
- [ ] HMAC API key created for spot trading (required)
- [ ] INTX/CDP credentials created **only if** derivatives are enabled
- [ ] WebSocket connectivity verified
- [ ] Rate limits understood

### Risk Settings
- [ ] Daily loss limits configured
- [ ] Leverage caps set appropriately (applies when derivatives enabled)
- [ ] Reduce-only mode tested
- [ ] Circuit breakers enabled

## Deployment Steps

### 1. Environment Preparation
```bash
# Clone repository
git clone https://github.com/your-org/GPT-Trader.git
cd GPT-Trader

# Install dependencies
poetry install

# Configure environment
cp config/environments/.env.template .env
# Edit `.env` with production spot values (set COINBASE_ENABLE_DERIVATIVES=0 unless INTX approved)
```

### 2. Pre-flight Validation
```bash
# Run comprehensive checks (env, credentials, risk toggles)
poetry run python scripts/production_preflight.py --profile canary

# Smoke test the trading loop
poetry run perps-bot --profile dev --dev-fast

# Inspect streaming telemetry
poetry run python scripts/perps_dashboard.py --profile dev --refresh 5 --window-min 5
```

### 3. Canary Deployment
```bash
# Start with canary profile (ultra-safe)
poetry run perps-bot --profile canary --dry-run

# Monitor for 24 hours
# Check logs: tail -f var/logs/perps_bot.log

# If successful, enable live spot trading
poetry run perps-bot --profile canary
```

### 4. Production Rollout
```bash
# Gradual scaling approach (spot)
poetry run perps-bot --profile prod --dry-run             # Validate config under prod settings
poetry run perps-bot --profile prod --reduce-only         # Warm start with exits only
poetry run perps-bot --profile prod                       # Full trading once stable
```

## Phased Rollout Plan

### Phase 1: Canary Testing (Days 1-3)
- Ultra-conservative settings
- 0.01 BTC max positions
- $10 daily loss limit
- 14:00-15:00 UTC window only

### Phase 2: Limited Production (Days 4-7)
- Increase position sizes to 0.1 BTC
- Expand trading window to 8 hours
- Monitor performance metrics

### Phase 3: Full Production (Day 8+)
- Full position sizing
- 24/7 operation
- All strategies enabled

### Optional: Enabling Derivatives
- Confirm Coinbase has approved INTX access for the account
- Set `COINBASE_ENABLE_DERIVATIVES=1` and configure CDP keys in `.env`
- Re-run pre-flight checks to validate derivatives connectivity
- Increase leverage caps and risk tolerances cautiously

## Production Readiness Requirements

### Technical Requirements
- ✅ 100% pass rate on required spot test suite (`poetry run pytest -q`)
- ✅ WebSocket reconnection logic
- ✅ Rate limiting implementation
- ✅ Error handling and recovery
- ✅ Comprehensive logging

### Operational Requirements
- ✅ Monitoring dashboards configured
- ✅ Alert thresholds defined
- ✅ Incident response procedures
- ✅ Rollback plan documented

### Security Requirements
- ✅ API keys in environment variables only
- ✅ No secrets in code or logs
- ✅ Audit logging enabled
- ✅ Access controls configured

## Monitoring

### Key Metrics
- Position count and sizes
- Daily PnL tracking
- Error rates and types
- WebSocket connection status
- API rate limit usage

### Alert Thresholds
- Daily loss > 1% - WARNING
- Daily loss > 2% - CRITICAL (halt trading)
- Error rate > 10% - WARNING
- WebSocket disconnected > 5 min - CRITICAL

## Rollback Procedure

If issues arise:
```bash
# 1. Enable reduce-only mode immediately
export RISK_REDUCE_ONLY_MODE=1

# 2. Close all positions
export RISK_REDUCE_ONLY_MODE=1
# Submit market exits via Coinbase UI or CLI previews while reduce-only is active.

# 3. Stop the bot
pkill -f perps-bot

# 4. Review logs and diagnose
tail -n 1000 var/logs/perps_bot.log | grep ERROR
```

## Verification

For verification procedures, see [Verification Guide](verification.md).
