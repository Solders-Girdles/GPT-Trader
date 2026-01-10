# Production Deployment Guide

---
status: current
last-updated: 2025-10-07
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
- [ ] uv package manager installed
- [ ] Git repository cloned
- [ ] Environment variables configured (.env)

### API Configuration
- [ ] CDP JWT credentials available for spot trading (required)
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
uv sync

# Configure environment
cp config/environments/.env.template .env
# Edit `.env` with production spot values (set COINBASE_ENABLE_DERIVATIVES=0 unless INTX approved)
```

### 2. Pre-flight Validation
```bash
# Run comprehensive checks (env, credentials, risk toggles)
uv run python scripts/production_preflight.py --profile canary

# Smoke test the trading loop
uv run gpt-trader run --profile dev --dev-fast

# Inspect streaming telemetry via TUI
uv run gpt-trader tui                  # Mode selector
uv run gpt-trader run --profile dev --tui  # Attach TUI to dev profile (optional)
```

### 3. Canary Deployment
```bash
# Start with canary profile (ultra-safe)
uv run gpt-trader run --profile canary --dry-run

# Monitor for 24 hours
# Check logs: tail -f ${COINBASE_TRADER_LOG_DIR:-var/logs}/coinbase_trader.log

# If successful, enable live spot trading
uv run gpt-trader run --profile canary
```

### 4. Production Rollout
```bash
# Gradual scaling approach (spot)
uv run gpt-trader run --profile prod --dry-run             # Validate config under prod settings
uv run gpt-trader run --profile prod --reduce-only         # Warm start with exits only
uv run gpt-trader run --profile prod                       # Full trading once stable
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
- ✅ 100% pass rate on required spot test suite (`uv run pytest -q`)
- ✅ WebSocket reconnection logic
- ✅ Rate limiting implementation
- ✅ Error handling and recovery
- ✅ Comprehensive logging

### Operational Requirements
- ✅ Monitoring dashboards configured (TUI)
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
pkill -f gpt-trader

# 4. Review logs and diagnose
tail -n 1000 ${COINBASE_TRADER_LOG_DIR:-var/logs}/coinbase_trader.log | grep ERROR
```

## Emergency Procedures

### Emergency Kill Switch
```bash
# STOP ALL TRADING IMMEDIATELY
export RISK_KILL_SWITCH_ENABLED=1
pkill -f gpt-trader
```

### Close All Positions
```bash
# Enable reduce-only mode
export RISK_REDUCE_ONLY_MODE=1

# Submit market exits via CLI
uv run gpt-trader orders preview \
  --symbol BTC-USD --side sell --type market --quantity CURRENT_SIZE
```

### Common Issues

**Excessive Losses**
1. Check if daily loss guard tripped (`daily_loss_limit_reached`)
2. Review recent fills in logs
3. Reduce position sizes or halt trading

**API Authentication Failures**
1. Run preflight: `uv run python scripts/production_preflight.py --profile canary`
2. Check API key permissions
3. Verify system time sync

**Connection Problems**
1. Check network: `ping api.coinbase.com`
2. Check status: https://status.coinbase.com/
3. Switch to REST-only: `export PERPS_ENABLE_STREAMING=0`

### Recovery After Emergency
1. Assess situation and document incident
2. Close or manage positions
3. Root cause analysis (review logs)
4. Implement fixes
5. Restart with paper trading first
6. Move to canary profile
7. Monitor closely for 24 hours

## Golden Rules

1. Never disable safety features in production
2. Always test changes in paper mode first
3. Start small and scale gradually
4. Monitor closely for first 24 hours after any change
5. Document all incidents and changes
6. When in doubt, reduce risk or stop trading
