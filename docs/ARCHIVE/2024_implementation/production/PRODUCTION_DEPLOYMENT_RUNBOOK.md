# Production Deployment Runbook

## Overview
This runbook provides step-by-step instructions for deploying the GPT-Trader perpetuals trading system to production.

## Environment Requirements

### Required Environment Variables
```bash
# Production Mode
export COINBASE_SANDBOX=0
export COINBASE_API_MODE=advanced
export COINBASE_ENABLE_DERIVATIVES=1

# CDP Credentials (Production)
export COINBASE_PROD_CDP_API_KEY="your-prod-api-key"
export COINBASE_PROD_CDP_PRIVATE_KEY="your-prod-private-key"

# Optional: Legacy fallback names
export COINBASE_CDP_API_KEY="your-api-key"
export COINBASE_CDP_PRIVATE_KEY="your-private-key"
```

### IP Whitelisting
Ensure your current IPv4/IPv6 addresses are whitelisted in Coinbase Advanced Trade settings.

## Deployment Phases

### Phase 0: Preflight Checks (Read-Only)
**Duration:** 5-10 minutes  
**Risk Level:** None (read-only)

#### Commands:
```bash
# 1. Run comprehensive preflight check
poetry run python scripts/prod_perps_preflight.py

# Expected output:
# ✅ JWT generated successfully
# ✅ accounts: XXms
# ✅ cfm_positions: XXms
# ✅ best_bid_ask: XXms
# ✅ server time: XXms
# ✅ WS user-channel subscribe attempted
# ✅ Public ticker stream available
# Overall: 🟢 PASS

# 2. Test WebSocket authentication (30 seconds)
poetry run python scripts/ws_auth_test.py --duration 30

# Expected: Connection established, user channel subscribed
```

#### Success Criteria:
- All REST endpoints respond < 500ms
- JWT generation successful
- WebSocket authentication works
- No errors in output

### Phase 1: Canary Test (Preview Mode)
**Duration:** 5 minutes  
**Risk Level:** None (preview only)

#### Commands:
```bash
# Preview reduce-only order (no actual placement)
poetry run python scripts/canary_reduce_only_test.py \
  --symbol BTC-PERP \
  --price 10 \
  --qty 0.001

# Expected: Order preview successful with details
```

#### Success Criteria:
- Order preview returns valid response
- Margin requirements calculated
- No errors or rejections

### Phase 2: Canary Monitoring (10-15 minutes)
**Duration:** 10-15 minutes  
**Risk Level:** Low (with guards)

#### Setup Canary Profile:
```bash
# Create canary profile in config/profiles/canary.yaml
cat > config/profiles/canary.yaml << EOF
name: canary
mode: production
symbols:
  - BTC-PERP
max_position_size: 0.001
max_orders_per_minute: 5
stop_loss_pct: 2.0
enable_reduce_only: true
EOF
```

#### Commands:
```bash
# 1. Start canary monitor (dry run first)
poetry run python scripts/canary_monitor.py \
  --profile canary \
  --duration-minutes 10 \
  --dry-run

# 2. If dry run successful, run live monitoring
poetry run python scripts/canary_monitor.py \
  --profile canary \
  --duration-minutes 15 \
  --dashboard

# Monitor will track:
# - Position sizes and PnL
# - Order flow and rate limits
# - API latencies
# - Guard violations
```

#### Guard Rails:
- Max position size: 0.01 BTC per symbol
- Max total exposure: 0.05 BTC
- Max drawdown: 2%
- Max orders/min: 10
- Min PnL threshold: -$50
- Max API latency: 500ms
- Heartbeat timeout: 30s

#### Kill Switch Triggers:
- 3+ guard violations
- Connection loss > 30s
- Drawdown > 2%
- Manual intervention (Ctrl+C)

### Phase 3: Gradual Scale-Up
**Duration:** 1-2 hours  
**Risk Level:** Medium

#### Progressive Scaling:
```bash
# Stage 1: Minimal size (30 min)
poetry run python scripts/run_perps_bot_v2.py \
  --profile stage1 \
  --duration-minutes 30

# Stage 2: Increased size (30 min)
poetry run python scripts/run_perps_bot_v2.py \
  --profile stage2 \
  --duration-minutes 30

# Stage 3: Target size (ongoing)
poetry run python scripts/run_perps_bot_v2.py \
  --profile production
```

## Monitoring & Alerts

### Real-Time Monitoring
```bash
# Dashboard (separate terminal)
poetry run python scripts/dashboard_server.py --port 8080

# Metrics exporter (Prometheus format)
poetry run python scripts/monitoring/metrics_exporter.py --port 9090

# Log aggregation
tail -f logs/perps_*.log | grep -E "ERROR|WARNING|VIOLATION"
```

### Alert Thresholds
| Metric | Warning | Critical | Action |
|--------|---------|----------|--------|
| PnL | -$25 | -$50 | Review positions |
| Drawdown | 1% | 2% | Reduce size/halt |
| API Latency | 300ms | 500ms | Check connection |
| Error Rate | 5/min | 10/min | Investigate logs |
| Position Size | 0.008 | 0.01 | Reduce exposure |

## Rollback Procedures

### Emergency Stop
```bash
# 1. Kill all running bots
pkill -f "run_perps_bot"

# 2. Cancel all open orders
poetry run python scripts/emergency_kill_switch.sh

# 3. Close all positions (manual)
# Use Coinbase Advanced Trade UI or:
poetry run python scripts/force_close_positions.py --confirm
```

### Gradual Wind-Down
```bash
# 1. Switch to reduce-only mode
export REDUCE_ONLY_MODE=1

# 2. Let positions close naturally
poetry run python scripts/wind_down_positions.py --timeout-minutes 30

# 3. Monitor closure
poetry run python scripts/monitor_position_closure.py
```

## Troubleshooting

### Common Issues

#### JWT Authentication Failures
```bash
# Verify credentials
poetry run python scripts/diagnose_cdp_key.py

# Check key permissions
# Ensure "trade" scope is enabled in Coinbase
```

#### High Latency
```bash
# Test connectivity
ping api.coinbase.com
traceroute api.coinbase.com

# Switch regions if needed
export API_REGION=us-east
```

#### Position Sync Issues
```bash
# Force position refresh
poetry run python scripts/sync_positions.py --force

# Validate against exchange
poetry run python scripts/validate_positions.py
```

## Post-Deployment Checklist

- [ ] All preflight checks pass
- [ ] Canary test successful (10-15 min)
- [ ] Monitoring dashboard active
- [ ] Alerts configured and tested
- [ ] Rollback procedures verified
- [ ] Team notified of deployment
- [ ] Initial positions within limits
- [ ] Logs being collected
- [ ] Metrics exported to monitoring
- [ ] Kill switch tested (dry run)

## Support Contacts

- **On-Call Engineer:** [Your contact]
- **Coinbase Support:** [Support ticket system]
- **Monitoring Dashboard:** http://localhost:8080
- **Logs Location:** `logs/perps_*.log`
- **Metrics Endpoint:** http://localhost:9090/metrics

## Appendix

### Sample Configurations

#### Stage 1 Profile (stage1.yaml)
```yaml
name: stage1
mode: production
symbols:
  - BTC-PERP
max_position_size: 0.0001
max_orders_per_minute: 2
stop_loss_pct: 1.0
take_profit_pct: 0.5
```

#### Stage 2 Profile (stage2.yaml)
```yaml
name: stage2
mode: production
symbols:
  - BTC-PERP
  - ETH-PERP
max_position_size: 0.001
max_orders_per_minute: 5
stop_loss_pct: 1.5
take_profit_pct: 1.0
```

#### Production Profile (production.yaml)
```yaml
name: production
mode: production
symbols:
  - BTC-PERP
  - ETH-PERP
  - SOL-PERP
max_position_size: 0.01
max_orders_per_minute: 10
stop_loss_pct: 2.0
take_profit_pct: 2.0
enable_dynamic_sizing: true
```

### Validation Scripts

All validation scripts are non-destructive and can be run at any time:

```bash
# Validate environment
poetry run python scripts/preflight_check.py

# Validate API connectivity
poetry run python scripts/validate_week1_implementation.py

# Validate order types
poetry run python scripts/validate_week3_orders.py

# Validate WebSocket
poetry run python scripts/validate_ws_week1.py
```

---

**Last Updated:** December 2024  
**Version:** 2.1.0  
**Status:** Production Ready