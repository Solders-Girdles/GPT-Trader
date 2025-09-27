# Verification & Validation Guide

---
status: current
last-updated: 2025-01-01
consolidates:
  - PRODUCTION_VERIFICATION_ARTIFACTS.md
  - PRODUCTION_VERIFICATION_COMPLETE.md
  - validation_runbook.md
---

## Overview

Complete verification procedures for GPT-Trader. The bot now runs spot-first; enable perpetuals only after INTX approval. Historical perps-specific steps are retained below for future activation and should be adapted as needed for spot-only deployments.

## Required Artifacts

### Pre-Deployment
- [ ] Test suite results (100% pass on active code)
- [ ] Performance benchmarks
- [ ] Security audit report
- [ ] API connectivity tests
- [ ] WebSocket stability tests

### Post-Deployment
- [ ] Live connection verification
- [ ] Order placement test (tiny size)
- [ ] Position tracking validation
- [ ] PnL calculation accuracy
- [ ] Risk limits enforcement

## Validation Procedures

### 1. Core Functionality

```bash
# Run test suite
poetry run pytest tests/unit/bot_v2 tests/unit/test_foundation.py -q
# EXPECTED: Selected spot suites passing (see `poetry run pytest --collect-only` → 422 selected)

# Run integration tests (explicit marker)
poetry run pytest -m integration tests/integration/bot_v2 -q
# EXPECTED: All critical paths passing
```

### 2. API Connectivity

```bash
# Test CDP authentication
python scripts/validate_week1_implementation.py
# EXPECTED: Authentication successful

# Test order placement
python scripts/test_gtd_single_order.py
# EXPECTED: Order placed and filled

# Test WebSocket streams
python scripts/validate_ws_week1.py
# EXPECTED: Streaming data received
```

### 3. Risk Management

```bash
# Test circuit breakers
python scripts/test_guard_triggers.py --test all
# EXPECTED: All guards trigger correctly

# Test reduce-only mode
python scripts/test_violation_trigger.py
# EXPECTED: Reduce-only enforced

# Test position limits
python scripts/validate_week3_orders.py
# EXPECTED: Limits enforced
```

### 4. Performance Validation

```bash
# Run performance benchmarks
python scripts/validation/validate_perps_e2e.py
# EXPECTED: <500ms order latency

# Test under load
python scripts/stage3_runner.py --duration-minutes 10
# EXPECTED: Stable operation
```

## Verification Checkpoints

### Day 1 - Initial Deployment
- [ ] Bot starts without errors
- [ ] WebSocket connects and stays connected
- [ ] Market data streaming works
- [ ] Canary profile limits enforced

### Day 3 - Canary Phase
- [ ] No unexpected losses
- [ ] All trades logged correctly
- [ ] Risk limits never breached
- [ ] Clean error logs

### Day 7 - Pre-Production
- [ ] Consistent profitability or controlled losses
- [ ] No memory leaks
- [ ] Stable WebSocket connection
- [ ] Accurate PnL tracking

### Day 14 - Full Production
- [ ] All strategies performing as expected
- [ ] Risk metrics within bounds
- [ ] System stability confirmed
- [ ] Monitoring alerts working

## Validation Scripts

### Quick Health Check
```bash
#!/bin/bash
# health_check.sh

echo "=== System Health Check ==="

# Check bot process
pgrep -f perps-bot && echo "✓ Bot running" || echo "✗ Bot not running"

# Check WebSocket
python -c "from scripts.ws_probe import test_ws; test_ws()" && echo "✓ WebSocket OK"

# Check recent errors
errors=$(tail -n 1000 logs/perps_bot.log | grep -c ERROR)
echo "Errors in last 1000 lines: $errors"

# Check positions
python -c "from bot_v2.cli import check_positions; check_positions()"
```

### Daily Validation
```bash
#!/bin/bash
# daily_validation.sh

echo "=== Daily Validation Report ==="
date

# PnL check
python scripts/calculate_daily_pnl.py

# Risk metrics
python scripts/check_risk_metrics.py

# Error analysis
python scripts/analyze_errors.py --last 24h

# Performance metrics
python scripts/performance_report.py --period daily
```

## Troubleshooting Validation Failures

### Test Suite Failures
1. Check Python version (3.12+ required)
2. Verify all dependencies: `poetry install`
3. Clear cache: `rm -rf .pytest_cache`
4. Run with verbose: `pytest -vvs`

### API Connection Failures
1. Verify environment variables set
2. Check network connectivity
3. Validate API key permissions
4. Test with curl directly

### WebSocket Failures
1. Check firewall/proxy settings
2. Verify WebSocket URL correct
3. Test with simple WebSocket client
4. Review connection logs

## Success Criteria

### Minimum Requirements
- ✅ All core tests passing
- ✅ API authentication working
- ✅ WebSocket stable for 1 hour+
- ✅ Risk limits enforced
- ✅ No critical errors in 24 hours

### Optimal Performance
- ✅ 99.9% uptime
- ✅ <1% daily drawdown
- ✅ <100ms average latency
- ✅ Zero unauthorized trades
- ✅ Complete audit trail

## Reporting

Generate verification report:
```bash
cat > verification_report_$(date +%Y%m%d).md << EOF
# Production Verification Report
Date: $(date)

## Test Results
- Unit Tests: $(pytest tests/unit/bot_v2 --tb=no -q | tail -1)
- Integration: $(pytest -m integration tests/integration/bot_v2 --tb=no -q | tail -1)

## API Status
- Authentication: $(python scripts/check_auth.py)
- WebSocket: $(python scripts/check_ws.py)

## Risk Validation
- Limits Enforced: YES/NO
- Reduce-Only Works: YES/NO

## Performance
- Order Latency: XXXms
- Memory Usage: XXXMB

## Recommendation
[ ] Ready for production
[ ] Requires fixes
EOF
```
