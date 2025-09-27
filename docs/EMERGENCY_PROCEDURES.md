# Emergency Procedures & Production Runbook

## 🚨 EMERGENCY CONTACTS

**Internal**
- Primary On-Call: ________________
- Backup On-Call: _________________
- Team Lead: _____________________

**External**
- Coinbase Support: https://help.coinbase.com/
- Status Page: https://status.coinbase.com/

## 🔴 IMMEDIATE ACTIONS

### 1. EMERGENCY KILL SWITCH
```bash
# STOP ALL TRADING IMMEDIATELY
export RISK_KILL_SWITCH_ENABLED=1
poetry run perps-bot --kill-switch

# Or directly in running process:
# Press Ctrl+C twice rapidly
```

### 2. CLOSE ALL POSITIONS
```bash
# Enable reduce-only mode to close positions
export RISK_REDUCE_ONLY_MODE=1
poetry run perps-bot --profile canary --reduce-only

# Manual position closure via API:
poetry run python scripts/emergency_close_positions.py
```

### 3. CHECK SYSTEM STATUS
```bash
# View current positions and P&L
poetry run python scripts/check_positions.py

# Check recent orders
poetry run python scripts/check_recent_orders.py

# System health check
poetry run python scripts/health_check.py
```

## 📊 MONITORING & ALERTS

### Real-Time Monitoring Commands
```bash
# Monitor live P&L
watch -n 5 'poetry run python scripts/show_pnl.py'

# Monitor positions
watch -n 10 'poetry run python scripts/show_positions.py'

# Monitor system metrics
poetry run python scripts/monitor_metrics.py

# Check error logs
tail -f logs/perps_bot.log | grep ERROR
```

### Key Metrics to Monitor
- **P&L**: Current session and daily P&L
- **Leverage**: Current vs maximum allowed
- **Liquidation Distance**: Buffer from liquidation price
- **Error Rate**: Order failures and API errors
- **Latency**: API response times
- **Mark Staleness**: Age of market data

## 🔧 COMMON ISSUES & SOLUTIONS

### Issue: Excessive Losses
**Symptoms**: Daily loss approaching or exceeding limit
**Actions**:
1. Check if daily loss limit is triggered automatically
2. Review recent trades: `poetry run python scripts/analyze_trades.py --today`
3. Reduce position sizes or halt trading
4. Review strategy parameters

### Issue: API Authentication Failures
**Symptoms**: 401/403 errors, "unauthorized" messages
**Actions**:
1. Verify credentials: `poetry run python scripts/testing/prod_perps_preflight.py`
2. Check API key permissions in Coinbase portal
3. Regenerate JWT token
4. Verify system time sync: `timedatectl status`

### Issue: Connection/Network Problems
**Symptoms**: Timeouts, connection refused, WebSocket disconnects
**Actions**:
1. Check network connectivity: `ping api.coinbase.com`
2. Check Coinbase status: https://status.coinbase.com/
3. Restart with exponential backoff
4. Switch to REST-only mode: `export PERPS_ENABLE_STREAMING=0`

### Issue: Stale Market Data
**Symptoms**: Mark prices not updating, staleness warnings
**Actions**:
1. Check WebSocket connection status
2. Force reconnect: Restart the bot
3. Use REST fallback: `export COINBASE_USE_REST_FALLBACK=1`
4. Check if market is open/active

### Issue: Order Rejections
**Symptoms**: Orders consistently rejected
**Actions**:
1. Check order size limits and increments
2. Verify account balance and margin
3. Check if in reduce-only mode
4. Review order parameters: `poetry run python scripts/debug_order.py`

### Issue: High Leverage Warning
**Symptoms**: Leverage approaching maximum
**Actions**:
1. Reduce position size immediately
2. Check liquidation distance
3. Add margin if needed
4. Enable conservative mode

## 📋 STARTUP PROCEDURES

### Pre-Production Checklist
```bash
# 1. Run comprehensive preflight
poetry run python scripts/production_preflight.py --profile canary

# 2. Test in dry-run mode
poetry run perps-bot --profile canary --dry-run --run-once

# 3. Verify risk limits
cat .env | grep RISK_

# 4. Check account status
poetry run python scripts/check_account_status.py
```

### Production Startup Sequence
```bash
# 1. Set production environment
export BROKER=coinbase
export COINBASE_API_MODE=advanced
export COINBASE_SANDBOX=0
export COINBASE_ENABLE_DERIVATIVES=1

# 2. Start with canary profile (minimal risk)
poetry run perps-bot --profile canary

# 3. Monitor for first hour
# - Check first 10 trades manually
# - Verify P&L calculations
# - Monitor error rate

# 4. Scale up gradually
# Day 1-3: Canary profile (0.01 BTC max)
# Day 4-7: Increase to 0.02 BTC if stable
# Week 2+: Consider production profile
```

## 🔄 RECOVERY PROCEDURES

### After Emergency Stop
1. **Assess the situation**
   - What triggered the emergency?
   - Current positions and P&L
   - Market conditions

2. **Document the incident**
   ```bash
   poetry run python scripts/generate_incident_report.py
   ```

3. **Close or manage positions**
   - Decide whether to close immediately or manage down
   - Consider market impact and slippage

4. **Root cause analysis**
   - Review logs: `logs/perps_bot_YYYYMMDD.log`
   - Check event store: `events/`
   - Analyze trade history

5. **Implement fixes**
   - Update configuration if needed
   - Fix any code issues
   - Update risk parameters

6. **Gradual restart**
   - Start with paper trading
   - Move to canary profile
   - Monitor closely for 24 hours

### After Network Outage
1. Check system time sync
2. Verify all connections restored
3. Check for missed fills
4. Reconcile positions
5. Resume with reduce-only initially

### After Large Loss
1. Stop trading immediately
2. Document all trades
3. Review risk parameters
4. Reduce position sizes
5. Consider paper trading
6. Implement stricter limits

## 📝 LOGGING & DEBUGGING

### Log Locations
- Main log: `logs/perps_bot.log`
- Error log: `logs/errors.log`
- Trade log: `logs/trades.log`
- Event store: `events/`

### Debug Commands
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Trace specific order
poetry run python scripts/trace_order.py --order-id <ID>

# Analyze P&L discrepancy
poetry run python scripts/analyze_pnl.py --date <YYYY-MM-DD>

# Debug WebSocket issues
poetry run python scripts/debug_websocket.py

# Test specific strategy
poetry run python scripts/test_strategy.py --strategy momentum
```

## 🔍 PERFORMANCE ANALYSIS

### Daily Review Checklist
- [ ] Total P&L vs expectations
- [ ] Win rate and average trade size
- [ ] Maximum drawdown
- [ ] Error rate and types
- [ ] Slippage analysis
- [ ] Risk limit usage

### Weekly Review
```bash
# Generate weekly report
poetry run python scripts/generate_weekly_report.py

# Analyze strategy performance
poetry run python scripts/analyze_strategy_performance.py

# Review risk metrics
poetry run python scripts/analyze_risk_metrics.py
```

## 🚀 SCALING PROCEDURES

### Increasing Position Sizes
**Prerequisites**:
- Minimum 5 days stable operation
- Error rate <5%
- Consistent positive P&L
- All risk checks passing

**Steps**:
1. Increase gradually (max 2x at a time)
2. Update `.env` risk parameters
3. Run preflight check
4. Monitor for 24 hours
5. Document changes

### Adding New Symbols
**Steps**:
1. Research symbol liquidity
2. Set symbol-specific limits
3. Test in paper mode first
4. Start with minimum size
5. Monitor correlation risk

## 🔐 SECURITY PROCEDURES

### Key Rotation
```bash
# 1. Generate new API key in Coinbase
# 2. Update .env with new credentials
# 3. Test authentication
poetry run python scripts/test_new_credentials.py
# 4. Deploy new credentials
# 5. Revoke old key after confirming
```

### Security Audit
- Review API permissions monthly
- Check for exposed credentials
- Monitor for unusual activity
- Enable 2FA on all accounts
- Use hardware keys if possible

## 📞 ESCALATION PROCEDURES

### Level 1: Warning
- Automated alerts triggered
- Monitor closely
- No immediate action required

### Level 2: Intervention Required
- Manual review needed
- Reduce risk parameters
- Consider reducing positions

### Level 3: Critical
- Immediate action required
- Activate kill switch if needed
- Close all positions
- Contact team lead

### Level 4: Emergency
- System compromise suspected
- Revoke all API keys immediately
- Close all positions manually
- Contact Coinbase support
- Document everything

## 🔄 MAINTENANCE WINDOWS

### Planned Maintenance
```bash
# 1. Announce maintenance 24h in advance
# 2. Close or reduce positions
# 3. Enable reduce-only mode
# 4. Stop bot gracefully
poetry run perps-bot --shutdown-graceful

# 5. Perform maintenance
# 6. Run tests
# 7. Restart with canary profile
```

### Coinbase Maintenance
- Monitor https://status.coinbase.com/
- Set reduce-only mode before maintenance
- Avoid trading during maintenance windows
- Verify connectivity after maintenance

## 📚 QUICK REFERENCE

### Environment Variables
```bash
# Critical safety controls
RISK_KILL_SWITCH_ENABLED=1          # Stop all trading
RISK_REDUCE_ONLY_MODE=1             # Only close positions
RISK_DAILY_LOSS_LIMIT=100           # Max daily loss (USD)
RISK_MAX_LEVERAGE=3                 # Maximum leverage
PERPS_PAPER=1                        # Paper trading mode
```

### Common Commands
```bash
# Start trading
poetry run perps-bot --profile canary

# Dry run test
poetry run perps-bot --profile canary --dry-run

# Single cycle test
poetry run perps-bot --profile dev --run-once

# Check status
poetry run python scripts/bot_status.py

# Emergency stop
poetry run perps-bot --kill-switch
```

### Profile Summary
- **dev**: Mock broker, fast cycles, extensive logging
- **canary**: 0.01 BTC max, $10 daily loss, reduce-only
- **prod**: Full sizing, production limits

## ⚠️ GOLDEN RULES

1. **Never disable safety features in production**
2. **Always test changes in paper mode first**
3. **Start small and scale gradually**
4. **Monitor closely for first 24 hours after any change**
5. **Document all incidents and changes**
6. **Keep emergency contacts updated**
7. **Maintain adequate liquidation buffers**
8. **Never trade with funds you cannot afford to lose**
9. **Have a plan for every scenario**
10. **When in doubt, reduce risk or stop trading**

---

**Last Updated**: December 2024
**Version**: 1.0.0
**Next Review**: Monthly or after any major incident