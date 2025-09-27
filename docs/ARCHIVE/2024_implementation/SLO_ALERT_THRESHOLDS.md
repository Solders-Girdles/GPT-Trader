# Service Level Objectives & Alert Thresholds

## Performance SLOs

### API Latency
| Metric | Target | Warning | Critical | Action |
|--------|--------|---------|----------|--------|
| p50 Latency | < 100ms | > 150ms | > 200ms | Monitor closely |
| p95 Latency | < 300ms | > 400ms | > 500ms | Scale down activity |
| p99 Latency | < 500ms | > 750ms | > 1000ms | Investigate root cause |

### Order Success Rates
| Metric | Target | Warning | Critical | Action |
|--------|--------|---------|----------|--------|
| Acceptance Rate | > 98% | < 95% | < 90% | Review rejection reasons |
| Cancel Success | > 99% | < 97% | < 95% | Check order states |
| Fill Rate (Limit) | > 60% | < 40% | < 20% | Adjust pricing logic |

### WebSocket Stability
| Metric | Target | Warning | Critical | Action |
|--------|--------|---------|----------|--------|
| Uptime | > 99.9% | < 99.5% | < 99% | Check connectivity |
| Reconnects/Hour | < 1 | > 3 | > 10 | Investigate network |
| Message Lag | < 50ms | > 100ms | > 200ms | Reduce subscriptions |
| Staleness Events | < 1/hr | > 5/hr | > 10/hr | Increase threshold |

## Trading SLOs

### Position Management
| Metric | Target | Warning | Critical | Action |
|--------|--------|---------|----------|--------|
| Position Sync Accuracy | 100% | < 99.9% | < 99% | Reconcile immediately |
| Max Position Size | < 1% | > 1.5% | > 2% | Reduce exposure |
| Margin Utilization | < 30% | > 50% | > 70% | Scale down |

### Risk Metrics
| Metric | Target | Warning | Critical | Action |
|--------|--------|---------|----------|--------|
| Daily Loss | < 1% | > 1.5% | > 2% | Trigger stop loss |
| Max Drawdown | < 3% | > 5% | > 7% | Pause trading |
| Sharpe Ratio (30d) | > 1.5 | < 1.0 | < 0.5 | Review strategy |
| Win Rate | > 55% | < 50% | < 45% | Analyze losses |

### Market Impact
| Metric | Target | Warning | Critical | Action |
|--------|--------|---------|----------|--------|
| Spread Cross (bps) | < 5 | > 10 | > 15 | Widen limits |
| Slippage (bps) | < 3 | > 5 | > 10 | Reduce size |
| Market Impact (bps) | < 10 | > 15 | > 20 | Split orders |

## Operational SLOs

### System Health
| Metric | Target | Warning | Critical | Action |
|--------|--------|---------|----------|--------|
| CPU Usage | < 60% | > 70% | > 85% | Scale resources |
| Memory Usage | < 70% | > 80% | > 90% | Restart services |
| Disk Usage | < 70% | > 80% | > 90% | Clean logs |
| Error Rate | < 0.1% | > 0.5% | > 1% | Debug errors |

### Data Quality
| Metric | Target | Warning | Critical | Action |
|--------|--------|---------|----------|--------|
| Price Staleness | < 1s | > 2s | > 5s | Check feed |
| Missing Data Points | < 0.1% | > 0.5% | > 1% | Investigate gaps |
| Outlier Detection | < 0.01% | > 0.1% | > 0.5% | Filter anomalies |

## Alert Configuration

### PagerDuty Alerts (Critical)
```yaml
alerts:
  - name: kill_switch_triggered
    condition: kill_switch == activated
    severity: P1
    escalation: immediate
    
  - name: daily_loss_breach
    condition: daily_pnl < -0.02
    severity: P1
    escalation: immediate
    
  - name: position_limit_breach
    condition: position_size > max_allowed
    severity: P1
    escalation: immediate
    
  - name: api_auth_failure
    condition: auth_failures > 5 in 1min
    severity: P1
    escalation: immediate
```

### Slack Alerts (Warning)
```yaml
alerts:
  - name: high_rejection_rate
    condition: rejection_rate > 0.05
    channel: "#trading-alerts"
    
  - name: websocket_reconnects
    condition: reconnects > 3 in 1hr
    channel: "#trading-alerts"
    
  - name: funding_rate_spike
    condition: abs(funding_rate) > 0.001
    channel: "#trading-alerts"
    
  - name: spread_widening
    condition: spread_bps > 20
    channel: "#market-conditions"
```

### Email Alerts (Info)
```yaml
alerts:
  - name: daily_summary
    schedule: "0 17 * * *"
    recipients: ["trading-team@company.com"]
    
  - name: weekly_performance
    schedule: "0 9 * * 1"
    recipients: ["management@company.com"]
    
  - name: monthly_report
    schedule: "0 9 1 * *"
    recipients: ["stakeholders@company.com"]
```

## Rejection Reason Tracking

### Expected Rejections (Acceptable)
- `POST_ONLY_WOULD_CROSS`: Price too aggressive
- `INSUFFICIENT_LIQUIDITY`: Market too thin
- `RSI_FILTER_BLOCKED`: Momentum check failed
- `VOLATILITY_FILTER_BLOCKED`: Vol too high
- `SIZED_DOWN`: Position reduced to limit

### Unexpected Rejections (Investigate)
- `INSUFFICIENT_FUNDS`: Margin calculation error
- `INVALID_PRODUCT_ID`: Symbol mapping issue
- `RATE_LIMIT_EXCEEDED`: Too many requests
- `INVALID_ORDER_SIZE`: Quantization error
- `UNAUTHORIZED`: Auth token issue

## Performance Benchmarks

### Latency Targets
```
Order Placement:
- Market Order: < 50ms
- Limit Order: < 75ms
- Cancel Order: < 50ms
- Modify Order: < 100ms

Data Processing:
- Tick to Signal: < 10ms
- Signal to Order: < 20ms
- Order to Confirm: < 100ms
```

### Throughput Targets
```
Orders:
- Place: 100/second capability, 10/second typical
- Cancel: 100/second capability, 5/second typical
- Modify: 50/second capability, 2/second typical

Market Data:
- Ticks: 1000/second processing
- Books: 100/second updates
- Trades: 500/second streaming
```

## Monitoring Dashboard Queries

### Key Metrics SQL
```sql
-- Acceptance Rate (Last Hour)
SELECT 
  COUNT(CASE WHEN status = 'accepted' THEN 1 END) * 100.0 / COUNT(*) as acceptance_rate,
  COUNT(CASE WHEN status = 'rejected' THEN 1 END) as rejected_count,
  COUNT(*) as total_orders
FROM orders
WHERE created_at > NOW() - INTERVAL '1 hour';

-- Rejection Breakdown
SELECT 
  rejection_reason,
  COUNT(*) as count,
  COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () as percentage
FROM orders
WHERE status = 'rejected'
  AND created_at > NOW() - INTERVAL '1 hour'
GROUP BY rejection_reason
ORDER BY count DESC;

-- Position & PnL
SELECT 
  symbol,
  position_size,
  unrealized_pnl,
  realized_pnl,
  total_pnl,
  funding_paid
FROM positions
WHERE is_active = true;
```

## Escalation Matrix

| Severity | Response Time | Owner | Escalation |
|----------|--------------|--------|------------|
| P1 (Critical) | < 15 min | On-call Engineer | CTO |
| P2 (High) | < 1 hour | Trading Team | Engineering Lead |
| P3 (Medium) | < 4 hours | DevOps | Trading Team |
| P4 (Low) | < 24 hours | Support | DevOps |

## Review Schedule

- **Daily**: Review rejection reasons, latency p95, PnL
- **Weekly**: Analyze performance trends, win rate, Sharpe
- **Monthly**: SLO compliance report, threshold adjustments
- **Quarterly**: Full system performance review

---

*Version: 1.0*
*Last Updated: 2025-08-30*
*Next Review: 2025-09-15*