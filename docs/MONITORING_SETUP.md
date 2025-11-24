# GPT Trader - Monitoring & Operations Setup

## Overview

This document describes the comprehensive monitoring and operational tools for the GPT Trader system. You'll have visibility into performance metrics, health status, and trading activity from Day 1.

## Table of Contents

1. [Daily Performance Reports](#daily-performance-reports)
2. [Prometheus Metrics](#prometheus-metrics)
3. [Grafana Dashboards](#grafana-dashboards)
4. [Health Checks](#health-checks)
5. [Alerting](#alerting)
6. [Cron Job Setup](#cron-job-setup)

---

## Daily Performance Reports

### Overview

Daily reports provide a comprehensive snapshot of your trading performance, including:
- **Account Metrics**: Equity, 24h change
- **PnL Breakdown**: Realized, unrealized, funding PnL
- **Performance**: Win rate, Profit Factor, Sharpe ratio, Max drawdown
- **Symbol Analysis**: Top performers with regime-based PnL table
- **Risk Controls**: Guard triggers, circuit breaker state
- **Health Metrics**: Stale marks, WebSocket reconnects, unfilled orders

### Running Reports

#### Command Line

Generate a daily report:

```bash
# Generate today's report (text format)
python -m gpt_trader.cli report daily

# Generate report for specific date
python -m gpt_trader.cli report daily --date 2025-10-22

# Generate JSON report
python -m gpt_trader.cli report daily --format json

# Generate both formats
python -m gpt_trader.cli report daily --format both

# Custom lookback period (default: 24 hours)
python -m gpt_trader.cli report daily --lookback-hours 48

# Specify profile
python -m gpt_trader.cli report daily --profile prod

# Print to stdout without saving
python -m gpt_trader.cli report daily --no-save
```

#### Report Output

Reports are saved to:
- Text: `var/data/coinbase_trader/{profile}/reports/daily_report_{date}.txt`
- JSON: `var/data/coinbase_trader/{profile}/reports/daily_report_{date}.json`

#### Example Text Report

```
================================================================================
Daily Trading Report - 2025-10-22
Profile: prod
Generated: 2025-10-22T14:30:00
================================================================================

ACCOUNT SUMMARY
--------------------------------------------------------------------------------
  Equity:          $50,234.56
  Change (24h):    +$1,234.56 (+2.52%)

PNL BREAKDOWN
--------------------------------------------------------------------------------
  Realized PnL:    +$1,100.00
  Unrealized PnL:  +$200.00
  Funding PnL:     -$65.44
  Fees Paid:       $120.00
  Total PnL:       +$1,234.56

PERFORMANCE METRICS
--------------------------------------------------------------------------------
  Win Rate:        65.00%
  Profit Factor:   2.34
  Sharpe Ratio:    1.87
  Max Drawdown:    $450.00 (0.92%)

TRADE STATISTICS
--------------------------------------------------------------------------------
  Total Trades:    20
  Winning:         13
  Losing:          7
  Avg Win:         $150.00
  Avg Loss:        $90.00
  Largest Win:     $450.00
  Largest Loss:    $210.00

TOP PERFORMERS BY SYMBOL
--------------------------------------------------------------------------------
Symbol       Regime     Total PnL    Realized    Unreal      Funding   Trades   Win%
--------------------------------------------------------------------------------
BTC-USD      trending   $780.50      $750.00     $50.00     -$19.50        8   75.0%
ETH-USD      ranging    $320.25      $300.00     $30.00     -$9.75         6   66.7%
SOL-USD      volatile   $134.31      $100.00     $50.00     -$15.69        4   50.0%

RISK CONTROLS
--------------------------------------------------------------------------------
  Guard Triggers:
    slippage_guard: 3
    exposure_guard: 1
  Circuit Breaker State:
    All systems normal

HEALTH METRICS
--------------------------------------------------------------------------------
  Stale Marks:     0
  WS Reconnects:   2
  Unfilled Orders: 0
  API Errors:      1

================================================================================
```

---

## Prometheus Metrics

### Overview

The system exports comprehensive metrics in Prometheus format for time-series monitoring and alerting.

### Starting the Metrics Exporter

```bash
# Default (prod profile, port 9000)
python scripts/monitoring/export_metrics.py

# Custom profile
python scripts/monitoring/export_metrics.py --metrics-file var/data/coinbase_trader/demo/metrics.json

# Custom port
python scripts/monitoring/export_metrics.py --port 9001
```

The exporter provides two endpoints:
- `http://localhost:9000/metrics` - Prometheus format
- `http://localhost:9000/metrics.json` - JSON format

### Available Metrics

#### Account Metrics
- `coinbase_trader_equity` - Account equity in USD
- `coinbase_trader_total_pnl_usd` - Total PnL
- `coinbase_trader_realized_pnl_usd` - Realized PnL
- `coinbase_trader_unrealized_pnl_usd` - Unrealized PnL
- `coinbase_trader_funding_pnl_usd` - Funding PnL

#### Position Metrics
- `coinbase_trader_open_orders` - Number of open orders
- `coinbase_trader_symbol_exposure_usd{symbol="BTC_USD"}` - Per-symbol exposure
- `coinbase_trader_symbol_pnl_usd{symbol="BTC_USD"}` - Per-symbol PnL

#### Performance Metrics
- `coinbase_trader_cycle_latency_ms` - Trading cycle latency
- `coinbase_trader_circuit_breaker_triggered` - Circuit breaker state (0=normal, 1=triggered)

#### Health Metrics
- `coinbase_trader_websocket_reconnects_total` - Total WebSocket reconnections
- `coinbase_trader_stale_marks_total` - Total stale mark detections
- `coinbase_trader_unfilled_orders_total` - Total unfilled order alerts
- `coinbase_trader_order_preview_failures_total` - Order preview failures

#### System Metrics
- `coinbase_trader_cpu_percent` - Process CPU usage
- `coinbase_trader_memory_percent` - Process memory usage
- `coinbase_trader_disk_percent` - Disk usage
- `coinbase_trader_network_sent_megabytes` - Network sent
- `coinbase_trader_network_received_megabytes` - Network received

### Prometheus Configuration

Add to your `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'gpt-trader'
    scrape_interval: 5s
    static_configs:
      - targets: ['localhost:9000']
```

---

## Grafana Dashboards

### Importing the Dashboard

1. Copy the dashboard JSON from `monitoring/grafana_dashboard.json`
2. In Grafana, click **+ → Import**
3. Paste the JSON content
4. Select your Prometheus data source
5. Click **Import**

### Dashboard Panels

The dashboard includes:

1. **Account Equity** - Time series of account value
2. **PnL Breakdown** - Realized, unrealized, funding, total PnL
3. **Per-Symbol Exposure** - Stacked area chart of position exposure
4. **Per-Symbol PnL** - Individual symbol performance
5. **Circuit Breaker State** - Alert status (green=normal, red=triggered)
6. **Cycle Latency** - Trading loop performance with thresholds
7. **Open Orders** - Current open order count
8. **Unfilled Orders Alert** - Recent unfilled order warnings
9. **WebSocket Reconnects** - Connection stability
10. **Stale Marks Detected** - Market data freshness
11. **System Resources** - CPU and memory usage
12. **Order Preview Failures** - Order validation issues

### Alert Thresholds

The dashboard includes visual thresholds:
- **Cycle Latency**: Warning at 500ms, Critical at 1000ms
- **Open Orders**: Warning at 10, Critical at 20
- **WS Reconnects**: Warning at 5/hour, Critical at 10/hour

---

## Health Checks

### Overview

The system includes several health checks that monitor critical subsystems.

### Available Health Checks

#### 1. Stale Fills Health Check

Detects orders that remain unfilled for too long.

```python
from gpt_trader.monitoring.health.checks import StaleFillsHealthCheck

checker = StaleFillsHealthCheck(
    orders_store=orders_store,
    max_age_minutes=10.0  # Alert if order unfilled for >10 minutes
)
result = await checker.check()
```

**Status Conditions:**
- `HEALTHY`: All open orders are recent
- `DEGRADED`: 1-4 orders unfilled for >10 minutes
- `UNHEALTHY`: 5+ orders unfilled for >10 minutes

#### 2. Stale Marks Health Check

Monitors market data freshness.

```python
from gpt_trader.monitoring.health.checks import StaleMarksHealthCheck

checker = StaleMarksHealthCheck(
    market_data_service=market_data,
    max_age_seconds=30.0  # Alert if marks >30s old
)
result = await checker.check()
```

**Status Conditions:**
- `HEALTHY`: All marks are fresh
- `DEGRADED`: Some marks are stale
- `UNHEALTHY`: >50% of marks are stale

#### 3. WebSocket Reconnect Health Check

Detects connection instability and reconnect loops.

```python
from gpt_trader.monitoring.health.checks import WebSocketReconnectHealthCheck

checker = WebSocketReconnectHealthCheck(
    websocket_handler=ws_handler,
    max_reconnects_per_hour=10,
    reconnect_loop_threshold=5  # 5 reconnects in 1 minute = loop
)
result = await checker.check()
```

**Status Conditions:**
- `HEALTHY`: Stable connection, <10 reconnects/hour
- `DEGRADED`: High reconnect rate or disconnected
- `UNHEALTHY`: Reconnect loop detected (5+ in 1 minute)

### Integrating Health Checks

Add health checks to your monitoring system:

```python
from gpt_trader.monitoring.health.registry import HealthCheckRegistry

registry = HealthCheckRegistry()

# Register health checks
registry.register(StaleFillsHealthCheck(orders_store))
registry.register(StaleMarksHealthCheck(market_data))
registry.register(WebSocketReconnectHealthCheck(ws_handler))

# Run all checks
results = await registry.check_all()

for result in results:
    print(f"{result.name}: {result.status} - {result.message}")
```

---

## Alerting

### Prometheus Alert Rules

Add to `monitoring/alert_rules.yml`:

```yaml
groups:
  - name: gpt_trader_alerts
    interval: 30s
    rules:
      # Circuit breaker triggered
      - alert: CircuitBreakerTriggered
        expr: coinbase_trader_circuit_breaker_triggered > 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Circuit breaker has been triggered"
          description: "Trading has been halted due to circuit breaker"

      # High cycle latency
      - alert: HighCycleLatency
        expr: coinbase_trader_cycle_latency_ms > 1000
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Trading cycle latency is high"
          description: "Cycle latency is {{ $value }}ms (threshold: 1000ms)"

      # Stale marks detected
      - alert: StaleMarketData
        expr: increase(coinbase_trader_stale_marks_total[5m]) > 10
        labels:
          severity: warning
        annotations:
          summary: "Frequent stale market data"
          description: "{{ $value }} stale marks detected in last 5 minutes"

      # WebSocket reconnect loop
      - alert: WebSocketReconnectLoop
        expr: increase(coinbase_trader_websocket_reconnects_total[1m]) >= 5
        labels:
          severity: critical
        annotations:
          summary: "WebSocket reconnect loop detected"
          description: "{{ $value }} reconnects in last minute"

      # Unfilled orders
      - alert: UnfilledOrders
        expr: increase(coinbase_trader_unfilled_orders_total[10m]) > 5
        labels:
          severity: warning
        annotations:
          summary: "Multiple unfilled orders detected"
          description: "{{ $value }} orders remain unfilled for >10 minutes"

      # Equity drop
      - alert: SignificantEquityDrop
        expr: (coinbase_trader_equity - coinbase_trader_equity offset 1h) / coinbase_trader_equity < -0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Significant equity drop detected"
          description: "Equity has dropped >5% in the last hour"
```

### Event Logging

The system automatically logs events to `events.jsonl` that trigger metrics:

- `circuit_breaker_triggered` - Circuit breaker activation
- `stale_mark_detected` - Stale market data
- `websocket_reconnect` - WebSocket reconnection
- `unfilled_order_alert` - Orders unfilled too long

These events are counted by the metrics exporter.

---

## Cron Job Setup

### Daily Report Email

Set up a cron job to generate and email daily reports:

```bash
# Edit crontab
crontab -e

# Add daily report job (runs at 6 AM every day)
0 6 * * * cd /path/to/GPT-Trader && python -m gpt_trader.cli report daily --profile prod && cat var/data/coinbase_trader/prod/reports/daily_report_$(date +\%Y-\%m-\%d).txt | mail -s "GPT Trader Daily Report" your-email@example.com
```

### Alternative: Script with Email

Create `scripts/send_daily_report.sh`:

```bash
#!/bin/bash
set -e

PROFILE="${1:-prod}"
DATE=$(date +%Y-%m-%d)

cd /path/to/GPT-Trader

# Generate report
python -m gpt_trader.cli report daily --profile "$PROFILE" --format both

# Send email with text report
REPORT_FILE="var/data/coinbase_trader/$PROFILE/reports/daily_report_$DATE.txt"
if [ -f "$REPORT_FILE" ]; then
    mail -s "GPT Trader Daily Report - $DATE" \
         -a "var/data/coinbase_trader/$PROFILE/reports/daily_report_$DATE.json" \
         your-email@example.com < "$REPORT_FILE"
fi
```

Make executable and add to cron:

```bash
chmod +x scripts/send_daily_report.sh

# Add to crontab
0 6 * * * /path/to/GPT-Trader/scripts/send_daily_report.sh prod
```

---

## Quick Start Checklist

- [ ] Start Prometheus metrics exporter: `python scripts/monitoring/export_metrics.py`
- [ ] Configure Prometheus to scrape metrics (see above)
- [ ] Import Grafana dashboard from `monitoring/grafana_dashboard.json`
- [ ] Set up Prometheus alert rules in `monitoring/alert_rules.yml`
- [ ] Test daily report generation: `python -m gpt_trader.cli report daily --no-save`
- [ ] Set up cron job for daily reports
- [ ] Configure AlertManager for email/Slack notifications

---

## Troubleshooting

### No metrics showing in Prometheus

1. Check metrics exporter is running: `curl http://localhost:9000/metrics`
2. Verify Prometheus scrape config includes correct target
3. Check Prometheus targets page: `http://localhost:9090/targets`

### Daily report has no data

1. Verify events file exists: `ls -la var/data/coinbase_trader/{profile}/events.jsonl`
2. Check metrics file: `cat var/data/coinbase_trader/{profile}/metrics.json`
3. Ensure trading bot has been running and generating events
4. Increase `--lookback-hours` if needed

### Health checks always failing

1. Verify dependencies are available (orders_store, market_data_service, etc.)
2. Check health check thresholds are appropriate for your setup
3. Review health check logs for specific errors

---

## Performance Tuning

### Metrics Export Optimization

For high-frequency trading, adjust scrape intervals:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'gpt-trader'
    scrape_interval: 1s  # Increase frequency for fast cycles
```

### Report Generation Performance

For large event files, optimize report generation:

```bash
# Reduce lookback period
python -m gpt_trader.cli report daily --lookback-hours 12

# Or rotate events.jsonl periodically
mv var/data/coinbase_trader/prod/events.jsonl var/data/coinbase_trader/prod/events.$(date +%Y%m%d).jsonl
touch var/data/coinbase_trader/prod/events.jsonl
```

---

## Support

For issues or questions:
- Check logs in `var/logs/`
- Review Prometheus metrics for anomalies
- Run health checks manually
- Generate detailed reports with `--format json` for analysis
