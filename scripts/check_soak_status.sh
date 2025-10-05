#!/bin/bash
# Quick status check for running soak test
# Usage: ./scripts/check_soak_status.sh

set -e

echo "=========================================="
echo "Sandbox Soak Test - Status Check"
echo "=========================================="
echo ""

# Check monitoring stack
echo "Monitoring Stack Status:"
echo "------------------------"
if curl -s http://localhost:9091/-/ready &> /dev/null; then
    echo "✓ Prometheus:    http://localhost:9091"
else
    echo "✗ Prometheus:    Not responding"
fi

if curl -s http://localhost:3000/api/health &> /dev/null; then
    echo "✓ Grafana:       http://localhost:3000"
else
    echo "✗ Grafana:       Not responding"
fi

if curl -s http://localhost:9093/-/ready &> /dev/null; then
    echo "✓ Alertmanager:  http://localhost:9093"
else
    echo "✗ Alertmanager:  Not responding"
fi
echo ""

# Check bot health
echo "Bot Health:"
echo "------------------------"
if curl -s http://localhost:9090/health &> /dev/null; then
    echo "✓ Health endpoint: http://localhost:9090/health"
    echo ""
    echo "Health Details:"
    curl -s http://localhost:9090/health | jq '
    {
        status: .status,
        uptime_seconds: .uptime_seconds,
        streaming: .streaming.connected,
        heartbeat_lag: .streaming.heartbeat_lag_seconds,
        fallback_active: .streaming.fallback_active,
        active_guards: .guards
    }' 2>/dev/null || echo "   (jq not installed - run: brew install jq)"
else
    echo "✗ Bot not responding at http://localhost:9090/health"
    echo "   Bot may not be running. Check with: ps aux | grep bot_v2"
fi
echo ""

# Check key metrics
echo "Key Metrics Snapshot:"
echo "------------------------"
if curl -s http://localhost:9090/metrics &> /dev/null; then
    echo "Uptime:           $(curl -s http://localhost:9090/metrics | grep '^bot_uptime_seconds{' | awk '{print $2}')s"
    echo "Streaming State:  $(curl -s http://localhost:9090/metrics | grep '^bot_streaming_connection_state{' | awk '{print $2}')"
    echo "Fallback Active:  $(curl -s http://localhost:9090/metrics | grep '^bot_streaming_fallback_active{' | awk '{print $2}')"
    echo "Active Guards:    $(curl -s http://localhost:9090/metrics | grep '^bot_guard_active{' | grep ' 1$' | wc -l | xargs)"
    echo "Guard Trips:      $(curl -s http://localhost:9090/metrics | grep '^bot_guard_trips_total{' | awk '{sum+=$2} END {print sum}')"
    echo "Order Attempts:   $(curl -s http://localhost:9090/metrics | grep 'bot_order_attempts_total{.*status="attempted"' | awk '{print $2}')"
    echo "Order Success:    $(curl -s http://localhost:9090/metrics | grep 'bot_order_attempts_total{.*status="success"' | awk '{print $2}')"
else
    echo "✗ Metrics endpoint not responding"
fi
echo ""

# Check active alerts
echo "Active Alerts:"
echo "------------------------"
ALERTS=$(curl -s http://localhost:9091/api/v1/alerts | jq -r '.data[] | select(.state=="firing") | .labels.alertname' 2>/dev/null || echo "")
if [ -z "$ALERTS" ]; then
    echo "✓ No alerts firing"
else
    echo "$ALERTS"
fi
echo ""

# Check recent logs (if running in background)
LATEST_LOG=$(ls -t logs/sandbox_soak_*.log 2>/dev/null | head -1)
if [ -n "$LATEST_LOG" ]; then
    echo "Latest Log File: $LATEST_LOG"
    echo "Last 5 lines:"
    tail -5 "$LATEST_LOG"
else
    echo "No log files found in logs/"
fi
echo ""

echo "=========================================="
echo "Quick Commands:"
echo "  View health:    curl http://localhost:9090/health | jq ."
echo "  View metrics:   curl http://localhost:9090/metrics | grep bot_"
echo "  View Grafana:   open http://localhost:3000"
echo "  View alerts:    open http://localhost:9093"
echo "  Tail logs:      tail -f $LATEST_LOG"
echo "=========================================="
