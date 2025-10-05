#!/bin/bash
# Collect metrics snapshots during soak test
# Usage: ./scripts/collect_soak_metrics.sh
# Run this in background: ./scripts/collect_soak_metrics.sh &

set -e

PROJECT_ROOT="/Users/rj/PycharmProjects/GPT-Trader"
DATA_DIR="$PROJECT_ROOT/data"
INTERVAL=3600  # 1 hour

mkdir -p "$DATA_DIR"

echo "Starting metrics collection (interval: ${INTERVAL}s)"
echo "Data directory: $DATA_DIR"
echo "Press Ctrl+C to stop"
echo ""

while true; do
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    echo "[$(date)] Collecting metrics snapshot: $TIMESTAMP"

    # Collect instant metrics
    curl -s "http://localhost:9091/api/v1/query?query=bot_uptime_seconds" \
        > "$DATA_DIR/soak_${TIMESTAMP}_uptime.json" 2>/dev/null || echo "Failed to collect uptime"

    curl -s "http://localhost:9091/api/v1/query?query=bot_guard_active" \
        > "$DATA_DIR/soak_${TIMESTAMP}_guards.json" 2>/dev/null || echo "Failed to collect guards"

    curl -s "http://localhost:9091/api/v1/query?query=bot_streaming_connection_state" \
        > "$DATA_DIR/soak_${TIMESTAMP}_streaming.json" 2>/dev/null || echo "Failed to collect streaming"

    curl -s "http://localhost:9091/api/v1/query?query=bot_guard_daily_loss_usd" \
        > "$DATA_DIR/soak_${TIMESTAMP}_daily_loss.json" 2>/dev/null || echo "Failed to collect daily loss"

    curl -s "http://localhost:9091/api/v1/query?query=bot_streaming_fallback_active" \
        > "$DATA_DIR/soak_${TIMESTAMP}_fallback.json" 2>/dev/null || echo "Failed to collect fallback"

    # Health snapshot
    curl -s "http://localhost:9090/health" \
        > "$DATA_DIR/soak_${TIMESTAMP}_health.json" 2>/dev/null || echo "Failed to collect health"

    # Alert snapshot
    curl -s "http://localhost:9091/api/v1/alerts" \
        > "$DATA_DIR/soak_${TIMESTAMP}_alerts.json" 2>/dev/null || echo "Failed to collect alerts"

    echo "   Snapshot complete: $TIMESTAMP"

    # Wait for next collection
    sleep $INTERVAL
done
