#!/bin/bash
# Export full time-series metrics after soak test completion
# Usage: ./scripts/export_final_metrics.sh START_TIME END_TIME
# Example: ./scripts/export_final_metrics.sh "2025-10-04T00:00:00Z" "2025-10-06T00:00:00Z"

set -e

if [ $# -lt 2 ]; then
    echo "Usage: $0 START_TIME END_TIME"
    echo "Example: $0 \"2025-10-04T00:00:00Z\" \"2025-10-06T00:00:00Z\""
    echo ""
    echo "To use current time and last 24 hours:"
    echo "  $0 \"\$(date -u -v-24H +%Y-%m-%dT%H:%M:%SZ)\" \"\$(date -u +%Y-%m-%dT%H:%M:%SZ)\""
    exit 1
fi

START_TIME="$1"
END_TIME="$2"
PROJECT_ROOT="/Users/rj/PycharmProjects/GPT-Trader"
DATA_DIR="$PROJECT_ROOT/data"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$DATA_DIR"

echo "=========================================="
echo "Soak Test - Final Metrics Export"
echo "=========================================="
echo "Start Time: $START_TIME"
echo "End Time:   $END_TIME"
echo "Output Dir: $DATA_DIR"
echo ""

# Export all bot metrics
echo "Exporting all bot metrics..."
curl -G "http://localhost:9091/api/v1/query_range" \
    --data-urlencode "query={__name__=~'bot_.*'}" \
    --data-urlencode "start=$START_TIME" \
    --data-urlencode "end=$END_TIME" \
    --data-urlencode "step=15s" \
    > "$DATA_DIR/soak_full_metrics_${TIMESTAMP}.json"

# Export specific metric groups
echo "Exporting guardrail metrics..."
curl -G "http://localhost:9091/api/v1/query_range" \
    --data-urlencode "query=bot_guard_active" \
    --data-urlencode "start=$START_TIME" \
    --data-urlencode "end=$END_TIME" \
    --data-urlencode "step=60s" \
    > "$DATA_DIR/soak_guards_${TIMESTAMP}.json"

echo "Exporting streaming metrics..."
curl -G "http://localhost:9091/api/v1/query_range" \
    --data-urlencode "query=bot_streaming_connection_state" \
    --data-urlencode "start=$START_TIME" \
    --data-urlencode "end=$END_TIME" \
    --data-urlencode "step=15s" \
    > "$DATA_DIR/soak_streaming_${TIMESTAMP}.json"

echo "Exporting daily loss tracking..."
curl -G "http://localhost:9091/api/v1/query_range" \
    --data-urlencode "query=bot_guard_daily_loss_usd" \
    --data-urlencode "start=$START_TIME" \
    --data-urlencode "end=$END_TIME" \
    --data-urlencode "step=60s" \
    > "$DATA_DIR/soak_daily_loss_${TIMESTAMP}.json"

echo "Exporting reconnect counts..."
curl -G "http://localhost:9091/api/v1/query_range" \
    --data-urlencode "query=bot_streaming_reconnect_total" \
    --data-urlencode "start=$START_TIME" \
    --data-urlencode "end=$END_TIME" \
    --data-urlencode "step=60s" \
    > "$DATA_DIR/soak_reconnects_${TIMESTAMP}.json"

echo "Exporting order statistics..."
curl -G "http://localhost:9091/api/v1/query_range" \
    --data-urlencode "query=bot_order_attempts_total" \
    --data-urlencode "start=$START_TIME" \
    --data-urlencode "end=$END_TIME" \
    --data-urlencode "step=60s" \
    > "$DATA_DIR/soak_orders_${TIMESTAMP}.json"

# Export alert history
echo "Exporting alert history..."
curl -s "http://localhost:9091/api/v1/alerts" \
    > "$DATA_DIR/soak_alerts_${TIMESTAMP}.json"

# Create summary
echo ""
echo "Export complete. Files created:"
ls -lh "$DATA_DIR"/soak_*_${TIMESTAMP}.json

echo ""
echo "=========================================="
echo "Next Steps:"
echo "  1. Archive data:  tar -czf archive/soak_${TIMESTAMP}.tar.gz data/"
echo "  2. Copy logs:     cp logs/sandbox_soak_*.log archive/"
echo "  3. Generate report using Python analysis script"
echo "=========================================="
