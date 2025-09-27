#!/bin/bash

# Schedule Phase 2 Canary Test
# Runs canary monitoring for 10-15 minutes with the canary profile

set -e

echo "🚀 Scheduling Phase 2 Canary Test"
echo "=================================="

# Check environment
if [ "$COINBASE_SANDBOX" != "0" ]; then
    echo "❌ Error: COINBASE_SANDBOX must be 0 for production"
    exit 1
fi

if [ "$COINBASE_API_MODE" != "advanced" ]; then
    echo "❌ Error: COINBASE_API_MODE must be 'advanced'"
    exit 1
fi

# Create canary profile if it doesn't exist
PROFILE_PATH="config/profiles/canary.yaml"
if [ ! -f "$PROFILE_PATH" ]; then
    echo "📝 Creating canary profile..."
    mkdir -p config/profiles
    cat > "$PROFILE_PATH" << EOF
name: canary
mode: production
symbols:
  - BTC-PERP
max_position_size: 0.001
max_orders_per_minute: 5
stop_loss_pct: 2.0
enable_reduce_only: true
risk_limits:
  max_daily_loss: 50.0
  max_position_value: 1000.0
  max_leverage: 2.0
EOF
    echo "✅ Canary profile created"
fi

# Phase 0: Preflight
echo ""
echo "📋 Phase 0: Running preflight checks..."
if poetry run python scripts/prod_perps_preflight.py; then
    echo "✅ Preflight checks passed"
else
    echo "❌ Preflight checks failed"
    exit 1
fi

# Phase 1: Preview canary order
echo ""
echo "📋 Phase 1: Testing canary order (preview)..."
if poetry run python scripts/canary_reduce_only_test.py --symbol BTC-PERP --price 10 --qty 0.001; then
    echo "✅ Canary order preview successful"
else
    echo "❌ Canary order preview failed"
    exit 1
fi

# Phase 2: Start monitoring
echo ""
echo "📋 Phase 2: Starting canary monitor..."
echo "Duration: 10 minutes (dry run for safety)"
echo ""

# Run canary monitor with dry-run flag for safety
poetry run python scripts/canary_monitor.py \
    --profile canary \
    --duration-minutes 10 \
    --dashboard \
    --dry-run

EXIT_CODE=$?

echo ""
echo "=================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Canary test completed successfully"
    echo ""
    echo "Next steps:"
    echo "1. Review metrics in results/canary_monitor_*.json"
    echo "2. If all metrics look good, run without --dry-run flag"
    echo "3. Proceed to Phase 3: Gradual scale-up"
else
    echo "⚠️ Canary test completed with issues (exit code: $EXIT_CODE)"
    echo "Review logs and metrics before proceeding"
fi

exit $EXIT_CODE