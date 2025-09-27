#!/bin/bash
# Week 3 Phase 2: Demo Profile Testing
# Tests with tiny real positions ($100-500 notional)

set -e  # Exit on error

echo "=================================================="
echo "WEEK 3 PHASE 2: DEMO PROFILE TESTING"
echo "=================================================="
echo ""
echo "⚠️  WARNING: This will place REAL orders with small notional"
echo "   Profile: demo ($100-500 positions)"
echo "   Symbols: BTC-PERP"
echo ""

read -p "Continue? (yes/no): " response
if [ "$response" != "yes" ]; then
    echo "Aborted"
    exit 0
fi

# Use production API (no sandbox)
unset COINBASE_SANDBOX

# Check for required environment variables
if [ -z "$COINBASE_API_KEY" ] || [ -z "$COINBASE_API_SECRET" ]; then
    echo "❌ Error: COINBASE_API_KEY and COINBASE_API_SECRET must be set"
    exit 1
fi

echo ""
echo "Starting demo profile test..."
echo "Press Ctrl+C to stop"
echo ""

# Run with demo profile settings
python scripts/run_perps_bot_v2_week3.py \
    --profile demo \
    --symbols BTC-PERP \
    --order-type limit \
    --post-only \
    --sizing-mode conservative \
    --max-impact-bps 10 \
    --max-spread-bps 5 \
    --min-depth-l10 200000 \
    --rsi-confirm \
    --liq-buffer-pct 30 \
    --health-file /tmp/week3_demo_health.json \
    --log-level INFO