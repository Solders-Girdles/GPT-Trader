#!/bin/bash
# Week 3 Phase 3: Production Canary
# Limited production deployment with conservative settings

set -e  # Exit on error

echo "=================================================="
echo "WEEK 3 PHASE 3: PRODUCTION CANARY"
echo "=================================================="
echo ""
echo "🚨 WARNING: This is PRODUCTION deployment"
echo "   Profile: prod (full-size positions)"
echo "   Symbols: BTC-PERP (expanding to ETH-PERP after 48h)"
echo "   Mode: Conservative with strict limits"
echo ""

read -p "Are you SURE you want to deploy to production? (yes/no): " response
if [ "$response" != "yes" ]; then
    echo "Aborted"
    exit 0
fi

read -p "Type 'PRODUCTION' to confirm: " confirm
if [ "$confirm" != "PRODUCTION" ]; then
    echo "Aborted"
    exit 0
fi

# Use production API
unset COINBASE_SANDBOX

# Check for required environment variables
if [ -z "$COINBASE_API_KEY" ] || [ -z "$COINBASE_API_SECRET" ]; then
    echo "❌ Error: Production API credentials not set"
    exit 1
fi

echo ""
echo "📊 Starting production canary with conservative settings..."
echo "   - Impact limit: 5 bps"
echo "   - Max spread: 3 bps"
echo "   - Min L1 depth: $100k"
echo "   - Min L10 depth: $500k"
echo "   - Liquidation buffer: 25%"
echo "   - Max slippage: 10 bps"
echo ""
echo "Press Ctrl+C to activate kill switch"
echo ""

# Log start time
echo "Start time: $(date)" > /tmp/week3_canary.log

# Run production canary
python scripts/run_perps_bot_v2_week3.py \
    --profile prod \
    --symbols BTC-PERP \
    --order-type market \
    --sizing-mode conservative \
    --max-impact-bps 5 \
    --max-spread-bps 3 \
    --min-depth-l1 100000 \
    --min-depth-l10 500000 \
    --min-vol-1m 100000 \
    --rsi-confirm \
    --liq-buffer-pct 25 \
    --max-slippage-bps 10 \
    --health-file /tmp/week3_prod_health.json \
    --metrics-file /tmp/week3_prod_metrics.json \
    --log-level INFO \
    2>&1 | tee -a /tmp/week3_canary.log