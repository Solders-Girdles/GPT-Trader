#!/bin/bash
# Week 2 Validation Script with Correct CLI Flags

echo "Week 2 Perpetuals Bot Validation"
echo "================================="
echo ""

# Set environment variables
export COINBASE_API_KEY="${COINBASE_API_KEY:-your_api_key}"
export COINBASE_API_SECRET="${COINBASE_API_SECRET:-your_secret}"
export COINBASE_PASSPHRASE="${COINBASE_PASSPHRASE:-your_passphrase}"
export EVENT_STORE_ROOT="${EVENT_STORE_ROOT:-./data/events}"

# Validation scenarios with correct flags

echo "Scenario 1: Conservative filters with RSI confirmation"
echo "------------------------------------------------------"
python scripts/run_perps_bot_v2.py \
    --symbols BTC-PERP \
    --max-spread-bps 5 \
    --min-depth-l1 100000 \
    --min-vol-1m 500000 \
    --rsi-confirm \
    --max-slippage-bps 10 \
    --liq-buffer-pct 25 \
    --dry-run \
    --cycles 10

echo ""
echo "Scenario 2: Moderate filters without RSI"
echo "----------------------------------------"
python scripts/run_perps_bot_v2.py \
    --symbols ETH-PERP \
    --max-spread-bps 10 \
    --min-depth-l1 50000 \
    --min-vol-1m 100000 \
    --max-slippage-bps 15 \
    --liq-buffer-pct 20 \
    --dry-run \
    --cycles 10

echo ""
echo "Scenario 3: Multi-symbol with aggressive filters"
echo "------------------------------------------------"
python scripts/run_perps_bot_v2.py \
    --symbols BTC-PERP,ETH-PERP,SOL-PERP \
    --max-spread-bps 20 \
    --min-depth-l1 25000 \
    --min-vol-1m 50000 \
    --max-slippage-bps 20 \
    --liq-buffer-pct 15 \
    --dry-run \
    --cycles 10

echo ""
echo "Scenario 4: Test rejection tracking"
echo "-----------------------------------"
# Very strict filters to force rejections
python scripts/run_perps_bot_v2.py \
    --symbols BTC-PERP \
    --max-spread-bps 1 \
    --min-depth-l1 1000000 \
    --min-vol-1m 10000000 \
    --rsi-confirm \
    --max-slippage-bps 1 \
    --liq-buffer-pct 50 \
    --dry-run \
    --cycles 5

echo ""
echo "Validation complete. Check logs for rejection metrics."
echo ""
echo "Expected behaviors:"
echo "- Scenario 1: Most entries rejected (conservative)"
echo "- Scenario 2: Some entries accepted (moderate)"
echo "- Scenario 3: More entries accepted (aggressive)"
echo "- Scenario 4: All entries rejected (impossible filters)"