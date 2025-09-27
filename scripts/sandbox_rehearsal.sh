#!/bin/bash
# Week 3 Phase 1: Sandbox Rehearsal Script
# Tests all order types and flows in sandbox environment

set -e  # Exit on error

echo "=================================================="
echo "WEEK 3 PHASE 1: SANDBOX REHEARSAL"
echo "=================================================="
echo ""
echo "Testing advanced order types in Coinbase Sandbox"
echo "This will place tiny test orders with all features"
echo ""

# Ensure we're in sandbox mode
export COINBASE_SANDBOX=1

# Check for required environment variables
if [ -z "$COINBASE_API_KEY" ] || [ -z "$COINBASE_API_SECRET" ]; then
    echo "❌ Error: COINBASE_API_KEY and COINBASE_API_SECRET must be set"
    echo "   Please set your Coinbase Sandbox API credentials"
    exit 1
fi

echo "✅ Environment configured for sandbox"
echo ""

# Run capability probe first
echo "1. Running capability probe..."
echo "--------------------------------"
python scripts/probe_capabilities.py --live --symbol BTC-PERP

echo ""
echo "2. Testing order types..."
echo "--------------------------------"

# Test market orders with strict filters
echo ""
echo "Testing MARKET orders..."
python scripts/run_perps_bot_v2_week3.py \
    --profile dev \
    --symbols BTC-PERP \
    --order-type market \
    --dry-run \
    --max-spread-bps 10 \
    --min-depth-l1 50000 \
    --rsi-confirm \
    --run-once

# Test limit post-only orders
echo ""
echo "Testing LIMIT POST-ONLY orders..."
python scripts/run_perps_bot_v2_week3.py \
    --profile dev \
    --symbols ETH-PERP \
    --order-type limit \
    --post-only \
    --limit-offset-bps 10 \
    --dry-run \
    --max-spread-bps 10 \
    --min-depth-l1 50000 \
    --run-once

# Test stop orders
echo ""
echo "Testing STOP orders..."
python scripts/run_perps_bot_v2_week3.py \
    --profile dev \
    --symbols BTC-PERP \
    --order-type stop \
    --stop-pct 2 \
    --dry-run \
    --run-once

# Test impact-aware sizing
echo ""
echo "Testing IMPACT-AWARE SIZING..."
python scripts/run_perps_bot_v2_week3.py \
    --profile dev \
    --symbols BTC-PERP \
    --sizing-mode conservative \
    --max-impact-bps 10 \
    --dry-run \
    --run-once

echo ""
echo "3. Running full validation suite..."
echo "--------------------------------"
RUN_SANDBOX_VALIDATIONS=1 python scripts/validate_week3_orders.py

echo ""
echo "=================================================="
echo "SANDBOX REHEARSAL COMPLETE"
echo "=================================================="
echo ""
echo "Next steps:"
echo "1. Review logs for any errors or warnings"
echo "2. Check metrics in EventStore"
echo "3. Verify all order types worked as expected"
echo "4. If successful, proceed to Phase 2 (Demo Profile)"
echo ""
echo "To proceed to Phase 2, run:"
echo "  bash scripts/demo_profile_test.sh"