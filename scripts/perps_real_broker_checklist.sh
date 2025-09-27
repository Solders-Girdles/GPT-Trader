#!/bin/bash
# Real Broker Execution Checklist
# Final verification before live trading

set -e

echo "=================================================="
echo "REAL BROKER CHECKLIST"
echo "=================================================="
echo ""
echo "This script verifies all requirements for real broker trading"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Track overall status
ALL_GOOD=true

echo "1. ENVIRONMENT CONFIGURATION"
echo "----------------------------"

# Required environment variables
export COINBASE_API_MODE=advanced
export COINBASE_AUTH_TYPE=JWT
export COINBASE_ENABLE_DERIVATIVES=1
export USE_REAL_ADAPTER=1
export EVENT_STORE_ROOT=${EVENT_STORE_ROOT:-/tmp/phase2_eventstore}

# Unset any mock/sandbox flags
unset PERPS_FORCE_MOCK
unset COINBASE_SANDBOX

echo "API Mode: $COINBASE_API_MODE"
echo "Auth Type: $COINBASE_AUTH_TYPE"
echo "Derivatives: $COINBASE_ENABLE_DERIVATIVES"
echo "EventStore: $EVENT_STORE_ROOT"

# Check credentials
echo ""
echo "2. CREDENTIALS CHECK"
echo "-------------------"

if [ -z "$COINBASE_CDP_API_KEY" ]; then
    echo -e "${RED}❌ COINBASE_CDP_API_KEY not set${NC}"
    ALL_GOOD=false
else
    echo -e "${GREEN}✅ COINBASE_CDP_API_KEY set${NC}"
fi

if [ -z "$COINBASE_CDP_PRIVATE_KEY" ]; then
    echo -e "${RED}❌ COINBASE_CDP_PRIVATE_KEY not set${NC}"
    ALL_GOOD=false
else
    echo -e "${GREEN}✅ COINBASE_CDP_PRIVATE_KEY set${NC}"
fi

# Clock sync check
echo ""
echo "3. SYSTEM CLOCK SYNC"
echo "--------------------"
CURRENT_TIME=$(date -u '+%Y-%m-%d %H:%M:%S UTC')
echo "Current UTC: $CURRENT_TIME"

# Check time sync
if command -v ntpdate &> /dev/null; then
    OFFSET=$(ntpdate -q time.google.com 2>/dev/null | grep -oE 'offset [0-9.-]+' | awk '{print $2}')
    if [ -n "$OFFSET" ]; then
        ABS_OFFSET=$(echo "$OFFSET" | tr -d '-' | cut -d. -f1)
        if [ "$ABS_OFFSET" -lt 2 ]; then
            echo -e "${GREEN}✅ Clock sync OK (offset: ${OFFSET}s)${NC}"
        else
            echo -e "${YELLOW}⚠️  Clock offset high: ${OFFSET}s${NC}"
        fi
    fi
else
    echo -e "${YELLOW}⚠️  Cannot verify time sync${NC}"
fi

# DNS/TLS check
echo ""
echo "4. NETWORK CONNECTIVITY"
echo "-----------------------"

echo -n "DNS resolution: "
if host api.coinbase.com &> /dev/null; then
    echo -e "${GREEN}✅ OK${NC}"
else
    echo -e "${RED}❌ Failed${NC}"
    ALL_GOOD=false
fi

echo -n "TLS/HTTPS: "
if curl -s -o /dev/null -w "%{http_code}" https://api.coinbase.com/api/v3/brokerage/products | grep -q "200\|401"; then
    echo -e "${GREEN}✅ OK${NC}"
else
    echo -e "${RED}❌ Failed${NC}"
    ALL_GOOD=false
fi

# Create directories
echo ""
echo "5. DIRECTORY SETUP"
echo "------------------"
mkdir -p $EVENT_STORE_ROOT
mkdir -p /tmp/phase2_logs
echo -e "${GREEN}✅ Directories created${NC}"

# Safety configuration
echo ""
echo "6. SAFETY CONFIGURATION"
echo "-----------------------"

cat > /tmp/phase2_safety.json << 'EOF'
{
    "phase": "Demo Profile",
    "settings": {
        "sizing_mode": "conservative",
        "max_impact_bps": 10,
        "target_notional": "$25-100",
        "max_position_size": 0.01,
        "daily_loss_limit": "$100",
        "leverage": 1,
        "order_type": "limit",
        "post_only": true,
        "auto_cancel_seconds": 30,
        "pre_funding_quiet_mins": 30,
        "kill_switch_enabled": true,
        "reduce_only_exits": true,
        "rsi_confirm": true,
        "max_spread_bps": 5,
        "min_depth_l1": 50000,
        "min_depth_l10": 200000
    }
}
EOF

echo -e "${GREEN}✅ Safety configuration saved${NC}"
echo "   - Conservative sizing"
echo "   - $25-100 positions"
echo "   - Post-only limits"
echo "   - $100 daily loss limit"
echo "   - Kill switch ready"

# Final summary
echo ""
echo "=================================================="
echo "FINAL STATUS"
echo "=================================================="

if [ "$ALL_GOOD" = true ]; then
    echo -e "${GREEN}✅ ALL CHECKS PASSED - READY${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Run capability probe:"
    echo "   python scripts/probe_capabilities.py --live"
    echo ""
    echo "2. Dry-run pulse (1 cycle):"
    echo "   python scripts/demos/perps_demo_runner.py --dry-run"
    echo ""
    echo "3. Live demo (tiny positions):"
    echo "   python scripts/demos/perps_demo_runner.py"
    echo ""
    echo "4. Monitor metrics:"
    echo "   python scripts/perps_metrics_monitor.py"
    echo ""
    echo "Safety reminder:"
    echo "- Start with BTC-PERP only"
    echo "- Monitor for 5-10 minutes"
    echo "- Test kill switch once (Ctrl+C)"
    echo "- Check /tmp/phase2_logs/ for details"
else
    echo -e "${RED}❌ CHECKS FAILED - RESOLVE ISSUES BEFORE PROCEEDING${NC}"
    echo ""
    echo "Required actions:"
    if [ -z "$COINBASE_CDP_API_KEY" ] || [ -z "$COINBASE_CDP_PRIVATE_KEY" ]; then
        echo "- Set CDP credentials:"
        echo "  export COINBASE_CDP_API_KEY='your_key'"
        echo "  export COINBASE_CDP_PRIVATE_KEY='your_private_key'"
    fi
fi
