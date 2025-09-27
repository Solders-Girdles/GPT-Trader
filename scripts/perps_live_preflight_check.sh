#!/bin/bash
# Live Pre-Flight Check
# Verifies environment, connectivity, and safety config before demo trading

set -e

echo "=================================================="
echo "LIVE PRE-FLIGHT CHECK"
echo "=================================================="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check environment variable
check_env() {
    local var_name=$1
    local var_value=${!var_name}
    if [ -z "$var_value" ]; then
        echo -e "${RED}❌ $var_name not set${NC}"
        return 1
    else
        echo -e "${GREEN}✅ $var_name set${NC}"
        return 0
    fi
}

# Function to check optional env
check_optional() {
    local var_name=$1
    local var_value=${!var_name}
    if [ -z "$var_value" ]; then
        echo -e "${YELLOW}⚠️  $var_name not set (optional)${NC}"
    else
        echo -e "${GREEN}✅ $var_name set${NC}"
    fi
}

echo "1. Environment Configuration"
echo "----------------------------"

# Required for Advanced Trade API
export COINBASE_API_MODE=advanced
export COINBASE_AUTH_TYPE=JWT
export COINBASE_ENABLE_DERIVATIVES=1
export USE_REAL_ADAPTER=1
unset PERPS_FORCE_MOCK
unset COINBASE_SANDBOX

echo "API Mode: $COINBASE_API_MODE"
echo "Auth Type: $COINBASE_AUTH_TYPE"
echo "Derivatives: $COINBASE_ENABLE_DERIVATIVES"
echo ""

# Check required credentials
echo "2. Credentials Check"
echo "-------------------"
all_good=true

check_env "COINBASE_CDP_API_KEY" || all_good=false
check_env "COINBASE_CDP_PRIVATE_KEY" || all_good=false
check_optional "EVENT_STORE_ROOT"

if [ "$all_good" = false ]; then
    echo -e "\n${RED}Missing required credentials. Please set:${NC}"
    echo "  export COINBASE_CDP_API_KEY='your_key'"
    echo "  export COINBASE_CDP_PRIVATE_KEY='your_private_key'"
    exit 1
fi

echo ""
echo "3. System Clock Check"
echo "--------------------"
# Check NTP sync
if command -v timedatectl &> /dev/null; then
    sync_status=$(timedatectl | grep "synchronized" | awk '{print $3}')
    if [ "$sync_status" = "yes" ]; then
        echo -e "${GREEN}✅ System clock synchronized${NC}"
    else
        echo -e "${YELLOW}⚠️  System clock may not be synchronized${NC}"
    fi
else
    # macOS alternative
    if command -v sntp &> /dev/null; then
        echo "Checking time sync..."
        sntp -t 1 time.apple.com &> /dev/null && echo -e "${GREEN}✅ Time sync OK${NC}" || echo -e "${YELLOW}⚠️  Time sync check failed${NC}"
    else
        echo -e "${YELLOW}⚠️  Cannot verify time sync${NC}"
    fi
fi

# Show current time for manual verification
echo "Current system time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"

echo ""
echo "4. Connectivity Test"
echo "-------------------"

# Test DNS resolution
echo -n "DNS resolution: "
if host api.coinbase.com &> /dev/null; then
    echo -e "${GREEN}✅ OK${NC}"
else
    echo -e "${RED}❌ Failed${NC}"
    exit 1
fi

# Test HTTPS connectivity
echo -n "HTTPS connectivity: "
if curl -s -o /dev/null -w "%{http_code}" https://api.coinbase.com/api/v3/brokerage/products | grep -q "200\|401"; then
    echo -e "${GREEN}✅ OK${NC}"
else
    echo -e "${RED}❌ Failed${NC}"
fi

echo ""
echo "5. Capability Probe"
echo "------------------"

# Create test script for real adapter
cat > /tmp/test_real_adapter.py << 'EOF'
#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, '.')

# Set environment
os.environ['COINBASE_API_MODE'] = 'advanced'
os.environ['COINBASE_AUTH_TYPE'] = 'JWT'
os.environ['COINBASE_ENABLE_DERIVATIVES'] = '1'
os.environ.pop('COINBASE_SANDBOX', None)
os.environ.pop('PERPS_FORCE_MOCK', None)

try:
    from bot_v2.orchestration.broker_factory import create_brokerage
    
    print("Creating real Coinbase adapter...")
    broker = create_brokerage()
    
    print("Testing connection...")
    if broker.connect():
        print("✅ Connected successfully")
        
        # Test list products
        print("\nFetching perpetual products...")
        from bot_v2.features.brokerages.core.interfaces import MarketType
        products = broker.list_products(market=MarketType.PERPETUAL)
        
        if products:
            print(f"✅ Found {len(products)} perpetual products")
            # Show first few
            for p in products[:3]:
                print(f"   - {p.symbol}")
        else:
            print("⚠️  No perpetual products found")
        
        # Test quote
        print("\nTesting quote for BTC-PERP...")
        quote = broker.get_quote("BTC-PERP")
        if quote:
            print(f"✅ BTC-PERP: bid={quote.bid}, ask={quote.ask}, spread={(quote.ask-quote.bid)/quote.ask*10000:.1f}bps")
        else:
            print("⚠️  Could not get quote")
        
        broker.disconnect()
    else:
        print("❌ Connection failed")
        print("   Check credentials and network")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
EOF

python /tmp/test_real_adapter.py

echo ""
echo "6. Safety Configuration"
echo "-----------------------"

cat > /tmp/demo_safety_config.json << 'EOF'
{
    "sizing_mode": "conservative",
    "max_impact_bps": 10,
    "max_spread_bps": 5,
    "min_depth_l1": 50000,
    "min_depth_l10": 200000,
    "rsi_confirm": true,
    "max_position_size": 0.01,
    "leverage": 1,
    "daily_loss_limit": 100,
    "liq_buffer_pct": 30,
    "reduce_only_exits": true,
    "kill_switch_enabled": true,
    "order_type": "limit",
    "post_only": true,
    "target_notional_min": 25,
    "target_notional_max": 100,
    "pre_funding_quiet_mins": 30
}
EOF

echo "Safety config prepared:"
echo -e "${GREEN}✅ Conservative sizing${NC}"
echo -e "${GREEN}✅ Strict filters (RSI confirm)${NC}"
echo -e "${GREEN}✅ Tiny notional (\$25-\$100)${NC}"
echo -e "${GREEN}✅ Post-only limit orders${NC}"
echo -e "${GREEN}✅ Daily loss limit (\$100)${NC}"
echo -e "${GREEN}✅ Kill switch enabled${NC}"

echo ""
echo "7. Demo Commands Ready"
echo "----------------------"

echo "Dry-run pulse (1 cycle):"
echo "  python scripts/run_perps_bot_v2_week3.py --profile demo --dry-run --run-once"
echo ""
echo "Live demo (tiny positions):"
echo "  python scripts/run_perps_bot_v2_week3.py \\"
echo "    --profile demo \\"
echo "    --symbols BTC-PERP \\"
echo "    --order-type limit \\"
echo "    --post-only \\"
echo "    --sizing-mode conservative \\"
echo "    --max-impact-bps 10 \\"
echo "    --rsi-confirm \\"
echo "    --health-file /tmp/phase2_health.json"

echo ""
echo "=================================================="
echo "PRE-FLIGHT CHECK COMPLETE"
echo "=================================================="

if [ "$all_good" = true ]; then
    echo -e "${GREEN}✅ System ready${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Run dry-run pulse to verify connectivity"
    echo "2. Place test orders with demo profile"
    echo "3. Monitor metrics and logs"
else
    echo -e "${RED}❌ Issues detected - resolve before proceeding${NC}"
fi
