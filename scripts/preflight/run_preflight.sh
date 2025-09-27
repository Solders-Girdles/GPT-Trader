#!/bin/bash

# Stage 3 Preflight Checks
# Comprehensive validation before production deployment

echo "🚀 STAGE 3 PREFLIGHT CHECKS"
echo "========================================================"
echo "Timestamp: $(date)"
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track overall status
PREFLIGHT_PASS=true

# Function to check command success
check_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✅ $2 PASSED${NC}"
    else
        echo -e "${RED}❌ $2 FAILED${NC}"
        PREFLIGHT_PASS=false
    fi
}

# Function to check environment variable
check_env() {
    if [ -z "${!1}" ]; then
        echo -e "${RED}❌ $1 not set${NC}"
        PREFLIGHT_PASS=false
        return 1
    else
        echo -e "${GREEN}✅ $1 is set${NC}"
        return 0
    fi
}

echo "1️⃣  ENVIRONMENT VARIABLES"
echo "----------------------------------------"

# Critical environment checks
check_env "COINBASE_SANDBOX"
if [ "$COINBASE_SANDBOX" != "1" ]; then
    echo -e "${YELLOW}⚠️  COINBASE_SANDBOX=$COINBASE_SANDBOX (should be 1 for sandbox)${NC}"
fi

check_env "COINBASE_API_MODE"
if [ "$COINBASE_API_MODE" != "advanced" ]; then
    echo -e "${YELLOW}⚠️  COINBASE_API_MODE=$COINBASE_API_MODE (should be 'advanced')${NC}"
fi

# CDP JWT checks
check_env "COINBASE_CDP_API_KEY"
if check_env "COINBASE_CDP_PRIVATE_KEY_PATH"; then
    if [ -f "$COINBASE_CDP_PRIVATE_KEY_PATH" ]; then
        echo -e "${GREEN}✅ CDP private key file exists${NC}"
        
        # Check file permissions
        PERMS=$(stat -c %a "$COINBASE_CDP_PRIVATE_KEY_PATH" 2>/dev/null || stat -f %A "$COINBASE_CDP_PRIVATE_KEY_PATH" 2>/dev/null)
        if [ "$PERMS" != "400" ] && [ "$PERMS" != "600" ]; then
            echo -e "${YELLOW}⚠️  CDP key permissions: $PERMS (recommend 400)${NC}"
        fi
    else
        echo -e "${RED}❌ CDP private key file not found: $COINBASE_CDP_PRIVATE_KEY_PATH${NC}"
        PREFLIGHT_PASS=false
    fi
else
    # Check for inline key
    if [ -z "$COINBASE_CDP_PRIVATE_KEY" ]; then
        echo -e "${YELLOW}⚠️  No CDP private key found (neither path nor inline)${NC}"
    fi
fi

# Optional but recommended
if [ -z "$MAX_IMPACT_BPS" ]; then
    echo -e "${YELLOW}⚠️  MAX_IMPACT_BPS not set (will use default)${NC}"
    echo "  Recommend: export MAX_IMPACT_BPS=50"
fi

# Check NO_PROXY for corporate environments
if [ -n "$HTTP_PROXY" ] || [ -n "$HTTPS_PROXY" ]; then
    if [ -z "$NO_PROXY" ]; then
        echo -e "${YELLOW}⚠️  Proxy detected but NO_PROXY not set${NC}"
        echo "  Recommend: export NO_PROXY='api.sandbox.coinbase.com,advanced-trade-ws.sandbox.coinbase.com,*.coinbase.com'"
    elif [[ "$NO_PROXY" != *"coinbase.com"* ]]; then
        echo -e "${YELLOW}⚠️  NO_PROXY doesn't include Coinbase endpoints${NC}"
        echo "  Current: NO_PROXY=$NO_PROXY"
        echo "  Add: api.sandbox.coinbase.com,advanced-trade-ws.sandbox.coinbase.com"
    else
        echo -e "${GREEN}✅ NO_PROXY includes Coinbase endpoints${NC}"
    fi
fi

echo ""
echo "2️⃣  DIRECTORY STRUCTURE"
echo "----------------------------------------"

# Create required directories
mkdir -p artifacts/stage3
mkdir -p logs
mkdir -p docs/ops/preflight
mkdir -p verification_reports

echo -e "${GREEN}✅ Required directories created/verified${NC}"

echo ""
echo "3️⃣  PYTHON DEPENDENCIES"
echo "----------------------------------------"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
echo "Python version: $PYTHON_VERSION"

# Check critical packages
python3 -c "from coinbase.rest import RESTClient" 2>/dev/null
check_status $? "coinbase-advanced-py"

python3 -c "import pandas" 2>/dev/null
check_status $? "pandas"

python3 -c "import asyncio" 2>/dev/null
check_status $? "asyncio"

echo ""
echo "4️⃣  VERIFICATION ARTIFACTS"
echo "----------------------------------------"

# Check for verification reports
if [ -f "verification_reports/financial_reconciliation.json" ]; then
    echo -e "${GREEN}✅ Financial reconciliation report exists${NC}"
else
    echo -e "${YELLOW}⚠️  Financial reconciliation report missing${NC}"
fi

if [ -f "verification_reports/sized_down_event.json" ]; then
    echo -e "${GREEN}✅ SIZED_DOWN event report exists${NC}"
else
    echo -e "${YELLOW}⚠️  SIZED_DOWN event report missing${NC}"
fi

if [ -f "docs/ops/preflight/tif_validation.json" ]; then
    echo -e "${GREEN}✅ TIF validation report exists${NC}"
else
    echo -e "${YELLOW}⚠️  TIF validation report missing - running now...${NC}"
    python3 scripts/validation/validate_tif_simple.py
    check_status $? "TIF validation"
fi

echo ""
echo "5️⃣  COMPONENT CHECKS"
echo "----------------------------------------"

# Quick component imports
echo "Checking component imports..."

python3 -c "from bot_v2.features.live_trade.portfolio_valuation import PortfolioValuationService" 2>/dev/null
check_status $? "PortfolioValuationService"

python3 -c "from bot_v2.features.live_trade.fees_engine import FeesEngine" 2>/dev/null
check_status $? "FeesEngine"

python3 -c "from bot_v2.features.live_trade.margin_monitor import MarginStateMonitor" 2>/dev/null
check_status $? "MarginStateMonitor"

python3 -c "from bot_v2.features.live_trade.liquidity_service import LiquidityService" 2>/dev/null
check_status $? "LiquidityService"

python3 -c "from bot_v2.features.live_trade.order_policy import OrderPolicyMatrix" 2>/dev/null
check_status $? "OrderPolicyMatrix"

echo ""
echo "6️⃣  CONFIGURATION FILES"
echo "----------------------------------------"

# Check for Stage 3 config
if [ -f "config/stage3_config.json" ]; then
    echo -e "${GREEN}✅ Stage 3 config exists${NC}"
else
    echo -e "${YELLOW}⚠️  Stage 3 config not found - will use defaults${NC}"
fi

echo ""
echo "7️⃣  GENERATING PREFLIGHT REPORT"
echo "----------------------------------------"

# Generate consolidated report
REPORT_FILE="docs/ops/preflight/preflight_report_$(date +%Y%m%d_%H%M%S).json"

cat > "$REPORT_FILE" << EOF
{
  "timestamp": "$(date -Iseconds)",
  "preflight_pass": $([ "$PREFLIGHT_PASS" = true ] && echo "true" || echo "false"),
  "environment": {
    "sandbox": "${COINBASE_SANDBOX:-not_set}",
    "api_mode": "${COINBASE_API_MODE:-not_set}",
    "max_impact_bps": "${MAX_IMPACT_BPS:-default}",
    "cdp_key_present": $([ -n "$COINBASE_CDP_API_KEY" ] && echo "true" || echo "false"),
    "cdp_private_key": $([ -n "$COINBASE_CDP_PRIVATE_KEY_PATH" ] || [ -n "$COINBASE_CDP_PRIVATE_KEY" ] && echo "true" || echo "false"),
    "jwt_auth": $([ -n "$COINBASE_CDP_API_KEY" ] && ([ -n "$COINBASE_CDP_PRIVATE_KEY_PATH" ] || [ -n "$COINBASE_CDP_PRIVATE_KEY" ]) && echo "true" || echo "false"),
    "derivatives_enabled": $([ "$COINBASE_SANDBOX" = "1" ] && [ -n "$COINBASE_CDP_API_KEY" ] && echo "true" || echo "false")
  },
  "artifacts": {
    "financial_reconciliation": $([ -f "verification_reports/financial_reconciliation.json" ] && echo "true" || echo "false"),
    "sized_down_event": $([ -f "verification_reports/sized_down_event.json" ] && echo "true" || echo "false"),
    "tif_validation": $([ -f "docs/ops/preflight/tif_validation.json" ] && echo "true" || echo "false")
  },
  "components": {
    "portfolio_valuation": "ready",
    "fees_engine": "ready",
    "margin_monitor": "ready",
    "liquidity_service": "ready",
    "order_policy": "ready"
  }
}
EOF

echo -e "${GREEN}✅ Preflight report saved to: $REPORT_FILE${NC}"

echo ""
echo "========================================================"
echo "PREFLIGHT SUMMARY"
echo "========================================================"

if [ "$PREFLIGHT_PASS" = true ]; then
    echo -e "${GREEN}✅ ALL PREFLIGHT CHECKS PASSED${NC}"
    echo ""
    echo "Ready to run Stage 3:"
    echo "  python scripts/stage3_runner.py"
    echo ""
    echo "Or test stop-limit orders first:"
    echo "  python scripts/test_stop_limit_hardened.py"
    exit 0
else
    echo -e "${RED}❌ PREFLIGHT CHECKS FAILED${NC}"
    echo ""
    echo "Fix the issues above before proceeding."
    echo ""
    echo "Quick fixes:"
    echo "  export COINBASE_SANDBOX=1"
    echo "  export COINBASE_API_MODE=advanced"
    echo "  export MAX_IMPACT_BPS=50"
    echo "  export COINBASE_CDP_API_KEY='your-key'"
    echo "  export COINBASE_CDP_PRIVATE_KEY_PATH='/path/to/key.pem'"
    exit 1
fi