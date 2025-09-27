#!/bin/bash
# Merge Checklist Script for v2.1.0
# Run this script to validate readiness and guide through merge process

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "======================================"
echo "   GPT-Trader v2.1.0 Merge Checklist  "
echo "======================================"
echo ""

# Function to check and report status
check_step() {
    local description="$1"
    local command="$2"
    
    echo -n "[ ] $description... "
    
    if eval "$command" > /dev/null 2>&1; then
        echo -e "${GREEN}✅${NC}"
        return 0
    else
        echo -e "${RED}❌${NC}"
        return 1
    fi
}

# Function for manual confirmation
confirm_step() {
    local description="$1"
    echo -n "[ ] $description (y/n)? "
    read -r response
    if [[ "$response" == "y" ]]; then
        echo -e "    ${GREEN}✅ Confirmed${NC}"
        return 0
    else
        echo -e "    ${YELLOW}⚠️  Skipped${NC}"
        return 1
    fi
}

echo "=== Pre-Merge Validation ==="
echo ""

# Automated checks
check_step "Type consolidation: No deprecated imports" \
    "! rg 'from .*live_trade\.types import' tests/ --type py"

check_step "Performance tests pass" \
    "python -m pytest tests/unit/bot_v2/features/brokerages/coinbase/test_performance.py -q"

check_step "Integration tests pass" \
    "python -m pytest tests/integration/bot_v2/test_live_trade_error_handling.py -q"

check_step "Critical fixes validated" \
    "python scripts/validate_critical_fixes.py > /dev/null 2>&1"

check_step ".env.template has performance settings" \
    "grep -q 'COINBASE_ENABLE_KEEP_ALIVE' .env.template"

check_step "Release notes exist" \
    "test -f RELEASE_NOTES_v2.1.0.md"

check_step "CHANGELOG updated" \
    "grep -q '2.1.0' CHANGELOG.md"

check_step "Coinbase README exists" \
    "test -f docs/COINBASE_README.md"

echo ""
echo "=== Manual Verification Steps ==="
echo ""

confirm_step "Have you reviewed all code changes"
confirm_step "Are CI checks green on the branch"
confirm_step "Have you backed up current deployment"

echo ""
echo "=== Smoke Tests ==="
echo ""

echo "Running local smoke tests..."
echo ""

# Test 1: Validate critical fixes
echo "1. Critical fixes validation:"
if python scripts/validate_critical_fixes.py 2>&1 | tail -3 | grep -q "SUCCESSFULLY"; then
    echo -e "   ${GREEN}✅ All critical fixes validated${NC}"
else
    echo -e "   ${RED}❌ Critical fixes validation failed${NC}"
fi

# Test 2: Check imports
echo ""
echo "2. Import verification:"
python -c "
from bot_v2.features.brokerages.core.interfaces import Order, OrderStatus
from bot_v2.features.brokerages.coinbase.client import CoinbaseClient
print('   ✅ Core imports working')
" 2>/dev/null || echo -e "   ${RED}❌ Import error${NC}"

# Test 3: Performance features
echo ""
echo "3. Performance features check:"
python -c "
from bot_v2.features.brokerages.coinbase.client import CoinbaseClient
client = CoinbaseClient(base_url='https://api.coinbase.com', enable_keep_alive=True)
assert client.enable_keep_alive == True
assert client._opener is not None
print('   ✅ Keep-alive enabled and working')
" 2>/dev/null || echo -e "   ${RED}❌ Performance features error${NC}"

echo ""
echo "=== Merge Steps ==="
echo ""
echo "If all checks pass, execute these commands:"
echo ""
echo -e "${YELLOW}# 1. Create and push tag:${NC}"
echo "   git tag -a v2.1.0 -m \"Type consolidation and performance optimizations\""
echo "   git push origin v2.1.0"
echo ""
echo -e "${YELLOW}# 2. Create GitHub release:${NC}"
echo "   gh release create v2.1.0 --title \"v2.1.0 - Performance & Type Consolidation\" \\"
echo "     --notes-file RELEASE_NOTES_v2.1.0.md"
echo ""
echo -e "${YELLOW}# 3. Merge to main:${NC}"
echo "   git checkout main"
echo "   git merge feat/qol-progress-logging"
echo "   git push origin main"
echo ""

echo "=== Post-Merge Monitoring ==="
echo ""
echo "Monitor these metrics for 24-48 hours:"
echo "  • API latency (should decrease by 20-40ms)"
echo "  • Rate limit warnings (should stay <80%)"
echo "  • Connection errors (should be stable/decrease)"
echo "  • WebSocket reconnection frequency"
echo ""

echo "=== Quick Rollback Commands ==="
echo ""
echo -e "${YELLOW}Feature disable (no code change):${NC}"
echo "   export COINBASE_ENABLE_KEEP_ALIVE=0"
echo "   export COINBASE_JITTER_FACTOR=0"
echo ""
echo -e "${YELLOW}Code rollback:${NC}"
echo "   git revert HEAD~2  # Revert both PRs"
echo ""

echo "======================================"
echo -e "${GREEN}Checklist Complete!${NC}"
echo "======================================