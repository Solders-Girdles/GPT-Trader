#!/bin/bash

# Repository Structure Fix Script
# This script implements the fixes identified in the audit report
# Run with: bash scripts/fix_repo_structure.sh [phase]

set -e

PHASE=${1:-all}

echo "🔧 Repository Structure Fix Script"
echo "=================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Phase 1: Critical Configuration Fixes
phase1_config_fixes() {
    echo -e "${YELLOW}Phase 1: Configuration Fixes${NC}"
    
    # Fix pytest.ini - add pythonpath
    if ! grep -q "^pythonpath = src" pytest.ini; then
        echo "Adding pythonpath to pytest.ini..."
        # Insert after line 8 (after python_functions)
        sed -i '' '8a\
\
# Python path configuration\
pythonpath = src\
' pytest.ini
        echo -e "${GREEN}✓ Fixed pytest.ini${NC}"
    else
        echo "pytest.ini already has pythonpath configured"
    fi
    
    # Create missing __init__.py
    if [ ! -f "src/bot_v2/__init__.py" ]; then
        echo "Creating src/bot_v2/__init__.py..."
        touch src/bot_v2/__init__.py
        echo -e "${GREEN}✓ Created src/bot_v2/__init__.py${NC}"
    else
        echo "src/bot_v2/__init__.py already exists"
    fi
    
    # Verify the changes
    echo "Verifying configuration..."
    python -c "import sys; sys.path.insert(0, 'src'); from bot_v2 import *; print('✓ Package import works')" 2>/dev/null || echo -e "${RED}⚠ Package import still has issues${NC}"
}

# Phase 2: Production Import Fixes
phase2_production_imports() {
    echo -e "${YELLOW}Phase 2: Production Import Fixes${NC}"
    
    # Critical production files
    PROD_FILES=(
        "scripts/run_perps_bot.py"
        "scripts/run_perps_bot_v2.py"
        "scripts/stage3_runner.py"
        "src/bot_v2/__main__.py"
        "src/bot_v2/orchestration/bot_manager.py"
    )
    
    for file in "${PROD_FILES[@]}"; do
        if [ -f "$file" ]; then
            echo "Fixing imports in $file..."
            sed -i '' 's/from src\.bot_v2/from bot_v2/g' "$file"
            sed -i '' 's/import src\.bot_v2/import bot_v2/g' "$file"
            echo -e "${GREEN}✓ Fixed $file${NC}"
        fi
    done
}

# Phase 3: Test Import Fixes
phase3_test_imports() {
    echo -e "${YELLOW}Phase 3: Test Import Fixes${NC}"
    
    # Fix all test imports
    echo "Fixing test imports..."
    find tests -type f -name "*.py" -exec sed -i '' 's/from src\.bot_v2/from bot_v2/g' {} \;
    find tests -type f -name "*.py" -exec sed -i '' 's/import src\.bot_v2/import bot_v2/g' {} \;
    echo -e "${GREEN}✓ Fixed test imports${NC}"
}

# Phase 4: Scripts Import Fixes
phase4_scripts_imports() {
    echo -e "${YELLOW}Phase 4: Scripts Import Fixes${NC}"
    
    # Fix all script imports
    echo "Fixing script imports..."
    find scripts -type f -name "*.py" -exec sed -i '' 's/from src\.bot_v2/from bot_v2/g' {} \;
    find scripts -type f -name "*.py" -exec sed -i '' 's/import src\.bot_v2/import bot_v2/g' {} \;
    echo -e "${GREEN}✓ Fixed script imports${NC}"
}

# Phase 5: Root Cleanup
phase5_root_cleanup() {
    echo -e "${YELLOW}Phase 5: Root Cleanup${NC}"
    
    # Create target directories
    mkdir -p scripts/utils scripts/env results verification_reports
    
    # Move utility scripts
    UTIL_SCRIPTS=(
        "add_legacy_credentials.py"
        "create_prod_config.py"
        "debug_permissions.py"
        "setup_api_keys.py"
        "setup_complete_api_keys.py"
        "setup_legacy_hmac.py"
        "update_legacy_config.py"
    )
    
    for script in "${UTIL_SCRIPTS[@]}"; do
        if [ -f "$script" ]; then
            echo "Moving $script to scripts/utils/"
            git mv "$script" "scripts/utils/" 2>/dev/null || mv "$script" "scripts/utils/"
        fi
    done
    
    # Move test files
    TEST_FILES=(
        "test_cdp_comprehensive.py"
        "test_current_setup.py"
        "test_full_cdp.py"
        "test_reality_check.py"
    )
    
    for test in "${TEST_FILES[@]}"; do
        if [ -f "$test" ]; then
            echo "Moving $test to tests/integration/"
            git mv "$test" "tests/integration/" 2>/dev/null || mv "$test" "tests/integration/"
        fi
    done
    
    # Move validation outputs
    echo "Moving validation outputs..."
    mv demo_validation_*.json verification_reports/ 2>/dev/null || true
    
    # Move environment scripts
    echo "Moving environment scripts..."
    for env_script in set_env*.sh; do
        if [ -f "$env_script" ]; then
            git mv "$env_script" "scripts/env/" 2>/dev/null || mv "$env_script" "scripts/env/"
        fi
    done
    
    echo -e "${GREEN}✓ Root cleanup complete${NC}"
}

# Verification function
verify_changes() {
    echo -e "${YELLOW}Verification${NC}"
    
    # Check for remaining src. imports
    echo -n "Checking for remaining 'src.' imports... "
    count=$(grep -r "from src\." --include="*.py" --exclude-dir=archived . 2>/dev/null | wc -l)
    if [ "$count" -eq 0 ]; then
        echo -e "${GREEN}✓ No 'src.' imports found${NC}"
    else
        echo -e "${RED}⚠ Found $count files with 'src.' imports${NC}"
    fi
    
    # Test pytest discovery
    echo -n "Testing pytest discovery... "
    if pytest --collect-only -q 2>/dev/null; then
        echo -e "${GREEN}✓ Pytest discovery works${NC}"
    else
        echo -e "${RED}⚠ Pytest discovery has issues${NC}"
    fi
    
    # Test package import
    echo -n "Testing package import... "
    if python -c "from bot_v2.features.live_trade import *" 2>/dev/null; then
        echo -e "${GREEN}✓ Package import works${NC}"
    else
        echo -e "${RED}⚠ Package import has issues${NC}"
    fi
}

# Main execution
main() {
    case $PHASE in
        1|phase1)
            phase1_config_fixes
            ;;
        2|phase2)
            phase2_production_imports
            ;;
        3|phase3)
            phase3_test_imports
            ;;
        4|phase4)
            phase4_scripts_imports
            ;;
        5|phase5)
            phase5_root_cleanup
            ;;
        all)
            phase1_config_fixes
            phase2_production_imports
            phase3_test_imports
            phase4_scripts_imports
            # phase5_root_cleanup  # Commented out by default as it moves files
            verify_changes
            ;;
        verify)
            verify_changes
            ;;
        *)
            echo "Usage: $0 [1|2|3|4|5|all|verify]"
            echo "  1: Configuration fixes (pytest.ini, __init__.py)"
            echo "  2: Production import fixes"
            echo "  3: Test import fixes"
            echo "  4: Scripts import fixes"
            echo "  5: Root cleanup (moves files)"
            echo "  all: Run phases 1-4 (not 5 by default)"
            echo "  verify: Verify changes"
            exit 1
            ;;
    esac
}

main

echo -e "${GREEN}✅ Repository structure fix complete!${NC}"
echo ""
echo "Next steps:"
echo "1. Review changes with: git diff"
echo "2. Run tests with: pytest"
echo "3. Test runner with: python scripts/run_perps_bot.py --profile dev --dev-fast --dry-run"
echo "4. Commit changes when satisfied"
