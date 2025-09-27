#!/usr/bin/env python3
"""
Emergency Foundation Fixes
==========================
Fix the critical broken components discovered in assessment.
"""

import sys
from pathlib import Path

def fix_signal_generation():
    """Fix case sensitivity in strategies"""
    
    print("🔧 Fixing signal generation case sensitivity...")
    
    fixes = [
        ("src/bot/strategy/demo_ma.py", [
            ("data['Close']", "data['close']"),
            ("data['Volume']", "data['volume']"),
            ("data['High']", "data['high']"),
            ("data['Low']", "data['low']"),
            ("data['Open']", "data['open']"),
        ]),
        ("src/bot/strategy/trend_breakout.py", [
            ("data['High']", "data['high']"),
            ("data['Low']", "data['low']"),
            ("data['Close']", "data['close']"),
            ("data['Volume']", "data['volume']"),
        ]),
    ]
    
    for filepath, replacements in fixes:
        file = Path(filepath)
        if file.exists():
            content = file.read_text()
            original = content
            
            for old, new in replacements:
                content = content.replace(old, new)
            
            if content != original:
                file.write_text(content)
                print(f"  ✅ Fixed {filepath}")
            else:
                print(f"  ℹ️ No changes needed in {filepath}")


def fix_imports():
    """Fix wrong class imports"""
    
    print("\n🔧 Checking actual class names...")
    
    # Check what actually exists
    checks = [
        ("src/bot/portfolio/allocator.py", "Allocator"),
        ("src/bot/risk/simple_risk_manager.py", "RiskManager"),
        ("src/bot/integration/orchestrator.py", "IntegratedOrchestrator"),
        ("src/bot/paper_trading/paper_trading_engine.py", "PaperTradingEngine"),
    ]
    
    for filepath, expected_class in checks:
        file = Path(filepath)
        if file.exists():
            content = file.read_text()
            
            # Find actual class name
            import re
            class_pattern = r"class\s+(\w+)"
            matches = re.findall(class_pattern, content)
            
            if matches:
                print(f"  📍 {filepath}")
                print(f"     Expected: {expected_class}")
                print(f"     Found: {', '.join(matches)}")
                
                # Check __init__ signature for PaperTradingEngine
                if "PaperTradingEngine" in matches:
                    init_pattern = r"def __init__\(self[^)]*\)"
                    init_matches = re.findall(init_pattern, content)
                    if init_matches:
                        print(f"     __init__ signature: {init_matches[0][:60]}...")


def check_method_existence():
    """Check if methods exist"""
    
    print("\n🔧 Checking method existence...")
    
    methods = [
        ("src/bot/integration/orchestrator.py", "run_integrated_backtest"),
        ("src/bot/integration/orchestrator.py", "run_backtest"),
        ("src/bot/portfolio/allocator.py", "allocate_signals"),
    ]
    
    for filepath, method_name in methods:
        file = Path(filepath)
        if file.exists():
            content = file.read_text()
            
            if f"def {method_name}" in content:
                print(f"  ✅ {filepath}: {method_name} exists")
            else:
                print(f"  ❌ {filepath}: {method_name} NOT FOUND")
                
                # Look for similar methods
                import re
                method_pattern = r"def\s+(\w*" + method_name[:5] + r"\w*)"
                similar = re.findall(method_pattern, content)
                if similar:
                    print(f"     Similar methods: {', '.join(set(similar))}")


def generate_fix_script():
    """Generate a script with all fixes needed"""
    
    print("\n📝 Generating fix script...")
    
    fixes_needed = """
# Foundation Fixes Required

## 1. Signal Generation
- Change all DataFrame column access from Title Case to lowercase
- Files: demo_ma.py, trend_breakout.py
- Fix: 'Close' → 'close', 'High' → 'high', etc.

## 2. Import Names
- bot.portfolio.allocator: Use 'Allocator' not 'PortfolioAllocator'
- bot.risk.simple_risk_manager: Use 'SimplifiedRiskManager' not 'SimpleRiskManager'

## 3. Method Names
- IntegratedOrchestrator: Use 'run_backtest' not 'run_integrated_backtest'

## 4. Paper Trading
- PaperTradingEngine: Check __init__ parameters (may not accept initial_capital)

## 5. Test Files
- 113 test collection errors - need systematic fix of all test imports
"""
    
    fix_file = Path("FOUNDATION_FIXES_NEEDED.md")
    fix_file.write_text(fixes_needed)
    print(f"  ✅ Fix list saved to: {fix_file}")
    
    return fixes_needed


def main():
    print("="*60)
    print("🚨 EMERGENCY FOUNDATION REPAIR")
    print("="*60)
    print()
    
    # Run fixes
    fix_signal_generation()
    fix_imports()
    check_method_existence()
    fixes = generate_fix_script()
    
    print("\n" + "="*60)
    print("📊 REPAIR SUMMARY")
    print("="*60)
    
    print(fixes)
    
    print("\n🔴 CRITICAL: The foundation is severely broken.")
    print("   We must fix these issues before ANY new development.")
    print("   Run 'poetry run python scripts/foundation_assessment.py' after fixes")
    print("   to verify improvements.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())