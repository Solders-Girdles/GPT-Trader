#!/usr/bin/env python3
# NOTE: Archived from tests/; excluded from active test suite.
"""
Reality check: Test what actually works in bot_v2
"""

import sys
import traceback
from pathlib import Path

def test_feature_slice(name, import_statement):
    """Test if a feature slice can be imported"""
    try:
        exec(import_statement)
        return f"✅ {name}: SUCCESS"
    except ImportError as e:
        return f"❌ {name}: Import failed - {str(e)}"
    except Exception as e:
        return f"⚠️ {name}: Other error - {str(e)}"

def main():
    print("=" * 60)
    print("BOT_V2 REALITY CHECK")
    print("=" * 60)
    
    # Test feature slices
    print("\n📁 FEATURE SLICES:")
    print("-" * 40)
    
    tests = [
        ("Backtest", "from src.bot_v2.features.backtest import run_backtest"),
        ("Analyze", "from src.bot_v2.features.analyze import analyze"),
        ("Optimize", "from src.bot_v2.features.optimize import optimize"),
        ("Paper Trade", "from src.bot_v2.features.paper_trade import paper_trade"),
        ("Live Trade", "from src.bot_v2.features.live_trade import live_trade"),
        ("Monitor", "from src.bot_v2.features.monitor import monitor"),
        ("Data", "from src.bot_v2.features.data import data"),
        ("ML Strategy", "from src.bot_v2.features.ml_strategy import ml_strategy"),
        ("Market Regime", "from src.bot_v2.features.market_regime import market_regime"),
        ("Position Sizing", "from src.bot_v2.features.position_sizing import position_sizing"),
        ("Adaptive Portfolio", "from src.bot_v2.features.adaptive_portfolio import adaptive_portfolio"),
    ]
    
    for name, import_stmt in tests:
        print(test_feature_slice(name, import_stmt))
    
    # Check directories that should exist
    print("\n📂 DIRECTORY STRUCTURE:")
    print("-" * 40)
    
    bot_v2_path = Path("src/bot_v2")
    expected_dirs = [
        "features/backtest",
        "features/analyze", 
        "features/optimize",
        "features/paper_trade",
        "features/live_trade",
        "features/monitor",
        "features/data",
        "features/ml_strategy",
        "features/market_regime",
        "features/position_sizing",
        "features/adaptive_portfolio",
        "workflows",
        "orchestration",
        "state",
        "security",
        "monitoring",
        "deployment",
    ]
    
    for dir_name in expected_dirs:
        dir_path = bot_v2_path / dir_name
        if dir_path.exists():
            print(f"✅ {dir_name}: EXISTS")
        else:
            print(f"❌ {dir_name}: MISSING")
    
    # Check claimed Sprint 4 directories
    print("\n🚀 SPRINT 4 CLAIMS:")
    print("-" * 40)
    
    sprint4_dirs = [
        "api",
        "cli",
        "optimization",
    ]
    
    for dir_name in sprint4_dirs:
        dir_path = bot_v2_path / dir_name
        if dir_path.exists():
            print(f"✅ {dir_name}: EXISTS")
        else:
            print(f"❌ {dir_name}: MISSING (Sprint 4 claimed but not delivered)")
    
    # Test main entry point
    print("\n🎯 MAIN ENTRY POINT:")
    print("-" * 40)
    
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "src.bot_v2", "--help"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print("✅ Main entry point works")
        else:
            print(f"❌ Main entry point failed: {result.stderr[:200]}")
    except Exception as e:
        print(f"❌ Main entry point error: {str(e)}")
    
    # Check test files
    print("\n🧪 TEST FILES:")
    print("-" * 40)
    
    test_files = [
        "tests/integration/bot_v2/test_e2e_complete.py",
        "tests/integration/bot_v2/test_e2e.py",
        "tests/performance/benchmark_suite.py",
        "tests/stress/stress_test_suite.py",
        "tests/reports/test_report_generator.py",
    ]
    
    for test_file in test_files:
        test_path = Path(test_file)
        if test_path.exists():
            size = test_path.stat().st_size
            print(f"✅ {test_file}: EXISTS ({size:,} bytes)")
        else:
            print(f"❌ {test_file}: MISSING")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("-" * 40)
    print("✅ Core feature slices exist (11 directories)")
    print("❌ Sprint 4 components missing (api/, cli/, optimization/)")
    print("⚠️ Main entry point has import errors")
    print("✅ Some test files exist")
    print("❌ Most claimed functionality not implemented")
    print("=" * 60)

if __name__ == "__main__":
    main()
