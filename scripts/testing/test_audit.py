#!/usr/bin/env python3
"""Test suite audit - identify what we claim vs what we have vs what we need."""

import subprocess
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple
import json

def run_pytest_collect(path: str) -> Tuple[int, List[str]]:
    """Collect tests from path and return count + names."""
    result = subprocess.run(
        ["poetry", "run", "pytest", path, "--co", "-q"],
        capture_output=True,
        text=True
    )
    lines = result.stdout.strip().split('\n')
    tests = [l for l in lines if 'test_' in l and '<Function' in l]
    return len(tests), tests

def run_tests(path: str) -> Dict[str, int]:
    """Run tests and get pass/fail/skip counts."""
    result = subprocess.run(
        ["poetry", "run", "pytest", path, "-q", "--tb=no"],
        capture_output=True,
        text=True
    )
    output = result.stdout + result.stderr
    
    # Parse the summary line
    passed = skipped = failed = 0
    for line in output.split('\n'):
        if 'passed' in line or 'failed' in line or 'skipped' in line:
            parts = line.split()
            for i, part in enumerate(parts):
                if 'passed' in part and i > 0:
                    passed = int(parts[i-1].replace('=', ''))
                if 'failed' in part and i > 0:
                    failed = int(parts[i-1].replace('=', ''))
                if 'skipped' in part and i > 0:
                    skipped = int(parts[i-1].replace('=', ''))
    
    return {'passed': passed, 'failed': failed, 'skipped': skipped}

def find_missing_tests() -> Dict[str, List[str]]:
    """Identify critical components without tests."""
    missing = {}
    
    # Critical source files that should have tests
    critical_files = [
        'src/bot_v2/orchestration/perps_bot.py',
        'src/bot_v2/orchestration/live_execution.py',
        'src/bot_v2/features/live_trade/execution_v3.py',
        'src/bot_v2/features/live_trade/pnl_tracker.py',
        'src/bot_v2/features/live_trade/risk.py',
        'src/bot_v2/features/brokerages/coinbase/adapter.py',
        'src/bot_v2/features/brokerages/coinbase/ws.py',
        'src/bot_v2/features/brokerages/coinbase/client.py',
    ]
    
    for src_file in critical_files:
        # Look for corresponding test file
        test_file = src_file.replace('src/', 'tests/unit/').replace('.py', '_test.py')
        alt_test_file = src_file.replace('src/', 'tests/unit/').replace('.py', '.py')
        test_prefix = src_file.replace('src/', 'tests/unit/test_').replace('.py', '.py')
        
        if not (Path(test_file).exists() or Path(alt_test_file).exists() or Path(test_prefix).exists()):
            missing[src_file] = [test_file]
    
    return missing

def main():
    print("=" * 80)
    print("TEST SUITE AUDIT REPORT")
    print("=" * 80)
    
    # 1. Overall test counts
    print("\n📊 TEST INVENTORY")
    print("-" * 40)
    
    test_dirs = {
        'Unit Tests': 'tests/unit',
        'Integration Tests': 'tests/integration',
        'Bot V2 Tests': 'tests/bot_v2',
    }
    
    total_collected = 0
    for name, path in test_dirs.items():
        if Path(path).exists():
            count, _ = run_pytest_collect(path)
            total_collected += count
            print(f"{name:20}: {count:4} tests")
    
    print(f"{'TOTAL':20}: {total_collected:4} tests")
    
    # 2. Test execution status
    print("\n🔴 TEST EXECUTION STATUS")
    print("-" * 40)
    
    critical_paths = [
        ('Foundation', 'tests/unit/test_foundation.py'),
        ('Perps Bot', 'tests/integration/bot_v2/perps/'),
        ('Coinbase Adapter', 'tests/unit/bot_v2/features/brokerages/coinbase/'),
        ('Live Trade', 'tests/unit/bot_v2/features/live_trade/'),
    ]
    
    for name, path in critical_paths:
        if Path(path).exists():
            stats = run_tests(path)
            total = stats['passed'] + stats['failed'] + stats['skipped']
            pass_rate = (stats['passed'] / total * 100) if total > 0 else 0
            
            status = "✅" if stats['failed'] == 0 else "❌"
            print(f"{status} {name:20}: {stats['passed']:3} pass, {stats['failed']:3} fail, {stats['skipped']:3} skip ({pass_rate:.0f}%)")
    
    # 3. Missing critical tests
    print("\n⚠️  MISSING CRITICAL TESTS")
    print("-" * 40)
    
    missing = find_missing_tests()
    if missing:
        for src, test_files in missing.items():
            print(f"❌ {src}")
            for tf in test_files:
                print(f"   → Missing: {tf}")
    else:
        print("✅ All critical files have tests")
    
    # 4. Coverage gaps analysis
    print("\n🎯 CRITICAL COVERAGE GAPS")
    print("-" * 40)
    
    gaps = [
        "WebSocket streaming (stream_trades, stream_orderbook)",
        "Position state reconciliation",
        "Order lifecycle (partial fills, modifications)",
        "PnL calculation with funding",
        "State recovery after restart",
        "Risk limit enforcement",
        "Error recovery and reconnection",
        "Canary profile loading",
        "Multi-threaded mark updates",
        "Circuit breaker triggers"
    ]
    
    for gap in gaps:
        print(f"  • {gap}")
    
    # 5. Recommendations
    print("\n📝 RECOMMENDATIONS")
    print("-" * 40)
    
    recommendations = [
        "1. Fix failing tests before adding new ones",
        "2. Add integration tests for complete order flow",
        "3. Create state persistence/recovery tests",
        "4. Test WebSocket reconnection scenarios",
        "5. Add performance benchmarks for critical paths",
        "6. Create failure mode tests (API errors, timeouts)",
        "7. Test multi-symbol position management",
        "8. Verify risk calculations with real scenarios",
    ]
    
    for rec in recommendations:
        print(f"  {rec}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()