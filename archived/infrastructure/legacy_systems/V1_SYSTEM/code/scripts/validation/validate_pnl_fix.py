#!/usr/bin/env python3
"""
Validate that P&L calculation is working correctly after fix.

Tests:
1. Run backtests with multiple strategies
2. Verify non-zero returns
3. Check that different strategies produce different results
4. Validate that trades are being executed
"""

import subprocess
import json
import re
from typing import Dict, List, Tuple
import sys

def run_backtest(symbol: str, strategy: str, start: str, end: str) -> Dict:
    """Run a backtest and parse results."""
    cmd = [
        "poetry", "run", "gpt-trader", "backtest",
        "--symbol", symbol,
        "--start", start,
        "--end", end,
        "--strategy", strategy
    ]
    
    print(f"Running: {strategy} on {symbol} from {start} to {end}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Parse output for key metrics
    output = result.stdout + result.stderr
    
    metrics = {}
    
    # Extract Total Return
    match = re.search(r"Total Return[:\s]+(-?\d+\.?\d*)%", output)
    if match:
        metrics['total_return'] = float(match.group(1))
    
    # Extract Sharpe Ratio
    match = re.search(r"Sharpe Ratio[:\s]+(-?\d+\.?\d*)", output)
    if match:
        metrics['sharpe'] = float(match.group(1))
    
    # Extract Max Drawdown
    match = re.search(r"Max Drawdown[:\s]+(-?\d+\.?\d*)%", output)
    if match:
        metrics['max_drawdown'] = float(match.group(1))
    
    # Extract Total Trades
    match = re.search(r"Total Trades[:\s]+(\d+)", output)
    if match:
        metrics['total_trades'] = int(match.group(1))
    
    # Check for allocations
    allocations = len(re.findall(r"Allocated \d+ positions", output))
    metrics['allocations'] = allocations
    
    return metrics


def main():
    """Run comprehensive P&L validation tests."""
    print("=" * 60)
    print("P&L CALCULATION VALIDATION TEST")
    print("=" * 60)
    
    # Test configurations
    tests = [
        ("AAPL", "demo_ma", "2024-01-01", "2024-06-30"),
        ("MSFT", "trend_breakout", "2024-01-01", "2024-06-30"),
        ("GOOGL", "mean_reversion", "2024-01-01", "2024-06-30"),
        ("AAPL", "momentum", "2024-03-01", "2024-06-30"),
        ("AAPL", "volatility", "2024-03-01", "2024-06-30"),
    ]
    
    results = []
    all_passed = True
    
    for symbol, strategy, start, end in tests:
        print(f"\nTest: {strategy} on {symbol}")
        print("-" * 40)
        
        metrics = run_backtest(symbol, strategy, start, end)
        
        # Validate results
        passed = True
        issues = []
        
        # Check 1: Total return should not be exactly 0
        if 'total_return' in metrics:
            if metrics['total_return'] == 0.0:
                issues.append("❌ Total return is exactly 0.00%")
                passed = False
            else:
                print(f"✅ Total return: {metrics['total_return']:.2f}%")
        else:
            issues.append("❌ Could not extract total return")
            passed = False
        
        # Check 2: Should have allocations
        if metrics.get('allocations', 0) > 0:
            print(f"✅ Allocations made: {metrics['allocations']}")
        else:
            issues.append("❌ No allocations made")
            passed = False
        
        # Check 3: Sharpe ratio should exist (can be negative)
        if 'sharpe' in metrics:
            print(f"✅ Sharpe ratio: {metrics['sharpe']:.2f}")
        else:
            issues.append("❌ Could not extract Sharpe ratio")
            passed = False
        
        # Check 4: Max drawdown should exist
        if 'max_drawdown' in metrics:
            print(f"✅ Max drawdown: {metrics['max_drawdown']:.2f}%")
        else:
            issues.append("❌ Could not extract max drawdown")
            passed = False
        
        # Record results
        results.append({
            'strategy': strategy,
            'symbol': symbol,
            'metrics': metrics,
            'passed': passed,
            'issues': issues
        })
        
        if not passed:
            all_passed = False
            print("\nIssues found:")
            for issue in issues:
                print(f"  {issue}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    # Check for variation in returns
    returns = [r['metrics'].get('total_return', 0) for r in results]
    unique_returns = len(set(returns))
    
    print(f"\nTests run: {len(results)}")
    print(f"Tests passed: {sum(1 for r in results if r['passed'])}")
    print(f"Unique return values: {unique_returns}")
    
    # Display all results
    print("\nResults by Strategy:")
    print("-" * 40)
    for r in results:
        status = "✅" if r['passed'] else "❌"
        ret = r['metrics'].get('total_return', 'N/A')
        if isinstance(ret, float):
            ret_str = f"{ret:+.2f}%"
        else:
            ret_str = ret
        print(f"{status} {r['strategy']:20s} on {r['symbol']:5s}: {ret_str}")
    
    # Final verdict
    print("\n" + "=" * 60)
    if all_passed and unique_returns > 1:
        print("✅ P&L CALCULATION IS WORKING!")
        print("   - All strategies produce non-zero returns")
        print("   - Different strategies produce different results")
        print("   - Trades are being executed and tracked")
        return 0
    else:
        print("❌ P&L CALCULATION STILL HAS ISSUES")
        if unique_returns <= 1:
            print("   - All strategies producing same return value")
        if not all_passed:
            print("   - Some tests failed validation")
        return 1


if __name__ == "__main__":
    sys.exit(main())