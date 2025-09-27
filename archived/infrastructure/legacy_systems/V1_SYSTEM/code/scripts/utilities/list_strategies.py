#!/usr/bin/env python3
"""List all available strategies and their test status."""

import os
import sys
from pathlib import Path
import importlib.util
import json
import logging

# Suppress logging during imports
logging.getLogger().setLevel(logging.CRITICAL)

def check_strategy_file(file_path):
    """Check if a strategy file contains a valid strategy class."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            # Look for class definitions that inherit from Strategy
            for line in content.split('\n'):
                line = line.strip()
                if line.startswith('class ') and '(Strategy' in line:
                    # Extract the class name  
                    class_name = line.split('class ')[1].split('(')[0].strip()
                    return class_name
        return None
    except Exception as e:
        return None

def main():
    """List all strategies and their status."""
    
    strategy_dir = Path("src/bot/strategy")
    test_dir = Path("tests/unit/strategy")
    
    # Load current status
    with open(".knowledge/PROJECT_STATE.json") as f:
        state = json.load(f)
    working = state['components']['strategies']['working_strategies']
    non_working = state['components']['strategies'].get('non_working', [])
    
    strategies = []
    
    # Find all strategy files
    for file in sorted(strategy_dir.glob("*.py")):
        if file.name in ['__init__.py', 'base.py', 'components.py', 'persistence.py']:
            continue
            
        strategy_name = file.stem
        test_file = test_dir / f"test_{strategy_name}.py"
        
        # Check if it's a real strategy
        class_name = check_strategy_file(file)
        if not class_name:
            continue
            
        # Determine status
        if strategy_name in working:
            status = "✅ Working"
        elif any(strategy_name in s for s in non_working):
            status = "❌ Not Connected"
        else:
            status = "❓ Unknown"
        
        # Check for tests
        has_tests = "✅" if test_file.exists() else "❌"
        
        strategies.append({
            'name': strategy_name,
            'class': class_name,
            'status': status,
            'has_tests': has_tests,
            'file': str(file)
        })
    
    # Print results
    print("\n" + "="*80)
    print(" "*25 + "GPT-TRADER STRATEGY INVENTORY")
    print("="*80)
    
    print(f"\n📊 Summary: {len(strategies)} strategies found, {len(working)} working\n")
    
    print(f"{'Strategy':<20} {'Class':<25} {'Status':<15} {'Tests':<8} {'Location'}")
    print("-"*80)
    
    for s in strategies:
        print(f"{s['name']:<20} {s['class']:<25} {s['status']:<15} {s['has_tests']:<8} {s['file']}")
    
    print("\n" + "="*80)
    print("\n🎯 Quick Actions:")
    print("  - Test a strategy: poetry run gpt-trader backtest --strategy <name>")
    print("  - Run strategy tests: poetry run pytest tests/unit/strategy/test_<name>.py")
    print("  - Import strategy: from bot.strategy.<name> import <ClassName>")
    print()

if __name__ == "__main__":
    main()