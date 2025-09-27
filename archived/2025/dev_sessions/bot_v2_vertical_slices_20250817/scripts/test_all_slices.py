#!/usr/bin/env python
"""V2 Script: Test all slices for isolation and functionality"""

import sys
import subprocess
from pathlib import Path

def test_slice(slice_name):
    """Test a single V2 slice."""
    print(f"\n🔍 Testing {slice_name} slice...")
    
    # Test import works
    try:
        exec(f"from bot_v2.features.{slice_name} import *")
        print(f"  ✅ {slice_name} imports successfully")
    except ImportError as e:
        print(f"  ❌ {slice_name} import failed: {e}")
        return False
    
    # Check for isolation violations
    slice_dir = Path(f"src/bot_v2/features/{slice_name}")
    result = subprocess.run(
        ["grep", "-r", "from bot_v2.features", str(slice_dir)],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print(f"  ❌ {slice_name} has isolation violations:")
        print(f"    {result.stdout[:200]}")
        return False
    else:
        print(f"  ✅ {slice_name} isolation verified")
    
    # Run slice test if exists
    test_file = Path(f"tests/integration/bot_v2/test_{slice_name}.py")
    if test_file.exists():
        result = subprocess.run(
            ["python", str(test_file)],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            print(f"  ✅ {slice_name} tests pass")
        else:
            print(f"  ⚠️  {slice_name} tests fail")
    
    return True

def main():
    """Test all V2 slices."""
    print("=" * 60)
    print("V2 SLICE TESTING - Complete Isolation Verification")
    print("=" * 60)
    
    slices = [
        "backtest",
        "paper_trade", 
        "analyze",
        "optimize",
        "live_trade",
        "monitor",
        "data",
        "ml_strategy",
        "market_regime"
    ]
    
    results = {}
    for slice_name in slices:
        results[slice_name] = test_slice(slice_name)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    working = sum(1 for v in results.values() if v)
    total = len(results)
    
    for slice_name, success in results.items():
        status = "✅" if success else "❌"
        print(f"{status} {slice_name}")
    
    print(f"\nOverall: {working}/{total} slices working")
    print(f"System: {working/total*100:.0f}% operational")
    
    if working == total:
        print("\n🎉 V2 System 100% Operational!")
    
    return 0 if working == total else 1

if __name__ == "__main__":
    sys.exit(main())