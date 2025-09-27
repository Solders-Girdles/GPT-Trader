#!/usr/bin/env python3
"""
Post-Merge Monitoring Script for v2.1.0
Monitors key metrics and reports status after deployment.
"""

import time
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def check_performance_features() -> Dict[str, bool]:
    """Verify performance features are working."""
    results = {}
    
    try:
        from bot_v2.features.brokerages.coinbase.client import CoinbaseClient
        
        # Check keep-alive
        client = CoinbaseClient(base_url="https://api.coinbase.com", enable_keep_alive=True)
        results['keep_alive_enabled'] = client.enable_keep_alive
        results['opener_created'] = client._opener is not None
        
        # Check jitter configuration
        from bot_v2.config import get_config
        sys_cfg = get_config("system")
        jitter = float(sys_cfg.get("jitter_factor", 0.1))
        results['jitter_configured'] = jitter > 0
        results['jitter_value'] = jitter
        
    except Exception as e:
        results['error'] = str(e)
        
    return results

def check_type_consolidation() -> Dict[str, bool]:
    """Verify type consolidation is working."""
    results = {}
    
    try:
        # Try importing from core (should work)
        from bot_v2.features.brokerages.core.interfaces import Order, OrderStatus
        results['core_imports_work'] = True
        
        # Check if deprecated imports show warning
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from bot_v2.features.live_trade.types import Order as DeprecatedOrder
            results['deprecation_warning_shown'] = len(w) > 0 and issubclass(w[0].category, DeprecationWarning)
            
    except Exception as e:
        results['error'] = str(e)
        
    return results

def check_rate_limiting() -> Dict[str, any]:
    """Check rate limiting configuration."""
    results = {}
    
    try:
        from bot_v2.features.brokerages.coinbase.client import CoinbaseClient
        
        client = CoinbaseClient(base_url="https://api.coinbase.com")
        results['rate_limit_per_minute'] = client.rate_limit_per_minute
        results['throttling_enabled'] = client.enable_throttle
        results['request_count'] = client._request_count
        results['warning_threshold'] = int(client.rate_limit_per_minute * 0.8)
        
    except Exception as e:
        results['error'] = str(e)
        
    return results

def run_test_suite() -> Tuple[bool, List[str]]:
    """Run key tests and return results."""
    import subprocess
    
    tests = [
        ("Performance tests", "python -m pytest tests/unit/bot_v2/features/brokerages/coinbase/test_performance.py -q"),
        ("Integration tests", "python -m pytest tests/integration/bot_v2/test_live_trade_error_handling.py -q"),
        ("Critical fixes", "python scripts/validate_critical_fixes.py")
    ]
    
    results = []
    all_passed = True
    
    for name, cmd in tests:
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                results.append(f"✅ {name}: PASSED")
            else:
                results.append(f"❌ {name}: FAILED")
                all_passed = False
        except subprocess.TimeoutExpired:
            results.append(f"⚠️  {name}: TIMEOUT")
            all_passed = False
        except Exception as e:
            results.append(f"❌ {name}: ERROR - {e}")
            all_passed = False
            
    return all_passed, results

def generate_report():
    """Generate monitoring report."""
    print("\n" + "="*60)
    print("   POST-MERGE MONITORING REPORT - v2.1.0")
    print("="*60)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Performance Features
    print("\n📊 Performance Features:")
    perf = check_performance_features()
    if 'error' in perf:
        print(f"   ❌ Error: {perf['error']}")
    else:
        print(f"   • Keep-Alive: {'✅ Enabled' if perf.get('keep_alive_enabled') else '❌ Disabled'}")
        print(f"   • Connection Pooling: {'✅ Active' if perf.get('opener_created') else '❌ Inactive'}")
        print(f"   • Jitter Factor: {perf.get('jitter_value', 'N/A')}")
    
    # Type Consolidation
    print("\n🔧 Type Consolidation:")
    types = check_type_consolidation()
    if 'error' in types:
        print(f"   ❌ Error: {types['error']}")
    else:
        print(f"   • Core Imports: {'✅ Working' if types.get('core_imports_work') else '❌ Failed'}")
        print(f"   • Deprecation Warning: {'✅ Active' if types.get('deprecation_warning_shown') else '⚠️ Not shown'}")
    
    # Rate Limiting
    print("\n⚡ Rate Limiting:")
    rate = check_rate_limiting()
    if 'error' in rate:
        print(f"   ❌ Error: {rate['error']}")
    else:
        print(f"   • Limit: {rate.get('rate_limit_per_minute', 'N/A')} req/min")
        print(f"   • Warning at: {rate.get('warning_threshold', 'N/A')} requests")
        print(f"   • Throttling: {'✅ Enabled' if rate.get('throttling_enabled') else '❌ Disabled'}")
    
    # Test Suite
    print("\n🧪 Test Suite:")
    all_passed, test_results = run_test_suite()
    for result in test_results:
        print(f"   {result}")
    
    # Recommendations
    print("\n💡 Recommendations:")
    if all_passed and perf.get('keep_alive_enabled') and types.get('core_imports_work'):
        print("   ✅ All systems operational - continue monitoring")
    else:
        print("   ⚠️  Issues detected:")
        if not perf.get('keep_alive_enabled'):
            print("      • Consider enabling keep-alive for better performance")
        if not all_passed:
            print("      • Investigate failing tests")
            
    # Rollback Commands
    print("\n🔄 Quick Rollback (if needed):")
    print("   export COINBASE_ENABLE_KEEP_ALIVE=0  # Disable keep-alive")
    print("   export COINBASE_JITTER_FACTOR=0      # Disable jitter")
    
    print("\n" + "="*60)
    print("   Report Complete")
    print("="*60 + "\n")

def continuous_monitor(duration_hours: int = 24):
    """Run continuous monitoring for specified duration."""
    print(f"\n🔍 Starting continuous monitoring for {duration_hours} hours...")
    print("Press Ctrl+C to stop\n")
    
    end_time = datetime.now() + timedelta(hours=duration_hours)
    check_interval = 3600  # Check every hour
    
    try:
        while datetime.now() < end_time:
            generate_report()
            
            remaining = (end_time - datetime.now()).total_seconds()
            if remaining > check_interval:
                print(f"Next check in {check_interval//60} minutes...")
                time.sleep(check_interval)
            else:
                break
                
    except KeyboardInterrupt:
        print("\n\n⏹️  Monitoring stopped by user")
        
    print("\n✅ Monitoring complete")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Post-merge monitoring for v2.1.0")
    parser.add_argument("--continuous", "-c", type=int, metavar="HOURS",
                       help="Run continuous monitoring for N hours")
    parser.add_argument("--once", "-o", action="store_true",
                       help="Run single check (default)")
    
    args = parser.parse_args()
    
    if args.continuous:
        continuous_monitor(args.continuous)
    else:
        generate_report()