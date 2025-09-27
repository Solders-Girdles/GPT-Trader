#!/usr/bin/env python3
"""
Test Violation Trigger

Simulates a guard violation to test the alert system in canary_monitor.
Temporarily lowers thresholds to trigger violations safely.

Usage:
  poetry run python scripts/test_violation_trigger.py
"""

import asyncio
import sys
from datetime import datetime, timedelta, timezone


async def test_violation():
    """Test violation trigger with lowered thresholds"""
    # Import with modified thresholds
    from scripts.canary_monitor import CanaryMonitor, GuardThresholds
    
    print("🧪 Testing Violation Trigger")
    print("=" * 50)
    
    # Create monitor with very low thresholds for testing
    monitor = CanaryMonitor(
        profile="test",
        duration_minutes=1,
        dry_run=True  # Keep dry-run for safety
    )
    
    # Override thresholds to trigger easily
    monitor.guards.max_latency_ms = 1.0  # 1ms - will trigger immediately
    monitor.guards.max_orders_per_min = 1  # Very low
    
    # Setup client and alerts
    try:
        monitor.setup_client()
        print("✅ Client setup complete")
        
        if monitor.alert_dispatcher:
            print("✅ Alert dispatcher configured")
        else:
            print("⚠️  No alert channels configured")
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return 1
    
    # Trigger violations
    print("\n📤 Triggering test violations...")
    
    # First violation (WARNING)
    monitor.trigger_violation("TEST: High latency detected (1000ms > 1ms)")
    await asyncio.sleep(1)
    
    # Second violation (should be CRITICAL)
    monitor.trigger_violation("TEST: Order rate limit exceeded (50 > 1)")
    await asyncio.sleep(1)
    
    # Third violation (would trigger kill switch if not dry-run)
    monitor.trigger_violation("TEST: Position size limit exceeded")
    
    print(f"\n📊 Test Results:")
    print(f"  Violations triggered: {len(monitor.metrics['violations'])}")
    print(f"  Kill switch triggered: {monitor.kill_switch_triggered}")
    print(f"  Dry run mode: {monitor.dry_run}")
    
    if monitor.alert_dispatcher:
        print("\n✅ Check your Slack/PagerDuty channels for alerts")
    
    return 0


def main():
    """Main entry point"""
    print("\n=== Violation Trigger Test ===")
    print("This will simulate guard violations to test the alert system.")
    print("The monitor runs in dry-run mode for safety.\n")
    
    result = asyncio.run(test_violation())
    return result


if __name__ == "__main__":
    sys.exit(main())