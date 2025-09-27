#!/usr/bin/env python3
"""
Test Alert System

Quick test to verify alert dispatcher works correctly.
Run this to test Slack/PagerDuty integration before going live.

Usage:
  poetry run python scripts/test_alert_system.py
"""

import asyncio
import os
import sys
from datetime import datetime, timezone


async def test_alerts():
    """Test alert system with a simulated violation"""
    from bot_v2.monitoring.alerts import (
        AlertDispatcher,
        AlertSeverity,
        SlackChannel,
        PagerDutyChannel,
        create_system_alert
    )
    
    print("🧪 Testing Alert System")
    print("=" * 50)
    
    dispatcher = AlertDispatcher()
    channels_configured = False
    
    # Check Slack
    if slack_webhook := os.getenv("SLACK_WEBHOOK_URL"):
        print(f"✅ Slack webhook configured")
        slack_channel = SlackChannel(webhook_url=slack_webhook)
        dispatcher.add_channel('slack', slack_channel)
        channels_configured = True
    else:
        print("⚠️  Slack not configured (set SLACK_WEBHOOK_URL)")
    
    # Check PagerDuty (support both env var names)
    pagerduty_key = os.getenv("PAGERDUTY_API_KEY") or os.getenv("PAGERDUTY_ROUTING_KEY")
    if pagerduty_key:
        print(f"✅ PagerDuty API key configured")
        pd_channel = PagerDutyChannel(api_key=pagerduty_key)
        dispatcher.add_channel('pagerduty', pd_channel)
        channels_configured = True
    else:
        print("⚠️  PagerDuty not configured (set PAGERDUTY_API_KEY)")
    
    if not channels_configured:
        print("\n❌ No alert channels configured. Set environment variables and try again.")
        return 1
    
    print("\n📤 Sending test alerts...")
    
    # Test WARNING alert
    warning_alert = create_system_alert(
        title="Test Warning",
        message="Canary Monitor Test (WARNING)",
        severity=AlertSeverity.WARNING,
        test=True,
        timestamp=datetime.now(timezone.utc).isoformat(),
        source="test_alert_system.py"
    )
    
    # Test CRITICAL alert
    critical_alert = create_system_alert(
        title="Test Critical",
        message="Canary Monitor Test (CRITICAL)",
        severity=AlertSeverity.CRITICAL,
        test=True,
        timestamp=datetime.now(timezone.utc).isoformat(),
        source="test_alert_system.py",
        note="This is a test - no action required"
    )
    
    # Dispatch alerts
    try:
        await dispatcher.dispatch(warning_alert)
        print("✅ WARNING alert sent")
        
        await asyncio.sleep(1)  # Small delay between alerts
        
        await dispatcher.dispatch(critical_alert)
        print("✅ CRITICAL alert sent")
        
        print("\n✅ Alert test completed successfully!")
        print("Check your Slack/PagerDuty channels for the test alerts.")
        return 0
        
    except Exception as e:
        print(f"\n❌ Alert test failed: {e}")
        return 1


def main():
    """Main entry point"""
    print("\n=== Alert System Test ===")
    print("This will send test alerts to configured channels.")
    print("Make sure SLACK_WEBHOOK_URL and/or PAGERDUTY_ROUTING_KEY are set.\n")
    
    result = asyncio.run(test_alerts())
    return result


if __name__ == "__main__":
    sys.exit(main())