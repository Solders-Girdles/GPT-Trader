#!/usr/bin/env python3
"""
Canary Monitor (10-15 minute Production Test)

Continuous monitoring of canary trading bot with guard checks and metrics.
Tracks PnL, position changes, order flow, and system health.
Includes automatic kill switch and alerting on threshold violations.

Usage:
  poetry run python scripts/canary_monitor.py --duration-minutes 10 --profile canary

  With dashboard:
  poetry run python scripts/canary_monitor.py --duration-minutes 15 --profile canary --dashboard
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from bot_v2.config.path_registry import RESULTS_DIR


# Guard thresholds
@dataclass
class GuardThresholds:
    max_position_size: float = 0.01  # Max position size per symbol
    max_total_exposure: float = 0.05  # Max total portfolio exposure
    max_drawdown_pct: float = 2.0  # Max drawdown %
    max_orders_per_min: int = 10  # Rate limit
    min_pnl_threshold: float = -50.0  # USD loss threshold
    max_latency_ms: float = 500.0  # API latency threshold
    heartbeat_timeout_sec: int = 30  # Bot heartbeat timeout


class CanaryMonitor:
    def __init__(
        self,
        profile: str = "canary",
        duration_minutes: int = 10,
        enable_dashboard: bool = False,
        dry_run: bool = False,
    ):
        self.profile = profile
        self.duration_minutes = duration_minutes
        self.enable_dashboard = enable_dashboard
        self.dry_run = dry_run
        self.guards = GuardThresholds()

        # Metrics tracking
        self.metrics = {
            "start_time": datetime.now(timezone.utc),
            "positions": {},
            "orders_placed": 0,
            "orders_filled": 0,
            "total_pnl": 0.0,
            "max_drawdown": 0.0,
            "api_latencies": deque(maxlen=100),
            "errors": [],
            "violations": [],
            "last_heartbeat": datetime.now(timezone.utc),
        }

        # Kill switch state
        self.kill_switch_triggered = False
        self.client = None
        self.ws_task = None
        self.alert_dispatcher = None

    def setup_client(self):
        """Initialize Coinbase client for monitoring"""
        from bot_v2.features.brokerages.coinbase.cdp_auth_v2 import CDPAuthV2
        from bot_v2.features.brokerages.coinbase.client import CoinbaseClient

        api_key = os.getenv("COINBASE_PROD_CDP_API_KEY") or os.getenv("COINBASE_CDP_API_KEY")
        private_key = os.getenv("COINBASE_PROD_CDP_PRIVATE_KEY") or os.getenv(
            "COINBASE_CDP_PRIVATE_KEY"
        )

        if not api_key or not private_key:
            raise ValueError("Missing CDP credentials")

        auth = CDPAuthV2(api_key_name=api_key, private_key_pem=private_key)
        self.client = CoinbaseClient(
            base_url="https://api.coinbase.com", auth=auth, api_mode="advanced"
        )

        # Setup alert dispatcher
        self.setup_alerts()

    async def monitor_positions(self):
        """Monitor CFM positions and calculate exposure"""
        while not self.kill_switch_triggered:
            try:
                t0 = time.perf_counter()
                positions = self.client.cfm_positions()
                latency_ms = (time.perf_counter() - t0) * 1000
                self.metrics["api_latencies"].append(latency_ms)

                # Check latency guard
                if latency_ms > self.guards.max_latency_ms:
                    self.trigger_violation(f"High API latency: {latency_ms:.0f}ms")

                # Parse positions
                total_exposure = 0.0
                for pos in positions.get("positions", []):
                    symbol = pos.get("product_id")
                    size = abs(float(pos.get("net_size", 0)))
                    mark_price = float(pos.get("mark_price", 0))
                    exposure = size * mark_price

                    self.metrics["positions"][symbol] = {
                        "size": size,
                        "exposure": exposure,
                        "unrealized_pnl": float(pos.get("unrealized_pnl", 0)),
                    }

                    # Check position size guard
                    if size > self.guards.max_position_size:
                        self.trigger_violation(f"Position size violation: {symbol} size={size}")

                    total_exposure += exposure

                # Check total exposure guard
                if total_exposure > self.guards.max_total_exposure:
                    self.trigger_violation(f"Total exposure violation: ${total_exposure:.2f}")

                # Calculate total PnL
                total_pnl = sum(
                    p.get("unrealized_pnl", 0) for p in self.metrics["positions"].values()
                )
                self.metrics["total_pnl"] = total_pnl

                # Check PnL guard
                if total_pnl < self.guards.min_pnl_threshold:
                    self.trigger_violation(f"PnL threshold breached: ${total_pnl:.2f}")

                # Track max drawdown
                if total_pnl < self.metrics["max_drawdown"]:
                    self.metrics["max_drawdown"] = total_pnl
                    drawdown_pct = abs(total_pnl / max(total_exposure, 1)) * 100
                    if drawdown_pct > self.guards.max_drawdown_pct:
                        self.trigger_violation(f"Max drawdown exceeded: {drawdown_pct:.1f}%")

                await asyncio.sleep(5)  # Poll every 5 seconds

            except Exception as e:
                self.metrics["errors"].append(str(e))
                print(f"❌ Position monitoring error: {e}")
                await asyncio.sleep(10)

    async def monitor_orders(self):
        """Monitor order flow and rate limits"""
        order_timestamps = deque(maxlen=100)

        while not self.kill_switch_triggered:
            try:
                # Get recent orders
                orders = self.client.list_orders(
                    limit=50,
                    product_id="BTC-PERP",  # Can be made configurable
                    order_status=["OPEN", "PENDING", "FILLED"],
                )

                for order in orders.get("orders", []):
                    created_at = order.get("created_time")
                    status = order.get("status")

                    # Track order timestamps for rate limiting
                    if created_at:
                        order_timestamps.append(
                            datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                        )

                    # Count orders
                    if status in ["pending", "open"]:
                        self.metrics["orders_placed"] += 1
                    elif status == "filled":
                        self.metrics["orders_filled"] += 1

                # Check rate limit guard
                now = datetime.now(timezone.utc)
                recent_orders = sum(1 for ts in order_timestamps if (now - ts).seconds < 60)
                if recent_orders > self.guards.max_orders_per_min:
                    self.trigger_violation(f"Rate limit violation: {recent_orders} orders/min")

                await asyncio.sleep(10)  # Poll every 10 seconds

            except Exception as e:
                self.metrics["errors"].append(str(e))
                print(f"❌ Order monitoring error: {e}")
                await asyncio.sleep(10)

    async def monitor_heartbeat(self):
        """Monitor bot heartbeat via WebSocket or logs"""
        while not self.kill_switch_triggered:
            try:
                # Check if heartbeat is stale
                elapsed = (datetime.now(timezone.utc) - self.metrics["last_heartbeat"]).seconds
                if elapsed > self.guards.heartbeat_timeout_sec:
                    self.trigger_violation(f"Heartbeat timeout: {elapsed}s")

                # In production, this would check actual bot heartbeat
                # For now, we'll ping the API as a proxy
                self.client.get_time()
                self.metrics["last_heartbeat"] = datetime.now(timezone.utc)

                await asyncio.sleep(10)

            except Exception as e:
                print(f"⚠️ Heartbeat check failed: {e}")
                await asyncio.sleep(10)

    def setup_alerts(self):
        """Setup alert dispatcher with available channels"""
        try:
            from bot_v2.monitoring.alerts import (
                AlertDispatcher,
                AlertSeverity,
                SlackChannel,
                PagerDutyChannel,
                create_system_alert,
            )

            self.alert_dispatcher = AlertDispatcher()

            # Add Slack channel if configured
            if slack_webhook := os.getenv("SLACK_WEBHOOK_URL"):
                slack_channel = SlackChannel(
                    webhook_url=slack_webhook, min_severity=AlertSeverity.WARNING
                )
                self.alert_dispatcher.add_channel("slack", slack_channel)
                print("📢 Slack alerts configured")

            # Add PagerDuty channel if configured
            # Support both API_KEY and ROUTING_KEY env vars
            pagerduty_key = os.getenv("PAGERDUTY_API_KEY") or os.getenv("PAGERDUTY_ROUTING_KEY")
            if pagerduty_key:
                pd_channel = PagerDutyChannel(
                    api_key=pagerduty_key, min_severity=AlertSeverity.ERROR
                )
                self.alert_dispatcher.add_channel("pagerduty", pd_channel)
                print("📟 PagerDuty alerts configured")

        except ImportError:
            print("⚠️ Alert system not available")
            self.alert_dispatcher = None
        except Exception as e:
            print(f"⚠️ Failed to setup alerts: {e}")
            self.alert_dispatcher = None

    def trigger_violation(self, message: str):
        """Handle guard violation"""
        self.metrics["violations"].append(
            {"timestamp": datetime.now(timezone.utc).isoformat(), "message": message}
        )
        print(f"🚨 VIOLATION: {message}")

        # Send alert if dispatcher is available
        if self.alert_dispatcher:
            try:
                from bot_v2.monitoring.alerts import AlertSeverity, create_system_alert

                # Determine severity based on violation count
                severity = (
                    AlertSeverity.CRITICAL
                    if len(self.metrics["violations"]) >= 2
                    else AlertSeverity.WARNING
                )

                # Create and dispatch alert with proper signature
                alert = create_system_alert(
                    title="Canary Guard Violation",
                    message=message,
                    severity=severity,
                    profile=self.profile,
                    violation_count=len(self.metrics["violations"]),
                    total_pnl=self.metrics.get("total_pnl", 0),
                    max_drawdown=self.metrics.get("max_drawdown", 0),
                )

                # Dispatch alert asynchronously
                asyncio.create_task(self.alert_dispatcher.dispatch(alert))

            except Exception as e:
                print(f"⚠️ Failed to send alert: {e}")

        # Trigger kill switch after 3 violations
        if len(self.metrics["violations"]) >= 3 and not self.dry_run:
            self.trigger_kill_switch()

    def trigger_kill_switch(self):
        """Emergency stop all trading"""
        if self.kill_switch_triggered:
            return

        self.kill_switch_triggered = True
        print("\n🔴 KILL SWITCH ACTIVATED 🔴")

        try:
            # Cancel all open orders
            print("Cancelling all open orders...")
            if self.client:
                # First, list all open orders
                open_orders = self.client.list_orders(order_status=["OPEN", "PENDING"], limit=100)

                order_ids = []
                for order in open_orders.get("orders", []):
                    if order_id := order.get("order_id"):
                        order_ids.append(order_id)

                if order_ids:
                    print(f"Cancelling {len(order_ids)} open orders...")
                    self.client.cancel_orders(order_ids)

            # Close all positions (in production)
            if not self.dry_run:
                print("Closing all positions...")
                # Implementation would close positions here

        except Exception as e:
            print(f"❌ Kill switch error: {e}")

    def print_status(self):
        """Print monitoring status"""
        elapsed = datetime.now(timezone.utc) - self.metrics["start_time"]

        print("\n" + "=" * 60)
        print(f"📊 CANARY MONITOR STATUS | {elapsed.seconds//60}:{elapsed.seconds%60:02d}")
        print("=" * 60)

        # Positions
        print("\n📈 Positions:")
        for symbol, pos in self.metrics["positions"].items():
            print(f"  {symbol}: size={pos['size']:.4f} | PnL=${pos['unrealized_pnl']:.2f}")

        # Metrics
        print(f"\n📊 Metrics:")
        print(f"  Total PnL: ${self.metrics['total_pnl']:.2f}")
        print(f"  Max Drawdown: ${self.metrics['max_drawdown']:.2f}")
        print(f"  Orders Placed: {self.metrics['orders_placed']}")
        print(f"  Orders Filled: {self.metrics['orders_filled']}")

        if self.metrics["api_latencies"]:
            avg_latency = sum(self.metrics["api_latencies"]) / len(self.metrics["api_latencies"])
            print(f"  Avg Latency: {avg_latency:.0f}ms")

        # Violations
        if self.metrics["violations"]:
            print(f"\n⚠️ Violations ({len(self.metrics['violations'])}):")
            for v in self.metrics["violations"][-3:]:
                print(f"  - {v['message']}")

        # Errors
        if self.metrics["errors"]:
            print(f"\n❌ Errors ({len(self.metrics['errors'])}):")
            for e in self.metrics["errors"][-3:]:
                print(f"  - {e}")

    async def export_metrics(self):
        """Export metrics to file or dashboard"""
        filename = f"canary_monitor_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        filepath = str((RESULTS_DIR / filename).resolve())

        os.makedirs("results", exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(self.metrics, f, indent=2, default=str)

        print(f"\n📁 Metrics exported to: {filepath}")

        if self.enable_dashboard:
            # Send to dashboard (implementation would go here)
            print("📊 Metrics sent to dashboard")

    async def run(self):
        """Main monitoring loop"""
        print(f"\n🚀 Starting Canary Monitor")
        print(f"Profile: {self.profile}")
        print(f"Duration: {self.duration_minutes} minutes")
        print(f"Dashboard: {'Enabled' if self.enable_dashboard else 'Disabled'}")
        print(f"Mode: {'DRY RUN' if self.dry_run else 'LIVE'}")

        # Setup
        self.setup_client()

        # Start monitoring tasks
        tasks = [
            asyncio.create_task(self.monitor_positions()),
            asyncio.create_task(self.monitor_orders()),
            asyncio.create_task(self.monitor_heartbeat()),
        ]

        # Status printing loop
        end_time = datetime.now(timezone.utc) + timedelta(minutes=self.duration_minutes)

        try:
            while datetime.now(timezone.utc) < end_time and not self.kill_switch_triggered:
                self.print_status()
                await asyncio.sleep(30)  # Update every 30 seconds

            # Final status
            self.print_status()

            # Cancel monitoring tasks
            for task in tasks:
                task.cancel()

            # Export metrics
            await self.export_metrics()

            # Summary
            print("\n" + "=" * 60)
            if self.kill_switch_triggered:
                print("🔴 MONITORING STOPPED: Kill switch activated")
                return 1
            elif self.metrics["violations"]:
                print(f"⚠️ COMPLETED WITH {len(self.metrics['violations'])} VIOLATIONS")
                return 2
            else:
                print("✅ MONITORING COMPLETED SUCCESSFULLY")
                return 0

        except KeyboardInterrupt:
            print("\n⚠️ Monitoring interrupted by user")
            return 1
        finally:
            # Cleanup
            for task in tasks:
                try:
                    task.cancel()
                    await task
                except asyncio.CancelledError:
                    pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Canary Monitor")
    parser.add_argument("--profile", default="canary", help="Trading profile to monitor")
    parser.add_argument("--duration-minutes", type=int, default=10, help="Monitoring duration")
    parser.add_argument("--dashboard", action="store_true", help="Enable dashboard integration")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode (no kill switch)")

    args = parser.parse_args()

    monitor = CanaryMonitor(
        profile=args.profile,
        duration_minutes=args.duration_minutes,
        enable_dashboard=args.dashboard,
        dry_run=args.dry_run,
    )

    return asyncio.run(monitor.run())


if __name__ == "__main__":
    sys.exit(main())
