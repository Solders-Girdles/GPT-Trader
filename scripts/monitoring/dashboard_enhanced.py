#!/usr/bin/env python3
"""
Enhanced monitoring dashboard with acceptance rate tracking.
"""

import os
import sys
import json
import asyncio
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import defaultdict, deque
from typing import Dict, List, Optional

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


class EnhancedDashboard:
    """Enhanced real-time monitoring dashboard."""

    def __init__(self):
        self.start_time = datetime.now(timezone.utc)

        # Metrics storage (last 1000 data points)
        self.metrics = {
            "orders": deque(maxlen=1000),
            "fills": deque(maxlen=1000),
            "rejections": deque(maxlen=1000),
            "latencies": deque(maxlen=1000),
            "positions": deque(maxlen=1000),
            "pnl": deque(maxlen=1000),
            "websocket_status": deque(maxlen=100),
            "errors": deque(maxlen=100),
        }

        # Aggregated counters
        self.counters = {
            "total_orders": 0,
            "accepted_orders": 0,
            "rejected_orders": 0,
            "fills": 0,
            "websocket_reconnects": 0,
            "errors": 0,
        }

        # Rejection reasons
        self.rejection_reasons = defaultdict(int)

        # Log file
        log_dir = Path("/tmp/trading_logs")
        log_dir.mkdir(exist_ok=True)
        self.log_file = (
            log_dir / f"dashboard_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.log"
        )

    def log(self, message: str):
        """Log message with timestamp."""
        timestamp = datetime.now(timezone.utc).isoformat()
        log_entry = f"[{timestamp}] {message}"

        with open(self.log_file, "a") as f:
            f.write(log_entry + "\n")

    def clear_screen(self):
        """Clear terminal screen."""
        os.system("clear" if os.name == "posix" else "cls")

    def record_order(self, order_data: dict):
        """Record order placement."""
        timestamp = datetime.now(timezone.utc)

        order_record = {
            "timestamp": timestamp,
            "symbol": order_data.get("symbol"),
            "side": order_data.get("side"),
            "type": order_data.get("type"),
            "size": order_data.get("size"),
            "price": order_data.get("price"),
            "status": order_data.get("status", "pending"),
        }

        self.metrics["orders"].append(order_record)
        self.counters["total_orders"] += 1

        if order_record["status"] == "accepted":
            self.counters["accepted_orders"] += 1
        elif order_record["status"] == "rejected":
            self.counters["rejected_orders"] += 1
            reason = order_data.get("rejection_reason", "unknown")
            self.rejection_reasons[reason] += 1

    def record_fill(self, fill_data: dict):
        """Record order fill."""
        timestamp = datetime.now(timezone.utc)

        fill_record = {
            "timestamp": timestamp,
            "order_id": fill_data.get("order_id"),
            "symbol": fill_data.get("symbol"),
            "side": fill_data.get("side"),
            "size": fill_data.get("size"),
            "price": fill_data.get("price"),
        }

        self.metrics["fills"].append(fill_record)
        self.counters["fills"] += 1

    def record_rejection(self, rejection_data: dict):
        """Record order rejection."""
        timestamp = datetime.now(timezone.utc)

        rejection_record = {
            "timestamp": timestamp,
            "reason": rejection_data.get("reason", "unknown"),
            "order_id": rejection_data.get("order_id"),
            "details": rejection_data.get("details"),
        }

        self.metrics["rejections"].append(rejection_record)
        self.counters["rejected_orders"] += 1
        self.rejection_reasons[rejection_record["reason"]] += 1

    def record_latency(self, latency_ms: float):
        """Record API latency."""
        timestamp = datetime.now(timezone.utc)

        self.metrics["latencies"].append({"timestamp": timestamp, "latency_ms": latency_ms})

    def record_position(self, position_data: dict):
        """Record position update."""
        timestamp = datetime.now(timezone.utc)

        position_record = {
            "timestamp": timestamp,
            "symbol": position_data.get("symbol"),
            "size": position_data.get("size"),
            "entry_price": position_data.get("entry_price"),
            "market_price": position_data.get("market_price"),
            "unrealized_pnl": position_data.get("unrealized_pnl"),
            "realized_pnl": position_data.get("realized_pnl"),
        }

        self.metrics["positions"].append(position_record)

    def record_pnl(self, pnl_data: dict):
        """Record P&L update."""
        timestamp = datetime.now(timezone.utc)

        pnl_record = {
            "timestamp": timestamp,
            "realized_pnl": pnl_data.get("realized_pnl", 0),
            "unrealized_pnl": pnl_data.get("unrealized_pnl", 0),
            "total_pnl": pnl_data.get("total_pnl", 0),
        }

        self.metrics["pnl"].append(pnl_record)

    def record_websocket_event(self, event_type: str, details: str = ""):
        """Record WebSocket event."""
        timestamp = datetime.now(timezone.utc)

        ws_record = {"timestamp": timestamp, "event": event_type, "details": details}

        self.metrics["websocket_status"].append(ws_record)

        if event_type == "reconnect":
            self.counters["websocket_reconnects"] += 1

    def record_error(self, error_data: dict):
        """Record error."""
        timestamp = datetime.now(timezone.utc)

        error_record = {
            "timestamp": timestamp,
            "type": error_data.get("type", "unknown"),
            "message": error_data.get("message"),
            "severity": error_data.get("severity", "error"),
        }

        self.metrics["errors"].append(error_record)
        self.counters["errors"] += 1

    def calculate_acceptance_rate(self, window_minutes: int = 5) -> float:
        """Calculate acceptance rate for last N minutes."""
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)

        recent_orders = [o for o in self.metrics["orders"] if o["timestamp"] > cutoff]

        if not recent_orders:
            return 0.0

        accepted = len([o for o in recent_orders if o["status"] == "accepted"])
        return (accepted / len(recent_orders)) * 100

    def calculate_avg_latency(self, window_minutes: int = 5) -> float:
        """Calculate average latency for last N minutes."""
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)

        recent_latencies = [
            l["latency_ms"] for l in self.metrics["latencies"] if l["timestamp"] > cutoff
        ]

        if not recent_latencies:
            return 0.0

        return sum(recent_latencies) / len(recent_latencies)

    def get_current_pnl(self) -> dict:
        """Get current P&L."""
        if not self.metrics["pnl"]:
            return {"realized": 0.0, "unrealized": 0.0, "total": 0.0}

        latest = self.metrics["pnl"][-1]
        return {
            "realized": latest["realized_pnl"],
            "unrealized": latest["unrealized_pnl"],
            "total": latest["total_pnl"],
        }

    def get_websocket_status(self) -> str:
        """Get WebSocket connection status."""
        if not self.metrics["websocket_status"]:
            return "Unknown"

        latest = self.metrics["websocket_status"][-1]

        # Check if last event was recent
        time_since = datetime.now(timezone.utc) - latest["timestamp"]
        if time_since > timedelta(seconds=30):
            return "Stale"

        if latest["event"] in ["connected", "message_received"]:
            return "Connected"
        elif latest["event"] == "disconnected":
            return "Disconnected"
        else:
            return latest["event"].title()

    def render_dashboard(self):
        """Render the dashboard."""
        self.clear_screen()

        # Header
        print("🚀 TRADING SYSTEM DASHBOARD")
        print("=" * 80)

        uptime = datetime.now(timezone.utc) - self.start_time
        print(
            f"Uptime: {str(uptime).split('.')[0]} | Environment: {os.getenv('COINBASE_API_MODE', 'unknown')}"
        )
        print(f"Timestamp: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")

        # Order Statistics
        print(f"\n📊 ORDER STATISTICS (Last 5 min)")
        print("-" * 50)

        acceptance_rate = self.calculate_acceptance_rate()
        acceptance_color = (
            "🟢" if acceptance_rate >= 90 else "🟡" if acceptance_rate >= 75 else "🔴"
        )

        print(f"Total Orders:     {self.counters['total_orders']}")
        print(f"Accepted:         {self.counters['accepted_orders']}")
        print(f"Rejected:         {self.counters['rejected_orders']}")
        print(f"Acceptance Rate:  {acceptance_color} {acceptance_rate:.1f}%")
        print(f"Fills:            {self.counters['fills']}")

        # Rejection Reasons
        if self.rejection_reasons:
            print(f"\n❌ REJECTION BREAKDOWN:")
            for reason, count in sorted(
                self.rejection_reasons.items(), key=lambda x: x[1], reverse=True
            ):
                percentage = (count / max(self.counters["rejected_orders"], 1)) * 100
                print(f"  {reason}: {count} ({percentage:.1f}%)")

        # Performance Metrics
        print(f"\n⚡ PERFORMANCE METRICS")
        print("-" * 50)

        avg_latency = self.calculate_avg_latency()
        latency_color = "🟢" if avg_latency < 200 else "🟡" if avg_latency < 500 else "🔴"

        print(f"Avg Latency:      {latency_color} {avg_latency:.0f}ms")
        print(f"WebSocket:        {self.get_websocket_status()}")
        print(f"WS Reconnects:    {self.counters['websocket_reconnects']}")
        print(f"Errors:           {self.counters['errors']}")

        # P&L Section
        pnl = self.get_current_pnl()
        pnl_color = "🟢" if pnl["total"] >= 0 else "🔴"

        print(f"\n💰 PROFIT & LOSS")
        print("-" * 50)
        print(f"Realized PnL:     ${pnl['realized']:.2f}")
        print(f"Unrealized PnL:   ${pnl['unrealized']:.2f}")
        print(f"Total PnL:        {pnl_color} ${pnl['total']:.2f}")

        # Recent Activity
        print(f"\n📋 RECENT ACTIVITY (Last 10)")
        print("-" * 50)

        # Show recent orders
        recent_orders = list(self.metrics["orders"])[-10:]
        for order in recent_orders:
            timestamp = order["timestamp"].strftime("%H:%M:%S")
            symbol = order["symbol"] or "N/A"
            side = order["side"] or "N/A"
            status = order["status"]
            status_icon = "✅" if status == "accepted" else "❌" if status == "rejected" else "⏳"

            print(f"  {timestamp} {status_icon} {symbol} {side.upper()}")

        # Show recent errors
        if self.metrics["errors"]:
            recent_errors = list(self.metrics["errors"])[-3:]
            if recent_errors:
                print(f"\n⚠️  RECENT ERRORS:")
                for error in recent_errors:
                    timestamp = error["timestamp"].strftime("%H:%M:%S")
                    error_type = error["type"]
                    message = error["message"][:50] if error["message"] else "N/A"
                    print(f"  {timestamp} {error_type}: {message}...")

        # Status indicators
        print(f"\n🚦 STATUS INDICATORS")
        print("-" * 50)

        # Overall health
        health_checks = {
            "Acceptance Rate ≥90%": acceptance_rate >= 90,
            "Latency <200ms": avg_latency < 200,
            "WebSocket Connected": self.get_websocket_status() == "Connected",
            "No Recent Errors": len(
                [
                    e
                    for e in self.metrics["errors"]
                    if (datetime.now(timezone.utc) - e["timestamp"]).seconds < 300
                ]
            )
            == 0,
        }

        for check, status in health_checks.items():
            icon = "✅" if status else "❌"
            print(f"  {icon} {check}")

        print("\n" + "=" * 80)
        print("Press Ctrl+C to stop monitoring")

    async def simulate_data(self):
        """Simulate trading data for testing."""
        import random

        while True:
            # Simulate order
            if random.random() < 0.3:  # 30% chance
                status = "accepted" if random.random() < 0.92 else "rejected"  # 92% acceptance
                rejection_reason = (
                    random.choice(["POST_ONLY_WOULD_CROSS", "INSUFFICIENT_FUNDS", "SIZE_TOO_SMALL"])
                    if status == "rejected"
                    else None
                )

                self.record_order(
                    {
                        "symbol": "BTC-PERP",
                        "side": random.choice(["buy", "sell"]),
                        "type": "limit",
                        "size": 0.0001,
                        "price": random.uniform(44000, 46000),
                        "status": status,
                        "rejection_reason": rejection_reason,
                    }
                )

            # Simulate fill
            if random.random() < 0.1:  # 10% chance
                self.record_fill(
                    {
                        "symbol": "BTC-PERP",
                        "side": random.choice(["buy", "sell"]),
                        "size": 0.0001,
                        "price": random.uniform(44000, 46000),
                    }
                )

            # Simulate latency
            if random.random() < 0.5:  # 50% chance
                self.record_latency(random.uniform(50, 300))

            # Simulate PnL update
            if random.random() < 0.2:  # 20% chance
                self.record_pnl(
                    {
                        "realized_pnl": random.uniform(-5, 5),
                        "unrealized_pnl": random.uniform(-2, 2),
                        "total_pnl": random.uniform(-7, 7),
                    }
                )

            # Simulate WebSocket event
            if random.random() < 0.1:  # 10% chance
                event_type = random.choice(["message_received", "connected", "reconnect"])
                self.record_websocket_event(event_type)

            await asyncio.sleep(1)

    async def run(self, simulate: bool = False):
        """Run the dashboard."""
        print("Starting enhanced dashboard...")

        # Start simulation if requested
        if simulate:
            asyncio.create_task(self.simulate_data())

        try:
            while True:
                self.render_dashboard()
                await asyncio.sleep(2)  # Update every 2 seconds
        except KeyboardInterrupt:
            print("\n\nDashboard stopped.")
            self.log("Dashboard stopped by user")


async def main():
    """Run enhanced dashboard."""
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced Trading Dashboard")
    parser.add_argument("--simulate", action="store_true", help="Simulate trading data for testing")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    args = parser.parse_args()

    dashboard = EnhancedDashboard()
    await dashboard.run(simulate=args.simulate)


if __name__ == "__main__":
    asyncio.run(main())
