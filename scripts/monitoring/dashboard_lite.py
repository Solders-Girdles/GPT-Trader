#!/usr/bin/env python3
"""
Lightweight monitoring dashboard for Coinbase Perpetuals trading.
Aggregates key metrics in real-time without heavy dependencies.
"""

import os
import sys
import time
import json
import asyncio
from datetime import datetime, timedelta, timezone
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple
from decimal import Decimal

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


class TradingDashboard:
    """Lightweight trading metrics dashboard."""
    
    def __init__(self):
        self.metrics = defaultdict(lambda: deque(maxlen=100))
        self.summary = {}
        self.alerts = []
        self.start_time = datetime.now(timezone.utc)
        
    def run(self, refresh_interval: int = 5):
        """Run dashboard with periodic refresh."""
        print("\n" + "="*80)
        print(" " * 20 + "🚀 TRADING DASHBOARD LITE")
        print("="*80)
        
        try:
            while True:
                self.update_metrics()
                self.check_alerts()
                self.display()
                time.sleep(refresh_interval)
        except KeyboardInterrupt:
            print("\n\n👋 Dashboard stopped")
    
    def update_metrics(self):
        """Update all metrics from various sources."""
        # Order metrics
        self._update_order_metrics()
        
        # Position metrics
        self._update_position_metrics()
        
        # Market metrics
        self._update_market_metrics()
        
        # System metrics
        self._update_system_metrics()
    
    def _update_order_metrics(self):
        """Update order-related metrics."""
        try:
            # Read from log file or database
            log_file = "/tmp/trading_orders.log"
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    lines = f.readlines()[-100:]  # Last 100 orders
                    
                accepted = sum(1 for l in lines if '"status":"accepted"' in l)
                rejected = sum(1 for l in lines if '"status":"rejected"' in l)
                total = len(lines)
                
                if total > 0:
                    self.summary['acceptance_rate'] = (accepted / total) * 100
                    self.summary['rejection_rate'] = (rejected / total) * 100
                    self.summary['total_orders'] = total
                    
                    # Track rejection reasons
                    rejections = defaultdict(int)
                    for line in lines:
                        if '"status":"rejected"' in line:
                            try:
                                data = json.loads(line)
                                reason = data.get('reason', 'unknown')
                                rejections[reason] += 1
                            except:
                                pass
                    self.summary['rejection_reasons'] = dict(rejections)
        except Exception as e:
            self.summary['order_error'] = str(e)
    
    def _update_position_metrics(self):
        """Update position and PnL metrics."""
        try:
            # Read from state file
            state_file = "/tmp/trading_state.json"
            if os.path.exists(state_file):
                with open(state_file, 'r') as f:
                    state = json.load(f)
                
                positions = state.get('positions', {})
                total_pnl = Decimal('0')
                total_funding = Decimal('0')
                
                for symbol, pos in positions.items():
                    pnl = Decimal(str(pos.get('unrealized_pnl', 0)))
                    funding = Decimal(str(pos.get('funding_paid', 0)))
                    total_pnl += pnl
                    total_funding += funding
                
                self.summary['total_pnl'] = float(total_pnl)
                self.summary['total_funding'] = float(total_funding)
                self.summary['position_count'] = len(positions)
                self.summary['positions'] = positions
        except Exception as e:
            self.summary['position_error'] = str(e)
    
    def _update_market_metrics(self):
        """Update market condition metrics."""
        try:
            # Read market data
            market_file = "/tmp/market_conditions.json"
            if os.path.exists(market_file):
                with open(market_file, 'r') as f:
                    market = json.load(f)
                
                self.summary['spread_bps'] = market.get('spread_bps', 0)
                self.summary['depth_imbalance'] = market.get('depth_imbalance', 0)
                self.summary['volatility'] = market.get('volatility', 0)
                self.summary['funding_rate'] = market.get('funding_rate', 0)
        except:
            pass  # Market data is optional
    
    def _update_system_metrics(self):
        """Update system performance metrics."""
        try:
            # API latency
            latency_file = "/tmp/api_latency.json"
            if os.path.exists(latency_file):
                with open(latency_file, 'r') as f:
                    latency = json.load(f)
                    self.summary['latency_p50'] = latency.get('p50', 0)
                    self.summary['latency_p95'] = latency.get('p95', 0)
                    self.summary['latency_p99'] = latency.get('p99', 0)
            
            # WebSocket status
            ws_file = "/tmp/websocket_status.json"
            if os.path.exists(ws_file):
                with open(ws_file, 'r') as f:
                    ws = json.load(f)
                    self.summary['ws_connected'] = ws.get('connected', False)
                    self.summary['ws_reconnects'] = ws.get('reconnect_count', 0)
                    self.summary['ws_last_message'] = ws.get('last_message_time', 'N/A')
            
            # Calculate uptime
            uptime = datetime.now(timezone.utc) - self.start_time
            self.summary['uptime_hours'] = uptime.total_seconds() / 3600
            
        except Exception as e:
            self.summary['system_error'] = str(e)
    
    def check_alerts(self):
        """Check for alert conditions."""
        self.alerts = []
        
        # Check acceptance rate
        if self.summary.get('acceptance_rate', 100) < 90:
            self.alerts.append(("⚠️ WARNING", f"Low acceptance rate: {self.summary['acceptance_rate']:.1f}%"))
        
        # Check PnL
        if self.summary.get('total_pnl', 0) < -100:
            self.alerts.append(("🔴 CRITICAL", f"Daily loss exceeds $100: ${self.summary['total_pnl']:.2f}"))
        
        # Check latency
        if self.summary.get('latency_p95', 0) > 500:
            self.alerts.append(("⚠️ WARNING", f"High latency: {self.summary['latency_p95']:.0f}ms"))
        
        # Check WebSocket
        if not self.summary.get('ws_connected', True):
            self.alerts.append(("🔴 CRITICAL", "WebSocket disconnected"))
        
        if self.summary.get('ws_reconnects', 0) > 5:
            self.alerts.append(("⚠️ WARNING", f"Excessive reconnects: {self.summary['ws_reconnects']}"))
        
        # Check spread
        if self.summary.get('spread_bps', 0) > 20:
            self.alerts.append(("⚠️ WARNING", f"Wide spread: {self.summary['spread_bps']:.1f} bps"))
        
        # Check funding
        if abs(self.summary.get('funding_rate', 0)) > 0.001:
            self.alerts.append(("📊 INFO", f"High funding rate: {self.summary['funding_rate']:.4f}"))
    
    def display(self):
        """Display dashboard."""
        # Clear screen (Unix/Mac)
        os.system('clear' if os.name == 'posix' else 'cls')
        
        # Header
        print("\n" + "="*80)
        print(" " * 20 + "🚀 TRADING DASHBOARD LITE")
        print("="*80)
        print(f"📅 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"⏱️  Uptime: {self.summary.get('uptime_hours', 0):.1f} hours")
        print("="*80)
        
        # Alerts
        if self.alerts:
            print("\n🚨 ALERTS")
            print("-"*40)
            for level, message in self.alerts:
                print(f"{level}: {message}")
        
        # Order Metrics
        print("\n📊 ORDER METRICS")
        print("-"*40)
        print(f"Acceptance Rate: {self.summary.get('acceptance_rate', 0):.1f}%")
        print(f"Rejection Rate:  {self.summary.get('rejection_rate', 0):.1f}%")
        print(f"Total Orders:    {self.summary.get('total_orders', 0)}")
        
        if self.summary.get('rejection_reasons'):
            print("\nRejection Breakdown:")
            for reason, count in sorted(self.summary['rejection_reasons'].items(), 
                                       key=lambda x: x[1], reverse=True)[:5]:
                print(f"  {reason}: {count}")
        
        # Position Metrics
        print("\n💼 POSITION METRICS")
        print("-"*40)
        print(f"Active Positions: {self.summary.get('position_count', 0)}")
        print(f"Total PnL:        ${self.summary.get('total_pnl', 0):.2f}")
        print(f"Funding Paid:     ${self.summary.get('total_funding', 0):.2f}")
        
        if self.summary.get('positions'):
            print("\nPositions:")
            for symbol, pos in self.summary['positions'].items():
                size = pos.get('size', 0)
                pnl = pos.get('unrealized_pnl', 0)
                print(f"  {symbol}: {size:.4f} | PnL: ${pnl:.2f}")
        
        # Market Metrics
        print("\n📈 MARKET CONDITIONS")
        print("-"*40)
        print(f"Spread:          {self.summary.get('spread_bps', 0):.1f} bps")
        print(f"Depth Imbalance: {self.summary.get('depth_imbalance', 0):.2f}")
        print(f"Volatility:      {self.summary.get('volatility', 0):.2f}%")
        print(f"Funding Rate:    {self.summary.get('funding_rate', 0):.5f}")
        
        # System Metrics
        print("\n⚙️  SYSTEM PERFORMANCE")
        print("-"*40)
        print(f"Latency p50:     {self.summary.get('latency_p50', 0):.0f}ms")
        print(f"Latency p95:     {self.summary.get('latency_p95', 0):.0f}ms")
        print(f"Latency p99:     {self.summary.get('latency_p99', 0):.0f}ms")
        
        ws_status = "✅ Connected" if self.summary.get('ws_connected', False) else "❌ Disconnected"
        print(f"WebSocket:       {ws_status}")
        print(f"Reconnects:      {self.summary.get('ws_reconnects', 0)}")
        print(f"Last Message:    {self.summary.get('ws_last_message', 'N/A')}")
        
        # Footer
        print("\n" + "="*80)
        print("Press Ctrl+C to exit")
    
    def export_metrics(self, filepath: str = "dashboard_metrics.json"):
        """Export current metrics to file."""
        export_data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'summary': self.summary,
            'alerts': self.alerts
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"📄 Metrics exported to {filepath}")


class MetricsCollector:
    """Collect metrics from various sources."""
    
    @staticmethod
    def simulate_metrics():
        """Simulate metrics for testing."""
        import random
        
        # Simulate order log
        with open("/tmp/trading_orders.log", 'w') as f:
            for i in range(50):
                status = "accepted" if random.random() > 0.1 else "rejected"
                reason = random.choice(["POST_ONLY_WOULD_CROSS", "INSUFFICIENT_LIQUIDITY", 
                                       "RSI_FILTER_BLOCKED"]) if status == "rejected" else None
                order = {
                    "id": f"order_{i}",
                    "status": status,
                    "reason": reason,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                f.write(json.dumps(order) + "\n")
        
        # Simulate position state
        positions = {
            "BTC-PERP": {
                "size": 0.0001,
                "unrealized_pnl": random.uniform(-50, 100),
                "funding_paid": random.uniform(-5, 5)
            }
        }
        
        with open("/tmp/trading_state.json", 'w') as f:
            json.dump({"positions": positions}, f)
        
        # Simulate market conditions
        market = {
            "spread_bps": random.uniform(5, 25),
            "depth_imbalance": random.uniform(-0.5, 0.5),
            "volatility": random.uniform(10, 50),
            "funding_rate": random.uniform(-0.0005, 0.0005)
        }
        
        with open("/tmp/market_conditions.json", 'w') as f:
            json.dump(market, f)
        
        # Simulate latency
        latency = {
            "p50": random.uniform(50, 150),
            "p95": random.uniform(200, 400),
            "p99": random.uniform(400, 800)
        }
        
        with open("/tmp/api_latency.json", 'w') as f:
            json.dump(latency, f)
        
        # Simulate WebSocket status
        ws = {
            "connected": random.random() > 0.1,
            "reconnect_count": random.randint(0, 10),
            "last_message_time": datetime.now(timezone.utc).isoformat()
        }
        
        with open("/tmp/websocket_status.json", 'w') as f:
            json.dump(ws, f)


def main():
    """Run the dashboard."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Trading Dashboard Lite")
    parser.add_argument("--simulate", action="store_true", 
                       help="Simulate metrics for testing")
    parser.add_argument("--refresh", type=int, default=5,
                       help="Refresh interval in seconds")
    parser.add_argument("--export", type=str,
                       help="Export metrics to file")
    
    args = parser.parse_args()
    
    if args.simulate:
        print("🎮 Running in simulation mode...")
        # Generate initial simulated data
        MetricsCollector.simulate_metrics()
        
        # Start background simulation
        import threading
        def update_simulation():
            while True:
                time.sleep(args.refresh)
                MetricsCollector.simulate_metrics()
        
        sim_thread = threading.Thread(target=update_simulation, daemon=True)
        sim_thread.start()
    
    dashboard = TradingDashboard()
    
    if args.export:
        dashboard.update_metrics()
        dashboard.export_metrics(args.export)
    else:
        dashboard.run(refresh_interval=args.refresh)


if __name__ == "__main__":
    main()