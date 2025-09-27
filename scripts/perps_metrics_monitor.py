#!/usr/bin/env python3
"""
Perps Metrics Monitor - Real-time tracking of demo trading metrics.
"""

import json
import time
from pathlib import Path
from datetime import datetime
from decimal import Decimal

def load_health_file(path="/tmp/phase2_health.json"):
    """Load health metrics from JSON file."""
    try:
        if Path(path).exists():
            with open(path) as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading health file: {e}")
    return {}

def load_event_store(root="/tmp/week3_eventstore"):
    """Load metrics from EventStore."""
    metrics = {
        'events': 0,
        'orders': 0,
        'positions': 0,
        'errors': 0
    }
    
    try:
        event_root = Path(root)
        if event_root.exists():
            for json_file in event_root.rglob("*.json"):
                try:
                    with open(json_file) as f:
                        data = json.load(f)
                        if 'type' in data:
                            metrics['events'] += 1
                            if data['type'] == 'order':
                                metrics['orders'] += 1
                            elif data['type'] == 'position':
                                metrics['positions'] += 1
                            elif data['type'] == 'error':
                                metrics['errors'] += 1
                except:
                    pass
    except Exception as e:
        print(f"Error loading EventStore: {e}")
    
    return metrics

def display_metrics():
    """Display current metrics."""
    print("\033[2J\033[H")  # Clear screen
    print("=" * 60)
    print("PERPS METRICS MONITOR")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Load health metrics
    health = load_health_file()
    if health:
        print("📊 Health Metrics:")
        print(f"   Status: {health.get('status', 'unknown')}")
        print(f"   Uptime: {health.get('uptime', 0)} seconds")
        print(f"   Last update: {health.get('last_update', 'never')}")
        print()
        
        if 'metrics' in health:
            m = health['metrics']
            print("📈 Order Metrics:")
            print(f"   Placed: {m.get('orders_placed', 0)}")
            print(f"   Filled: {m.get('orders_filled', 0)}")
            print(f"   Cancelled: {m.get('orders_cancelled', 0)}")
            print(f"   Rejected: {m.get('orders_rejected', 0)}")
            print(f"   Post-only rejected: {m.get('post_only_rejected', 0)}")
            print()
            
            # Calculate rates
            total = m.get('orders_placed', 0)
            if total > 0:
                fill_rate = m.get('orders_filled', 0) / total * 100
                reject_rate = m.get('orders_rejected', 0) / total * 100
                print("📉 Rates:")
                print(f"   Fill rate: {fill_rate:.1f}%")
                print(f"   Reject rate: {reject_rate:.1f}%")
                print()
        
        if 'positions' in health:
            print("💼 Positions:")
            for symbol, pos in health['positions'].items():
                print(f"   {symbol}: {pos.get('qty', 0)} @ {pos.get('avg_price', 0)}")
                print(f"      PnL: ${pos.get('unrealized_pnl', 0):.2f}")
            print()
        
        if 'pnl' in health:
            pnl = health['pnl']
            print("💰 PnL Summary:")
            print(f"   Realized: ${pnl.get('realized', 0):.2f}")
            print(f"   Unrealized: ${pnl.get('unrealized', 0):.2f}")
            print(f"   Total: ${pnl.get('total', 0):.2f}")
            print(f"   Funding: ${pnl.get('funding', 0):.2f}")
            print()
    
    # Load EventStore metrics
    events = load_event_store()
    if events['events'] > 0:
        print("📁 EventStore:")
        print(f"   Total events: {events['events']}")
        print(f"   Orders: {events['orders']}")
        print(f"   Positions: {events['positions']}")
        print(f"   Errors: {events['errors']}")
        print()
    
    print("=" * 60)
    print("Press Ctrl+C to exit")

def main():
    """Main monitoring loop."""
    print("Starting Metrics Monitor...")
    print("Monitoring /tmp/phase2_health.json")
    print()
    
    try:
        while True:
            display_metrics()
            time.sleep(5)  # Refresh every 5 seconds
    except KeyboardInterrupt:
        print("\n\nMonitor stopped")

if __name__ == "__main__":
    main()
