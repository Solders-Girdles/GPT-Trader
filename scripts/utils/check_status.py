#!/usr/bin/env python3
"""
Check Current Paper Trading Status
Quick script to check the current state of paper trading sessions.
"""

import json
from pathlib import Path
from datetime import datetime

def check_status():
    """Check and display current paper trading status."""
    results_dir = Path(__file__).parent.parent / 'results'
    
    print("=" * 70)
    print("PAPER TRADING STATUS CHECK")
    print("=" * 70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Find all session files
    session_files = list(results_dir.glob('live_*.json')) + list(results_dir.glob('*.json'))
    
    if not session_files:
        print("❌ No paper trading sessions found")
        return
    
    # Get most recent file
    latest_file = max(session_files, key=lambda f: f.stat().st_mtime)
    
    # Check if file was recently updated (within last 5 seconds)
    file_age = datetime.now().timestamp() - latest_file.stat().st_mtime
    is_active = file_age < 5
    
    print(f"Latest Session: {latest_file.name}")
    print(f"Last Updated: {file_age:.0f} seconds ago")
    print(f"Status: {'🟢 ACTIVE' if is_active else '🔴 INACTIVE'}")
    print()
    
    # Load and display session data
    try:
        with open(latest_file, 'r') as f:
            data = json.load(f)
        
        # Display session info
        if 'session_name' in data:
            print(f"Session Name: {data['session_name']}")
        if 'strategy' in data:
            print(f"Strategy: {data.get('strategy', 'Unknown')}")
        if 'symbols' in data:
            print(f"Symbols: {', '.join(data.get('symbols', []))}")
        
        print()
        
        # Display metrics
        metrics = data.get('metrics', {})
        if metrics:
            print("CURRENT METRICS:")
            print("-" * 40)
            print(f"Equity: ${metrics.get('equity', 0):.2f}")
            print(f"Return: {metrics.get('total_return', 0):+.2f}%")
            print(f"Cash: ${metrics.get('cash', 0):.2f}")
            print(f"Positions Value: ${metrics.get('positions_value', 0):.2f}")
            print(f"Number of Trades: {metrics.get('num_trades', 0)}")
            print(f"Number of Positions: {metrics.get('num_positions', 0)}")
            print(f"Win Rate: {metrics.get('win_rate', 0):.1f}%")
            print(f"Drawdown: {metrics.get('drawdown', 0):.2f}%")
            
            if 'duration' in metrics:
                print(f"Duration: {metrics['duration']}")
        
        # Display positions
        positions = data.get('positions', {})
        if positions:
            print()
            print("OPEN POSITIONS:")
            print("-" * 40)
            for symbol, pos in positions.items():
                pnl = (pos.get('current_price', 0) - pos.get('entry_price', 0)) * pos.get('quantity', 0)
                pnl_pct = ((pos.get('current_price', 0) - pos.get('entry_price', 0)) / pos.get('entry_price', 1)) * 100
                print(f"{symbol}: {pos.get('quantity', 0):.6f} @ ${pos.get('entry_price', 0):.2f}")
                print(f"  Current: ${pos.get('current_price', 0):.2f} | P&L: ${pnl:.2f} ({pnl_pct:+.1f}%)")
        
        # Display recent trades
        trades = data.get('trades', [])
        if trades:
            print()
            print(f"RECENT TRADES (Last 5 of {len(trades)}):")
            print("-" * 40)
            for trade in trades[-5:]:
                time_str = trade.get('timestamp', 'Unknown')
                if time_str != 'Unknown':
                    try:
                        dt = datetime.fromisoformat(time_str)
                        time_str = dt.strftime('%H:%M:%S')
                    except:
                        pass
                
                symbol = trade.get('symbol', 'Unknown')
                side = trade.get('side', 'Unknown')
                price = trade.get('price', 0)
                
                print(f"{time_str} {side.upper():4} {symbol} @ ${price:.2f}")
                
                if 'pnl' in trade and trade['pnl'] != 0:
                    print(f"  P&L: ${trade['pnl']:.2f}")
    
    except Exception as e:
        print(f"❌ Error reading session data: {e}")
    
    print()
    print("=" * 70)
    
    # Show monitoring options
    if is_active:
        print()
        print("📊 MONITORING OPTIONS:")
        print("1. Web Dashboard: http://localhost:8888")
        print("2. Terminal Monitor: python scripts/live_monitor.py")
        print("3. Full Analysis: python scripts/monitor_paper_trading.py")

if __name__ == "__main__":
    check_status()