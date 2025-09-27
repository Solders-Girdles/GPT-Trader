#!/usr/bin/env python3
"""
Live Paper Trading Monitor
Real-time monitoring dashboard for paper trading sessions.
Updates automatically and shows current positions, P&L, and performance.
"""

import os
import sys
import time
import json
import curses
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from collections import deque

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class LiveMonitor:
    """Live monitoring dashboard for paper trading."""
    
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.last_file = None
        self.last_modified = None
        self.trade_history = deque(maxlen=10)  # Last 10 trades
        self.equity_history = deque(maxlen=50)  # Last 50 equity points
        self.start_time = datetime.now()
        
    def get_latest_file(self) -> Optional[Path]:
        """Get the most recently modified results file."""
        json_files = list(self.results_dir.glob("*.json"))
        if not json_files:
            return None
        return max(json_files, key=lambda f: f.stat().st_mtime)
    
    def load_latest_data(self) -> Optional[Dict]:
        """Load data from the latest results file."""
        latest_file = self.get_latest_file()
        if not latest_file:
            return None
        
        # Check if file has been updated
        current_modified = latest_file.stat().st_mtime
        if latest_file == self.last_file and current_modified == self.last_modified:
            return None  # No changes
        
        try:
            with open(latest_file, 'r') as f:
                data = json.load(f)
            self.last_file = latest_file
            self.last_modified = current_modified
            return data
        except:
            return None
    
    def format_currency(self, value: float) -> str:
        """Format currency with color based on positive/negative."""
        if value > 0:
            return f"+${value:.2f}"
        elif value < 0:
            return f"-${abs(value):.2f}"
        return f"${value:.2f}"
    
    def format_percentage(self, value: float) -> str:
        """Format percentage with color."""
        if value > 0:
            return f"+{value:.2f}%"
        elif value < 0:
            return f"{value:.2f}%"
        return f"{value:.2f}%"
    
    def draw_dashboard(self, stdscr, data: Dict):
        """Draw the monitoring dashboard."""
        stdscr.clear()
        height, width = stdscr.getmaxyx()
        
        # Initialize colors
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)
        curses.init_pair(5, curses.COLOR_WHITE, curses.COLOR_BLACK)
        curses.init_pair(6, curses.COLOR_MAGENTA, curses.COLOR_BLACK)
        
        # Title
        title = "📊 LIVE PAPER TRADING MONITOR"
        stdscr.addstr(0, (width - len(title)) // 2, title, curses.color_pair(4) | curses.A_BOLD)
        
        # Get metrics
        metrics = data.get('metrics', {})
        trades = data.get('trades', [])
        positions = data.get('positions', {})
        
        # Update histories
        current_equity = metrics.get('equity', 10000)
        self.equity_history.append(current_equity)
        if trades:
            for trade in trades[-10:]:
                if trade not in self.trade_history:
                    self.trade_history.append(trade)
        
        # Row counter
        row = 2
        
        # Session info
        session_time = datetime.now() - self.start_time
        stdscr.addstr(row, 2, f"Session Time: {str(session_time).split('.')[0]}", curses.color_pair(5))
        stdscr.addstr(row, width//2, f"Last Update: {datetime.now().strftime('%H:%M:%S')}", curses.color_pair(5))
        row += 2
        
        # Main metrics
        stdscr.addstr(row, 2, "━" * (width - 4), curses.color_pair(5))
        row += 1
        stdscr.addstr(row, 2, "PERFORMANCE METRICS", curses.color_pair(4) | curses.A_BOLD)
        row += 1
        stdscr.addstr(row, 2, "━" * (width - 4), curses.color_pair(5))
        row += 2
        
        # Equity and returns
        initial_capital = 10000
        total_return = metrics.get('total_return', 0)
        return_color = curses.color_pair(1) if total_return > 0 else curses.color_pair(2)
        
        stdscr.addstr(row, 2, f"Equity: ${current_equity:.2f}", curses.color_pair(5))
        stdscr.addstr(row, 30, f"Return: ", curses.color_pair(5))
        stdscr.addstr(row, 38, self.format_percentage(total_return), return_color | curses.A_BOLD)
        
        # Cash and positions value
        row += 1
        cash = metrics.get('cash', initial_capital)
        positions_value = metrics.get('positions_value', 0)
        stdscr.addstr(row, 2, f"Cash: ${cash:.2f}", curses.color_pair(5))
        stdscr.addstr(row, 30, f"Positions: ${positions_value:.2f}", curses.color_pair(5))
        
        # Trading stats
        row += 2
        num_trades = metrics.get('num_trades', 0)
        win_rate = metrics.get('win_rate', 0)
        win_color = curses.color_pair(1) if win_rate > 50 else curses.color_pair(2)
        
        stdscr.addstr(row, 2, f"Trades: {num_trades}", curses.color_pair(5))
        stdscr.addstr(row, 20, f"Win Rate: ", curses.color_pair(5))
        stdscr.addstr(row, 30, f"{win_rate:.1f}%", win_color)
        
        drawdown = metrics.get('drawdown', 0)
        dd_color = curses.color_pair(2) if drawdown > 5 else curses.color_pair(3)
        stdscr.addstr(row, 40, f"Drawdown: {drawdown:.2f}%", dd_color)
        
        # Open Positions
        row += 3
        stdscr.addstr(row, 2, "━" * (width - 4), curses.color_pair(5))
        row += 1
        stdscr.addstr(row, 2, f"OPEN POSITIONS ({len(positions)})", curses.color_pair(4) | curses.A_BOLD)
        row += 1
        stdscr.addstr(row, 2, "━" * (width - 4), curses.color_pair(5))
        row += 2
        
        if positions:
            # Header
            stdscr.addstr(row, 2, "Symbol", curses.color_pair(6))
            stdscr.addstr(row, 15, "Qty", curses.color_pair(6))
            stdscr.addstr(row, 30, "Entry", curses.color_pair(6))
            stdscr.addstr(row, 45, "Current", curses.color_pair(6))
            stdscr.addstr(row, 60, "P&L", curses.color_pair(6))
            row += 1
            stdscr.addstr(row, 2, "-" * (width - 4), curses.color_pair(5))
            row += 1
            
            for symbol, pos in list(positions.items())[:5]:  # Show max 5 positions
                qty = pos.get('quantity', 0)
                avg_price = pos.get('avg_price', 0)
                current_value = pos.get('value', 0)
                
                stdscr.addstr(row, 2, symbol[:10], curses.color_pair(5))
                stdscr.addstr(row, 15, f"{qty:.6f}"[:12], curses.color_pair(5))
                stdscr.addstr(row, 30, f"${avg_price:.2f}", curses.color_pair(5))
                
                if qty > 0:
                    current_price = current_value / qty
                    pnl = (current_price - avg_price) * qty
                    pnl_pct = (current_price - avg_price) / avg_price * 100 if avg_price > 0 else 0
                    
                    stdscr.addstr(row, 45, f"${current_price:.2f}", curses.color_pair(5))
                    
                    pnl_color = curses.color_pair(1) if pnl > 0 else curses.color_pair(2)
                    stdscr.addstr(row, 60, f"${pnl:.2f} ({pnl_pct:+.1f}%)", pnl_color)
                row += 1
        else:
            stdscr.addstr(row, 2, "No open positions", curses.color_pair(3))
            row += 1
        
        # Recent Trades
        row += 2
        if row < height - 15:
            stdscr.addstr(row, 2, "━" * (width - 4), curses.color_pair(5))
            row += 1
            stdscr.addstr(row, 2, f"RECENT TRADES (Last {len(self.trade_history)})", curses.color_pair(4) | curses.A_BOLD)
            row += 1
            stdscr.addstr(row, 2, "━" * (width - 4), curses.color_pair(5))
            row += 2
            
            if self.trade_history:
                for trade in list(self.trade_history)[-5:]:  # Show last 5 trades
                    if row >= height - 3:
                        break
                    
                    timestamp = trade.get('timestamp', '')
                    if timestamp:
                        try:
                            dt = datetime.fromisoformat(timestamp)
                            time_str = dt.strftime('%H:%M:%S')
                        except:
                            time_str = "Unknown"
                    else:
                        time_str = "Unknown"
                    
                    symbol = trade.get('symbol', 'Unknown')
                    side = trade.get('side', 'Unknown')
                    qty = trade.get('quantity', 0)
                    price = trade.get('price', 0)
                    
                    side_color = curses.color_pair(1) if side.lower() == 'buy' else curses.color_pair(2)
                    trade_str = f"{time_str} {side.upper():4} {symbol:10} {qty:.6f} @ ${price:.2f}"
                    
                    # Add P&L for sells
                    if side.lower() == 'sell' and 'pnl' in trade:
                        pnl = trade['pnl']
                        pnl_color = curses.color_pair(1) if pnl > 0 else curses.color_pair(2)
                        trade_str += f" P&L: ${pnl:.2f}"
                    
                    stdscr.addstr(row, 2, trade_str[:width-4], side_color)
                    row += 1
            else:
                stdscr.addstr(row, 2, "No trades yet", curses.color_pair(3))
        
        # Footer
        footer_row = height - 2
        stdscr.addstr(footer_row, 2, "━" * (width - 4), curses.color_pair(5))
        footer_row += 1
        stdscr.addstr(footer_row, 2, "Press 'q' to quit | 'r' to refresh | Updates every 2 seconds", curses.color_pair(3))
        
        stdscr.refresh()
    
    def run(self, stdscr):
        """Main monitoring loop."""
        # Setup
        curses.curs_set(0)  # Hide cursor
        stdscr.nodelay(1)   # Non-blocking input
        stdscr.timeout(100) # Refresh timeout
        
        last_data = {}
        no_data_counter = 0
        
        while True:
            # Check for quit
            key = stdscr.getch()
            if key == ord('q'):
                break
            elif key == ord('r'):
                stdscr.clear()
            
            # Load latest data
            new_data = self.load_latest_data()
            if new_data:
                last_data = new_data
                no_data_counter = 0
            else:
                no_data_counter += 1
            
            # Draw dashboard
            if last_data:
                try:
                    self.draw_dashboard(stdscr, last_data)
                except curses.error:
                    pass  # Terminal resize or other issue
            else:
                # No data message
                height, width = stdscr.getmaxyx()
                msg = "Waiting for paper trading data..."
                stdscr.clear()
                stdscr.addstr(height // 2, (width - len(msg)) // 2, msg)
                stdscr.addstr(height // 2 + 2, (width - 40) // 2, "Start a paper trading session to see data")
                stdscr.refresh()
            
            # Wait before next update
            time.sleep(2)


def main():
    """Main entry point."""
    results_dir = Path(__file__).parent.parent / 'results'
    
    if not results_dir.exists():
        print("Results directory not found. Creating...")
        results_dir.mkdir(exist_ok=True)
    
    print("Starting Live Paper Trading Monitor...")
    print("This will display real-time updates from your paper trading sessions.")
    print("Start a paper trading session in another terminal to see live data.")
    print("\nPress Enter to continue...")
    input()
    
    monitor = LiveMonitor(results_dir)
    
    try:
        curses.wrapper(monitor.run)
    except KeyboardInterrupt:
        pass
    
    print("\nMonitor stopped.")


if __name__ == "__main__":
    main()