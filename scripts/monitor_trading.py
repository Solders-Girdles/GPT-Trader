#!/usr/bin/env python3
"""
Real-time monitoring script for GPT-Trader Perpetuals Bot.

Provides live monitoring of positions, P&L, risk metrics, and system health.

Usage:
    poetry run python scripts/monitor_trading.py
    poetry run python scripts/monitor_trading.py --interval 5
    poetry run python scripts/monitor_trading.py --profile canary
"""

from __future__ import annotations

import os
import sys
import time
import json
import argparse
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any
from pathlib import Path
import threading
import signal

# Color codes for terminal output
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    CLEAR = '\033[2J\033[H'  # Clear screen and move cursor to top


class TradingMonitor:
    """Real-time trading monitor."""
    
    def __init__(self, profile: str = "canary", interval: int = 10):
        self.profile = profile
        self.interval = interval
        self.running = True
        self.broker = None
        self.start_time = datetime.now(timezone.utc)
        self.initial_equity: Optional[Decimal] = None
        self.peak_equity: Optional[Decimal] = None
        self.session_trades = 0
        self.session_pnl = Decimal("0")
        
        # Event tracking
        self.last_order_time: Optional[datetime] = None
        self.error_count = 0
        self.warning_count = 0
        
        # Initialize broker connection
        self._init_broker()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print("\n\nShutting down monitor...")
        self.running = False
        sys.exit(0)
    
    def _init_broker(self) -> None:
        """Initialize broker connection."""
        try:
            from bot_v2.orchestration.broker_factory import create_brokerage
            
            # Use mock if in paper mode
            if os.getenv("PERPS_PAPER", "0") == "1":
                from bot_v2.orchestration.mock_broker import MockBroker
                self.broker = MockBroker()
                print(f"{Colors.YELLOW}Using mock broker (paper mode){Colors.RESET}")
            else:
                self.broker = create_brokerage()
                if not self.broker.connect():
                    raise RuntimeError("Failed to connect to broker")
                print(f"{Colors.GREEN}Connected to live broker{Colors.RESET}")
        except Exception as e:
            print(f"{Colors.RED}Failed to initialize broker: {e}{Colors.RESET}")
            sys.exit(1)
    
    def clear_screen(self) -> None:
        """Clear terminal screen."""
        print(Colors.CLEAR, end='')
    
    def format_number(self, value: Decimal, decimals: int = 2, prefix: str = "") -> str:
        """Format number with color coding."""
        if value > 0:
            color = Colors.GREEN
            sign = "+"
        elif value < 0:
            color = Colors.RED
            sign = ""
        else:
            color = Colors.WHITE
            sign = ""
        
        return f"{color}{prefix}{sign}{value:.{decimals}f}{Colors.RESET}"
    
    def format_percentage(self, value: Decimal) -> str:
        """Format percentage with color coding."""
        if value > 0:
            color = Colors.GREEN
            sign = "+"
        elif value < 0:
            color = Colors.RED
            sign = ""
        else:
            color = Colors.WHITE
            sign = ""
        
        return f"{color}{sign}{value:.2f}%{Colors.RESET}"
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions."""
        try:
            positions = self.broker.list_positions()
            return [
                {
                    "symbol": getattr(p, "symbol", ""),
                    "qty": getattr(p, "qty", Decimal("0")),
                    "side": getattr(p, "side", ""),
                    "entry_price": getattr(p, "entry_price", Decimal("0")),
                    "mark_price": getattr(p, "mark_price", Decimal("0")),
                    "unrealized_pnl": getattr(p, "unrealized_pnl", Decimal("0")),
                    "margin": getattr(p, "margin", Decimal("0")),
                }
                for p in positions
            ]
        except Exception:
            return []
    
    def get_account_info(self) -> Dict[str, Any]:
        """Get account information."""
        try:
            balances = self.broker.list_balances()
            usd_balance = next((b for b in balances if b.asset == "USD"), None)
            
            equity = usd_balance.available if usd_balance else Decimal("0")
            
            # Track initial and peak equity
            if self.initial_equity is None:
                self.initial_equity = equity
            if self.peak_equity is None or equity > self.peak_equity:
                self.peak_equity = equity
            
            return {
                "equity": equity,
                "initial_equity": self.initial_equity,
                "peak_equity": self.peak_equity,
                "session_pnl": equity - self.initial_equity if self.initial_equity else Decimal("0"),
                "drawdown": (self.peak_equity - equity) if self.peak_equity else Decimal("0"),
            }
        except Exception:
            return {
                "equity": Decimal("0"),
                "initial_equity": Decimal("0"),
                "peak_equity": Decimal("0"),
                "session_pnl": Decimal("0"),
                "drawdown": Decimal("0"),
            }
    
    def get_recent_orders(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent orders."""
        try:
            orders = self.broker.list_orders(limit=limit)
            return [
                {
                    "id": o.id[:8] if hasattr(o, "id") else "",
                    "symbol": getattr(o, "symbol", ""),
                    "side": getattr(o, "side", "").value if hasattr(getattr(o, "side", ""), "value") else "",
                    "qty": getattr(o, "qty", Decimal("0")),
                    "status": getattr(o, "status", "").value if hasattr(getattr(o, "status", ""), "value") else "",
                    "created_at": getattr(o, "created_at", None),
                }
                for o in orders[:limit]
            ]
        except Exception:
            return []
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get bot health status from file."""
        try:
            health_file = Path(f"data/perps_bot/{self.profile}/health.json")
            if health_file.exists():
                with open(health_file) as f:
                    return json.load(f)
            return {"ok": False, "message": "Health file not found"}
        except Exception:
            return {"ok": False, "message": "Error reading health"}
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Calculate risk metrics."""
        positions = self.get_positions()
        account = self.get_account_info()
        
        total_margin = sum(p["margin"] for p in positions)
        total_notional = sum(
            p["qty"] * p["mark_price"] for p in positions
        )
        
        leverage = (
            total_notional / account["equity"]
            if account["equity"] > 0
            else Decimal("0")
        )
        
        drawdown_pct = (
            (account["drawdown"] / account["peak_equity"]) * 100
            if account["peak_equity"] > 0
            else Decimal("0")
        )
        
        return {
            "leverage": leverage,
            "total_margin": total_margin,
            "total_notional": total_notional,
            "drawdown_pct": drawdown_pct,
            "position_count": len(positions),
        }
    
    def display_dashboard(self) -> None:
        """Display monitoring dashboard."""
        self.clear_screen()
        
        # Header
        print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 80}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}GPT-TRADER MONITORING DASHBOARD{Colors.RESET}")
        print(f"{Colors.CYAN}Profile: {self.profile} | Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}{Colors.RESET}")
        print(f"{Colors.CYAN}{'=' * 80}{Colors.RESET}\n")
        
        # Account Information
        account = self.get_account_info()
        print(f"{Colors.BOLD}📊 ACCOUNT{Colors.RESET}")
        print(f"  Equity: {self.format_number(account['equity'], prefix='$')}")
        print(f"  Session P&L: {self.format_number(account['session_pnl'], prefix='$')} "
              f"({self.format_percentage((account['session_pnl'] / account['initial_equity']) * 100 if account['initial_equity'] > 0 else Decimal('0'))})")
        print(f"  Peak Equity: ${account['peak_equity']:.2f}")
        print(f"  Drawdown: {self.format_number(account['drawdown'], prefix='$')} "
              f"({self.format_percentage((account['drawdown'] / account['peak_equity']) * 100 if account['peak_equity'] > 0 else Decimal('0'))})")
        print()
        
        # Risk Metrics
        risk = self.get_risk_metrics()
        print(f"{Colors.BOLD}⚠️  RISK METRICS{Colors.RESET}")
        print(f"  Leverage: {risk['leverage']:.2f}x")
        print(f"  Total Margin: ${risk['total_margin']:.2f}")
        print(f"  Total Notional: ${risk['total_notional']:.2f}")
        print(f"  Position Count: {risk['position_count']}")
        print()
        
        # Positions
        positions = self.get_positions()
        if positions:
            print(f"{Colors.BOLD}📈 POSITIONS{Colors.RESET}")
            print(f"  {'Symbol':<12} {'Side':<6} {'Qty':<10} {'Entry':<10} {'Mark':<10} {'P&L':<12} {'%':<8}")
            print(f"  {'-' * 76}")
            
            for pos in positions:
                pnl_pct = ((pos['mark_price'] - pos['entry_price']) / pos['entry_price'] * 100
                          if pos['entry_price'] > 0 else Decimal('0'))
                if pos['side'] == 'short':
                    pnl_pct = -pnl_pct
                
                print(f"  {pos['symbol']:<12} {pos['side']:<6} {pos['qty']:<10.4f} "
                      f"${pos['entry_price']:<9.2f} ${pos['mark_price']:<9.2f} "
                      f"{self.format_number(pos['unrealized_pnl'], prefix='$'):<12} "
                      f"{self.format_percentage(pnl_pct):<8}")
        else:
            print(f"{Colors.BOLD}📈 POSITIONS{Colors.RESET}")
            print(f"  No open positions")
        print()
        
        # Recent Orders
        orders = self.get_recent_orders(5)
        if orders:
            print(f"{Colors.BOLD}📝 RECENT ORDERS{Colors.RESET}")
            print(f"  {'ID':<10} {'Symbol':<12} {'Side':<6} {'Qty':<10} {'Status':<12}")
            print(f"  {'-' * 50}")
            
            for order in orders:
                status_color = (
                    Colors.GREEN if order['status'] == 'FILLED' else
                    Colors.YELLOW if order['status'] in ['PENDING', 'SUBMITTED'] else
                    Colors.RED
                )
                print(f"  {order['id']:<10} {order['symbol']:<12} {order['side']:<6} "
                      f"{order['qty']:<10.4f} {status_color}{order['status']:<12}{Colors.RESET}")
        print()
        
        # Health Status
        health = self.get_health_status()
        health_color = Colors.GREEN if health.get("ok") else Colors.RED
        health_status = "HEALTHY" if health.get("ok") else "UNHEALTHY"
        
        print(f"{Colors.BOLD}🏥 SYSTEM HEALTH{Colors.RESET}")
        print(f"  Status: {health_color}{health_status}{Colors.RESET}")
        if health.get("message"):
            print(f"  Message: {health.get('message')}")
        if health.get("error"):
            print(f"  Error: {Colors.RED}{health.get('error')}{Colors.RESET}")
        print()
        
        # Session Statistics
        uptime = datetime.now(timezone.utc) - self.start_time
        hours = uptime.total_seconds() / 3600
        
        print(f"{Colors.BOLD}📊 SESSION STATISTICS{Colors.RESET}")
        print(f"  Uptime: {int(hours)}h {int((hours % 1) * 60)}m")
        print(f"  Errors: {self.error_count}")
        print(f"  Warnings: {self.warning_count}")
        print()
        
        # Footer
        print(f"{Colors.CYAN}{'=' * 80}{Colors.RESET}")
        print(f"{Colors.CYAN}Refreshing every {self.interval} seconds | Press Ctrl+C to exit{Colors.RESET}")
    
    def check_alerts(self) -> None:
        """Check for alert conditions."""
        account = self.get_account_info()
        risk = self.get_risk_metrics()
        
        # Check for high leverage
        if risk["leverage"] > 3:
            self.warning_count += 1
            print(f"\n{Colors.YELLOW}⚠️  WARNING: High leverage detected: {risk['leverage']:.2f}x{Colors.RESET}")
        
        # Check for large drawdown
        if risk["drawdown_pct"] > 5:
            self.warning_count += 1
            print(f"\n{Colors.YELLOW}⚠️  WARNING: Significant drawdown: {risk['drawdown_pct']:.2f}%{Colors.RESET}")
        
        # Check for large losses
        if account["session_pnl"] < -100:
            self.error_count += 1
            print(f"\n{Colors.RED}🚨 ALERT: Session loss exceeds $100: ${account['session_pnl']:.2f}{Colors.RESET}")
    
    def run(self) -> None:
        """Run the monitoring loop."""
        print(f"{Colors.GREEN}Starting monitor for profile: {self.profile}{Colors.RESET}")
        print(f"{Colors.GREEN}Update interval: {self.interval} seconds{Colors.RESET}\n")
        
        while self.running:
            try:
                self.display_dashboard()
                self.check_alerts()
                time.sleep(self.interval)
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.error_count += 1
                print(f"\n{Colors.RED}Error updating dashboard: {e}{Colors.RESET}")
                time.sleep(self.interval)
        
        print(f"\n{Colors.CYAN}Monitor stopped. Final statistics:{Colors.RESET}")
        account = self.get_account_info()
        print(f"Session P&L: {self.format_number(account['session_pnl'], prefix='$')}")
        print(f"Total Errors: {self.error_count}")
        print(f"Total Warnings: {self.warning_count}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Real-time monitoring for GPT-Trader"
    )
    parser.add_argument(
        "--profile", "-p",
        default="canary",
        choices=["dev", "canary", "prod"],
        help="Trading profile to monitor (default: canary)"
    )
    parser.add_argument(
        "--interval", "-i",
        type=int,
        default=10,
        help="Update interval in seconds (default: 10)"
    )
    
    args = parser.parse_args()
    
    # Check if bot is running
    health_file = Path(f"data/perps_bot/{args.profile}/health.json")
    if not health_file.exists():
        print(f"{Colors.YELLOW}Warning: Bot health file not found. Bot may not be running.{Colors.RESET}")
        print(f"Start the bot with: poetry run perps-bot --profile {args.profile}")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
    
    # Start monitor
    monitor = TradingMonitor(profile=args.profile, interval=args.interval)
    
    try:
        monitor.run()
    except KeyboardInterrupt:
        print(f"\n{Colors.CYAN}Monitor stopped by user{Colors.RESET}")
    except Exception as e:
        print(f"\n{Colors.RED}Monitor error: {e}{Colors.RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()
