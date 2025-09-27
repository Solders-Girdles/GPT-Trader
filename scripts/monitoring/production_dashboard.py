#!/usr/bin/env python3
"""
Production Trading Dashboard.

Real-time dashboard for monitoring portfolio equity, fees, margin,
liquidity conditions, and system health.
"""

import asyncio
import json
import sys
from decimal import Decimal
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from bot_v2.features.live_trade.portfolio_valuation import PortfolioValuationService
from bot_v2.features.live_trade.fees_engine import FeesEngine
from bot_v2.features.live_trade.margin_monitor import MarginStateMonitor
from bot_v2.features.live_trade.liquidity_service import LiquidityService
from bot_v2.features.live_trade.order_policy import create_order_policy_matrix


class ProductionDashboard:
    """
    Production trading dashboard for comprehensive monitoring.
    
    Aggregates data from all production services and provides
    real-time health monitoring and alerting.
    """
    
    def __init__(self, environment: str = "sandbox"):
        self.environment = environment
        self.start_time = datetime.now()
        
        # Initialize services (would be dependency injected in real system)
        self.portfolio_service = PortfolioValuationService()
        self.fees_engine = FeesEngine()
        self.margin_monitor = MarginStateMonitor()
        self.liquidity_service = LiquidityService()
        
        # Dashboard state
        self._health_status = "UNKNOWN"
        self._last_update = datetime.now()
        self._alert_count = 0
        self._system_metrics = {}
        
    async def initialize(self):
        """Initialize dashboard services."""
        print(f"🚀 Initializing Production Dashboard ({self.environment})")
        
        # Initialize order policy matrix
        self.policy_matrix = await create_order_policy_matrix(self.environment)
        
        # Initialize fee tier
        await self.fees_engine.tier_resolver.get_current_tier()
        
        print("✅ Dashboard initialized successfully")
    
    async def update_market_data(self, market_data: Dict):
        """Update services with latest market data."""
        # Update portfolio valuation with marks
        if 'marks' in market_data:
            self.portfolio_service.update_mark_prices(market_data['marks'])
        
        # Update liquidity service with order book
        if 'order_books' in market_data:
            for symbol, book_data in market_data['order_books'].items():
                if 'bids' in book_data and 'asks' in book_data:
                    self.liquidity_service.analyze_order_book(
                        symbol=symbol,
                        bids=[(Decimal(str(p)), Decimal(str(s))) for p, s in book_data['bids']],
                        asks=[(Decimal(str(p)), Decimal(str(s))) for p, s in book_data['asks']]
                    )
        
        # Update trade data
        if 'trades' in market_data:
            for trade in market_data['trades']:
                self.liquidity_service.update_trade_data(
                    symbol=trade['symbol'],
                    price=Decimal(str(trade['price'])),
                    size=Decimal(str(trade['size']))
                )
        
        self._last_update = datetime.now()
    
    async def update_account_data(self, account_data: Dict):
        """Update services with latest account data."""
        # Mock balance and position objects for demonstration
        balances = account_data.get('balances', [])
        positions = account_data.get('positions', [])
        
        # Update portfolio service
        self.portfolio_service.update_account_data(balances, positions)
        
        # Compute current margin state
        total_equity = sum(Decimal(str(b.get('total', 0))) for b in balances)
        cash_balance = sum(Decimal(str(b.get('available', 0))) for b in balances)
        
        position_dict = {}
        for pos in positions:
            if pos.get('quantity', 0) != 0:
                position_dict[pos['symbol']] = {
                    'quantity': Decimal(str(pos['quantity'])),
                    'mark_price': Decimal(str(pos.get('mark_price', 0)))
                }
        
        await self.margin_monitor.compute_margin_state(
            total_equity=total_equity,
            cash_balance=cash_balance,
            positions=position_dict
        )
    
    async def generate_dashboard_data(self) -> Dict:
        """Generate comprehensive dashboard data."""
        now = datetime.now()
        uptime = now - self.start_time
        
        # Portfolio metrics
        portfolio_snapshot = self.portfolio_service.compute_current_valuation()
        equity_curve = self.portfolio_service.get_equity_curve(hours_back=24)
        daily_metrics = self.portfolio_service.get_daily_metrics()
        
        # Fee metrics
        fee_summary = await self.fees_engine.get_fee_summary(hours_back=24)
        current_tier = await self.fees_engine.tier_resolver.get_current_tier()
        
        # Margin metrics
        margin_state = self.margin_monitor.get_current_state()
        margin_history = self.margin_monitor.get_margin_history(hours_back=24)
        window_transition = await self.margin_monitor.check_window_transition()
        
        # Liquidity metrics for tracked symbols
        liquidity_data = {}
        for symbol in ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD"]:
            snapshot = self.liquidity_service.get_liquidity_snapshot(symbol)
            if snapshot:
                liquidity_data[symbol] = snapshot
        
        # Policy summary
        policy_summary = self.policy_matrix.get_policy_summary()
        
        # Health assessment
        health_status = self._assess_system_health(
            portfolio_snapshot, margin_state, fee_summary, liquidity_data
        )
        
        return {
            'meta': {
                'timestamp': now.isoformat(),
                'environment': self.environment,
                'uptime_seconds': int(uptime.total_seconds()),
                'health_status': health_status['status'],
                'alert_count': health_status['alert_count'],
                'last_update': self._last_update.isoformat()
            },
            'portfolio': {
                'snapshot': portfolio_snapshot.to_dict(),
                'equity_curve': equity_curve,
                'daily_metrics': daily_metrics,
                'position_count': len(portfolio_snapshot.positions),
                'stale_marks': list(portfolio_snapshot.stale_marks),
                'missing_positions': list(portfolio_snapshot.missing_positions)
            },
            'fees': {
                'summary': fee_summary,
                'current_tier': {
                    'name': current_tier.tier_name,
                    'maker_rate': float(current_tier.maker_rate),
                    'taker_rate': float(current_tier.taker_rate)
                },
                'tier_volume_threshold': float(current_tier.volume_threshold)
            },
            'margin': {
                'current_state': margin_state.to_dict() if margin_state else None,
                'history_24h': margin_history,
                'window_transition': window_transition,
                'utilization_alert': margin_state.margin_utilization > Decimal('0.8') if margin_state else False,
                'margin_call_risk': margin_state.is_margin_call if margin_state else False
            },
            'liquidity': liquidity_data,
            'policies': policy_summary,
            'health': health_status
        }
    
    def _assess_system_health(
        self,
        portfolio_snapshot,
        margin_state,
        fee_summary,
        liquidity_data
    ) -> Dict:
        """Assess overall system health."""
        alerts = []
        warnings = []
        
        # Portfolio health
        if portfolio_snapshot.stale_marks:
            alerts.append(f"Stale marks for {len(portfolio_snapshot.stale_marks)} symbols")
        
        if portfolio_snapshot.missing_positions:
            alerts.append(f"Missing position data for {len(portfolio_snapshot.missing_positions)} symbols")
        
        # Margin health
        if margin_state:
            if margin_state.is_liquidation_risk:
                alerts.append("LIQUIDATION RISK - Immediate attention required")
            elif margin_state.is_margin_call:
                alerts.append("MARGIN CALL - Reduce positions")
            elif margin_state.margin_utilization > Decimal('0.8'):
                warnings.append(f"High margin utilization: {margin_state.margin_utilization:.1%}")
        
        # Fee health
        if fee_summary.get('total_fees', 0) > 1000:  # $1k in fees per day
            warnings.append(f"High daily fees: ${fee_summary['total_fees']:.2f}")
        
        # Liquidity health
        poor_liquidity_symbols = []
        for symbol, data in liquidity_data.items():
            if data.get('condition') in ['poor', 'critical']:
                poor_liquidity_symbols.append(symbol)
        
        if poor_liquidity_symbols:
            warnings.append(f"Poor liquidity in: {', '.join(poor_liquidity_symbols)}")
        
        # Determine overall status
        if alerts:
            status = "CRITICAL"
        elif warnings:
            status = "WARNING"
        else:
            status = "HEALTHY"
        
        self._health_status = status
        self._alert_count = len(alerts) + len(warnings)
        
        return {
            'status': status,
            'alerts': alerts,
            'warnings': warnings,
            'alert_count': len(alerts),
            'warning_count': len(warnings),
            'assessment_time': datetime.now().isoformat()
        }
    
    async def generate_alerts(self) -> List[Dict]:
        """Generate actionable alerts."""
        alerts = []
        now = datetime.now()
        
        # Check margin state
        margin_state = self.margin_monitor.get_current_state()
        if margin_state:
            if margin_state.is_liquidation_risk:
                alerts.append({
                    'severity': 'CRITICAL',
                    'type': 'LIQUIDATION_RISK',
                    'message': f'Account at liquidation risk - Equity: ${margin_state.total_equity:.2f}',
                    'action': 'Reduce positions immediately',
                    'timestamp': now.isoformat()
                })
            elif margin_state.margin_utilization > Decimal('0.9'):
                alerts.append({
                    'severity': 'HIGH',
                    'type': 'MARGIN_UTILIZATION',
                    'message': f'Margin utilization at {margin_state.margin_utilization:.1%}',
                    'action': 'Consider reducing position sizes',
                    'timestamp': now.isoformat()
                })
        
        # Check for window transitions
        window_transition = await self.margin_monitor.check_window_transition()
        if window_transition and window_transition['should_reduce_risk']:
            alerts.append({
                'severity': 'MEDIUM',
                'type': 'MARGIN_WINDOW_CHANGE',
                'message': f'Margin window changing to {window_transition["next_window"]} in {window_transition["minutes_until"]} minutes',
                'action': f'Consider reducing leverage to {window_transition["next_max_leverage"]:.1f}x',
                'timestamp': now.isoformat()
            })
        
        # Check portfolio health
        portfolio_snapshot = self.portfolio_service.compute_current_valuation()
        if len(portfolio_snapshot.stale_marks) > 0:
            alerts.append({
                'severity': 'MEDIUM',
                'type': 'STALE_MARKET_DATA',
                'message': f'Stale marks for {len(portfolio_snapshot.stale_marks)} symbols',
                'action': 'Check market data connectivity',
                'timestamp': now.isoformat()
            })
        
        return alerts
    
    def format_console_output(self, dashboard_data: Dict) -> str:
        """Format dashboard data for console display."""
        meta = dashboard_data['meta']
        portfolio = dashboard_data['portfolio']
        margin = dashboard_data['margin']
        fees = dashboard_data['fees']
        health = dashboard_data['health']
        
        output = f"""
🔥 PRODUCTION TRADING DASHBOARD ({meta['environment'].upper()}) 🔥
{'='*70}
Status: {health['status']} | Uptime: {meta['uptime_seconds']//3600}h {(meta['uptime_seconds']%3600)//60}m | Alerts: {health['alert_count']}

💰 PORTFOLIO
  Equity: ${portfolio['snapshot']['total_equity_usd']:,.2f}
  Cash: ${portfolio['snapshot']['cash_balance']:,.2f} 
  Positions Value: ${portfolio['snapshot']['positions_value']:,.2f}
  Realized PnL: ${portfolio['snapshot']['realized_pnl']:+,.2f}
  Unrealized PnL: ${portfolio['snapshot']['unrealized_pnl']:+,.2f}
  Positions: {portfolio['position_count']}

📊 MARGIN & RISK
  Utilization: {margin['current_state']['margin_utilization']:.1%} 
  Leverage: {margin['current_state']['leverage']:.2f}x
  Available: ${margin['current_state']['margin_available']:,.2f}
  Window: {margin['current_state']['window'].upper()}
  
💸 FEES (24h)
  Total Paid: ${fees['summary']['total_fees']:.2f}
  Maker: ${fees['summary']['maker_fees']:.2f} | Taker: ${fees['summary']['taker_fees']:.2f}
  Trade Count: {fees['summary']['trade_count']}
  Current Tier: {fees['current_tier']['name']} ({fees['current_tier']['maker_rate']:.3%}/{fees['current_tier']['taker_rate']:.3%})
"""
        
        # Add alerts
        if health['alerts']:
            output += f"\n🚨 CRITICAL ALERTS:\n"
            for alert in health['alerts']:
                output += f"  • {alert}\n"
        
        if health['warnings']:
            output += f"\n⚠️  WARNINGS:\n"
            for warning in health['warnings']:
                output += f"  • {warning}\n"
        
        output += f"\nLast Update: {meta['last_update']}\n{'='*70}"
        
        return output


async def run_dashboard_demo():
    """Run dashboard demo with mock data."""
    dashboard = ProductionDashboard("sandbox")
    await dashboard.initialize()
    
    print("📈 Starting dashboard demo...")
    
    # Mock market data update
    market_data = {
        'marks': {
            'BTC-USD': Decimal('50000'),
            'ETH-USD': Decimal('3000'),
            'SOL-USD': Decimal('100')
        },
        'order_books': {
            'BTC-USD': {
                'bids': [[49950, 10], [49940, 5], [49930, 8]],
                'asks': [[50050, 8], [50060, 6], [50070, 10]]
            }
        },
        'trades': [
            {'symbol': 'BTC-USD', 'price': 50000, 'size': 0.5}
        ]
    }
    
    await dashboard.update_market_data(market_data)
    
    # Mock account data
    account_data = {
        'balances': [
            {'currency': 'USD', 'total': 100000, 'available': 75000}
        ],
        'positions': [
            {'symbol': 'BTC-USD', 'quantity': 2, 'mark_price': 50000}
        ]
    }
    
    await dashboard.update_account_data(account_data)
    
    # Generate dashboard
    dashboard_data = await dashboard.generate_dashboard_data()
    console_output = dashboard.format_console_output(dashboard_data)
    
    print(console_output)
    
    # Generate alerts
    alerts = await dashboard.generate_alerts()
    if alerts:
        print("\n📢 ACTIVE ALERTS:")
        for alert in alerts:
            print(f"  {alert['severity']}: {alert['message']}")
    
    # Save dashboard data to file
    output_file = Path("dashboard_output.json")
    with open(output_file, 'w') as f:
        json.dump(dashboard_data, f, indent=2, default=str)
    
    print(f"\n💾 Dashboard data saved to {output_file}")
    
    return dashboard_data


async def main():
    """Main dashboard execution."""
    try:
        dashboard_data = await run_dashboard_demo()
        print("\n✅ Dashboard demo completed successfully")
        return 0
    except Exception as e:
        print(f"\n❌ Dashboard demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)