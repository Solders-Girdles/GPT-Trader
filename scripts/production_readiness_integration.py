#!/usr/bin/env python3
"""
Production Readiness Integration Script.

Comprehensive integration test and demonstration of all production-ready
components working together: PortfolioValuation, FeesEngine, MarginMonitor,
LiquidityService, and OrderPolicy.
"""

import asyncio
import json
import sys
from decimal import Decimal
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from bot_v2.features.live_trade.portfolio_valuation import PortfolioValuationService, PortfolioSnapshot
from bot_v2.features.live_trade.fees_engine import FeesEngine, FeeType
from bot_v2.features.live_trade.margin_monitor import MarginStateMonitor, MarginWindow
from bot_v2.features.live_trade.liquidity_service import LiquidityService, LiquidityCondition
from bot_v2.features.live_trade.order_policy import create_order_policy_matrix
from bot_v2.features.live_trade.pnl_tracker import PnLTracker
from bot_v2.features.brokerages.core.interfaces import Balance, Position


class ProductionTradingSystem:
    """
    Integrated production trading system.
    
    Demonstrates all production-ready components working together
    in a cohesive trading workflow with proper financial correctness.
    """
    
    def __init__(self, initial_capital: Decimal = Decimal('100000')):
        self.initial_capital = initial_capital
        self.current_equity = initial_capital
        
        # Initialize core services
        self.pnl_tracker = PnLTracker()
        self.portfolio_service = PortfolioValuationService(pnl_tracker=self.pnl_tracker)
        self.fees_engine = FeesEngine()
        self.margin_monitor = MarginStateMonitor()
        self.liquidity_service = LiquidityService()
        
        # System state
        self._positions = {}
        self._cash_balance = initial_capital
        self._trade_history = []
        
        print(f"🚀 Production Trading System initialized with ${initial_capital:,.2f}")
    
    async def initialize(self):
        """Initialize all services."""
        self.policy_matrix = await create_order_policy_matrix("sandbox")
        await self.fees_engine.tier_resolver.get_current_tier()
        print("✅ All services initialized")
    
    async def execute_trade(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        price: Decimal,
        order_type: str = "LIMIT",
        tif: str = "GTC",
        post_only: bool = False
    ) -> Dict:
        """
        Execute trade with full production pipeline.
        
        This demonstrates the complete workflow:
        1. Order validation against policy
        2. Fee calculation and cost estimation
        3. Margin requirement checks
        4. Liquidity impact analysis
        5. Trade execution simulation
        6. Portfolio and PnL updates
        """
        print(f"\n📋 Executing Trade: {symbol} {side} {quantity} @ ${price}")
        
        trade_result = {
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': price,
            'timestamp': datetime.now().isoformat(),
            'success': False,
            'reason': '',
            'fees': Decimal('0'),
            'impact_bps': Decimal('0')
        }
        
        try:
            # Step 1: Order Policy Validation
            print("  1️⃣ Validating order policy...")
            allowed, reason = self.policy_matrix.validate_order(
                symbol=symbol,
                order_type=order_type,
                tif=tif,
                quantity=quantity,
                price=price,
                post_only=post_only
            )
            
            if not allowed:
                trade_result['reason'] = f"Policy violation: {reason}"
                print(f"  ❌ Order rejected: {reason}")
                return trade_result
            
            print(f"  ✅ Order policy validated")
            
            # Step 2: Fee Calculation
            print("  2️⃣ Calculating fees...")
            notional = quantity * price
            fee_calc = await self.fees_engine.calculate_order_fee(
                symbol=symbol,
                notional=notional,
                is_post_only=post_only
            )
            
            print(f"  💸 Estimated fee: ${fee_calc.fee_amount:.2f} ({fee_calc.fee_type.value})")
            
            # Step 3: Margin Requirement Check
            print("  3️⃣ Checking margin requirements...")
            
            # Update current positions for margin calculation
            position_data = {}
            for pos_symbol, pos_info in self._positions.items():
                position_data[pos_symbol] = {
                    'quantity': pos_info['quantity'],
                    'mark_price': pos_info.get('mark_price', price if pos_symbol == symbol else Decimal('50000'))
                }
            
            # Simulate position after trade
            if side == 'buy':
                new_quantity = position_data.get(symbol, {}).get('quantity', Decimal('0')) + quantity
            else:
                new_quantity = position_data.get(symbol, {}).get('quantity', Decimal('0')) - quantity
            
            position_data[symbol] = {
                'quantity': new_quantity,
                'mark_price': price
            }
            
            # Calculate margin state
            total_cost = notional + fee_calc.fee_amount if side == 'buy' else notional - fee_calc.fee_amount
            projected_cash = self._cash_balance - total_cost if side == 'buy' else self._cash_balance + total_cost
            projected_equity = projected_cash  # Simplified - would include unrealized PnL
            
            margin_snapshot = await self.margin_monitor.compute_margin_state(
                total_equity=projected_equity,
                cash_balance=projected_cash,
                positions=position_data
            )
            
            if margin_snapshot.is_margin_call:
                trade_result['reason'] = "Trade would cause margin call"
                print(f"  ❌ Margin call risk - utilization would be {margin_snapshot.margin_utilization:.1%}")
                return trade_result
            
            print(f"  ✅ Margin check passed - utilization: {margin_snapshot.margin_utilization:.1%}")
            
            # Step 4: Liquidity Impact Analysis
            print("  4️⃣ Analyzing market impact...")
            
            # Mock order book for impact analysis
            mid_price = price
            spread = mid_price * Decimal('0.001')  # 10bps spread
            
            mock_bids = [(mid_price - spread/2 - i*Decimal('10'), Decimal('10')) for i in range(10)]
            mock_asks = [(mid_price + spread/2 + i*Decimal('10'), Decimal('10')) for i in range(10)]
            
            self.liquidity_service.analyze_order_book(symbol, mock_bids, mock_asks)
            
            impact_estimate = self.liquidity_service.estimate_market_impact(
                symbol=symbol,
                side=side,
                quantity=quantity,
                book_data=(mock_bids, mock_asks)
            )
            
            print(f"  📊 Estimated impact: {impact_estimate.estimated_impact_bps:.2f}bps")
            
            if impact_estimate.recommended_slicing:
                print(f"  ⚠️  Recommend slicing - max slice: {impact_estimate.max_slice_size}")
            
            # Step 5: Execute Trade
            print("  5️⃣ Executing trade...")
            
            # Update positions
            if symbol not in self._positions:
                self._positions[symbol] = {
                    'quantity': Decimal('0'),
                    'avg_price': Decimal('0'),
                    'mark_price': price
                }
            
            # Record trade in PnL tracker
            is_reduce = (
                (side == 'sell' and self._positions[symbol]['quantity'] > 0) or
                (side == 'buy' and self._positions[symbol]['quantity'] < 0)
            )
            
            pnl_result = self.portfolio_service.update_trade(
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=price,
                fees=fee_calc.fee_amount,
                is_reduce=is_reduce
            )
            
            # Update position tracking
            if side == 'buy':
                old_qty = self._positions[symbol]['quantity']
                old_price = self._positions[symbol]['avg_price']
                new_qty = old_qty + quantity
                
                if old_qty >= 0:  # Adding to long or opening long
                    new_avg_price = ((old_qty * old_price) + (quantity * price)) / new_qty if new_qty != 0 else Decimal('0')
                else:  # Reducing short
                    new_avg_price = old_price if new_qty < 0 else price
                
                self._positions[symbol]['quantity'] = new_qty
                self._positions[symbol]['avg_price'] = new_avg_price
            
            else:  # sell
                old_qty = self._positions[symbol]['quantity']
                old_price = self._positions[symbol]['avg_price']
                new_qty = old_qty - quantity
                
                if old_qty <= 0:  # Adding to short or opening short
                    new_avg_price = ((abs(old_qty) * old_price) + (quantity * price)) / abs(new_qty) if new_qty != 0 else Decimal('0')
                else:  # Reducing long
                    new_avg_price = old_price if new_qty > 0 else price
                
                self._positions[symbol]['quantity'] = new_qty
                self._positions[symbol]['avg_price'] = new_avg_price
            
            # Update cash balance
            if side == 'buy':
                self._cash_balance -= (notional + fee_calc.fee_amount)
            else:
                self._cash_balance += (notional - fee_calc.fee_amount)
            
            # Record actual fee
            await self.fees_engine.record_actual_fee(
                symbol=symbol,
                notional=notional,
                actual_fee=fee_calc.fee_amount,
                fee_type=fee_calc.fee_type
            )
            
            # Store trade record
            trade_record = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'side': side,
                'quantity': float(quantity),
                'price': float(price),
                'notional': float(notional),
                'fees': float(fee_calc.fee_amount),
                'impact_bps': float(impact_estimate.estimated_impact_bps),
                'realized_pnl': float(pnl_result.get('realized_pnl', 0)),
                'is_reduce': is_reduce
            }
            
            self._trade_history.append(trade_record)
            
            trade_result.update({
                'success': True,
                'reason': 'Trade executed successfully',
                'fees': fee_calc.fee_amount,
                'impact_bps': impact_estimate.estimated_impact_bps,
                'realized_pnl': pnl_result.get('realized_pnl', Decimal('0')),
                'new_position': float(self._positions[symbol]['quantity']),
                'new_avg_price': float(self._positions[symbol]['avg_price'])
            })
            
            print(f"  ✅ Trade executed successfully")
            print(f"     New position: {self._positions[symbol]['quantity']} @ ${self._positions[symbol]['avg_price']:.2f}")
            print(f"     Realized PnL: ${pnl_result.get('realized_pnl', 0):+.2f}")
            print(f"     Fees paid: ${fee_calc.fee_amount:.2f}")
            
        except Exception as e:
            trade_result['reason'] = f"Execution error: {str(e)}"
            print(f"  ❌ Trade failed: {e}")
        
        return trade_result
    
    async def update_market_prices(self, mark_prices: Dict[str, Decimal]):
        """Update market prices and recalculate portfolio."""
        print(f"\n📊 Updating market prices...")
        
        # Update position marks
        for symbol, mark_price in mark_prices.items():
            if symbol in self._positions:
                self._positions[symbol]['mark_price'] = mark_price
        
        # Update portfolio service
        self.portfolio_service.update_mark_prices(mark_prices)
        
        # Mock account data for portfolio service
        balances = [Balance(currency="USD", total=self._cash_balance, available=self._cash_balance)]
        positions = []
        
        for symbol, pos_data in self._positions.items():
            if pos_data['quantity'] != 0:
                positions.append(Position(
                    symbol=symbol,
                    quantity=pos_data['quantity'],
                    avg_price=pos_data['avg_price']
                ))
        
        self.portfolio_service.update_account_data(balances, positions)
        
        # Generate portfolio snapshot
        snapshot = self.portfolio_service.compute_current_valuation()
        
        print(f"  📈 Portfolio updated:")
        print(f"     Total Equity: ${snapshot.total_equity_usd:,.2f}")
        print(f"     Unrealized PnL: ${snapshot.unrealized_pnl:+,.2f}")
        print(f"     Realized PnL: ${snapshot.realized_pnl:+,.2f}")
        
        return snapshot
    
    async def generate_system_report(self) -> Dict:
        """Generate comprehensive system report."""
        print(f"\n📊 Generating system report...")
        
        # Get current portfolio state
        current_marks = {symbol: pos['mark_price'] for symbol, pos in self._positions.items()}
        portfolio_snapshot = await self.update_market_prices(current_marks)
        
        # Get fee summary
        fee_summary = await self.fees_engine.get_fee_summary(hours_back=24)
        
        # Get margin state
        position_data = {
            symbol: {
                'quantity': pos['quantity'],
                'mark_price': pos['mark_price']
            }
            for symbol, pos in self._positions.items()
            if pos['quantity'] != 0
        }
        
        margin_state = await self.margin_monitor.compute_margin_state(
            total_equity=portfolio_snapshot.total_equity_usd,
            cash_balance=self._cash_balance,
            positions=position_data
        )
        
        # Get daily metrics
        daily_metrics = self.portfolio_service.get_daily_metrics()
        
        report = {
            'system_summary': {
                'initial_capital': float(self.initial_capital),
                'current_equity': float(portfolio_snapshot.total_equity_usd),
                'total_return': float((portfolio_snapshot.total_equity_usd - self.initial_capital) / self.initial_capital),
                'cash_balance': float(self._cash_balance),
                'positions_count': len([p for p in self._positions.values() if p['quantity'] != 0]),
                'trades_executed': len(self._trade_history)
            },
            'portfolio': portfolio_snapshot.to_dict(),
            'fees': fee_summary,
            'margin': margin_state.to_dict() if margin_state else None,
            'daily_metrics': daily_metrics,
            'trade_history': self._trade_history,
            'current_positions': {
                symbol: {
                    'quantity': float(pos['quantity']),
                    'avg_price': float(pos['avg_price']),
                    'mark_price': float(pos['mark_price']),
                    'unrealized_pnl': float((pos['mark_price'] - pos['avg_price']) * pos['quantity']) if pos['quantity'] != 0 else 0
                }
                for symbol, pos in self._positions.items()
                if pos['quantity'] != 0
            }
        }
        
        return report


async def run_production_demo():
    """Run comprehensive production trading demo."""
    print("🚀 Starting Production Trading System Demo")
    print("=" * 60)
    
    # Initialize system
    system = ProductionTradingSystem(initial_capital=Decimal('100000'))
    await system.initialize()
    
    # Demo trading scenario
    scenarios = [
        # Open BTC long position
        {
            'symbol': 'BTC-USD',
            'side': 'buy',
            'quantity': Decimal('2'),
            'price': Decimal('50000'),
            'description': 'Open BTC long position'
        },
        # Add to BTC position
        {
            'symbol': 'BTC-USD', 
            'side': 'buy',
            'quantity': Decimal('0.5'),
            'price': Decimal('51000'),
            'description': 'Add to BTC position'
        },
        # Open ETH position
        {
            'symbol': 'ETH-USD',
            'side': 'buy', 
            'quantity': Decimal('10'),
            'price': Decimal('3000'),
            'description': 'Open ETH long position'
        },
        # Partial close BTC (take profit)
        {
            'symbol': 'BTC-USD',
            'side': 'sell',
            'quantity': Decimal('1'),
            'price': Decimal('52000'),
            'description': 'Partial close BTC (take profit)'
        }
    ]
    
    # Execute trading scenarios
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n🎯 SCENARIO {i}: {scenario['description']}")
        
        trade_result = await system.execute_trade(
            symbol=scenario['symbol'],
            side=scenario['side'],
            quantity=scenario['quantity'],
            price=scenario['price'],
            post_only=True  # Use maker orders for better fees
        )
        
        if trade_result['success']:
            print(f"✅ Trade successful")
        else:
            print(f"❌ Trade failed: {trade_result['reason']}")
        
        # Small delay for demo
        await asyncio.sleep(0.5)
    
    # Market movement simulation
    print(f"\n📈 MARKET MOVEMENT: Simulating price changes...")
    
    market_updates = [
        {'BTC-USD': Decimal('51500'), 'ETH-USD': Decimal('3100')},
        {'BTC-USD': Decimal('52500'), 'ETH-USD': Decimal('3200')},
        {'BTC-USD': Decimal('49000'), 'ETH-USD': Decimal('2800')}  # Market downturn
    ]
    
    for i, prices in enumerate(market_updates, 1):
        print(f"\n  📊 Market Update {i}: {', '.join(f'{k}=${v:,.0f}' for k, v in prices.items())}")
        await system.update_market_prices(prices)
        
        # Check margin after market movement
        margin_state = system.margin_monitor.get_current_state()
        if margin_state and margin_state.margin_utilization > Decimal('0.8'):
            print(f"  ⚠️  High margin utilization: {margin_state.margin_utilization:.1%}")
    
    # Generate final report
    print(f"\n📊 FINAL SYSTEM REPORT")
    print("=" * 60)
    
    final_report = await system.generate_system_report()
    
    # Print summary
    summary = final_report['system_summary']
    print(f"Initial Capital: ${summary['initial_capital']:,.2f}")
    print(f"Final Equity: ${summary['current_equity']:,.2f}")
    print(f"Total Return: {summary['total_return']:+.2%}")
    print(f"Cash Balance: ${summary['cash_balance']:,.2f}")
    print(f"Active Positions: {summary['positions_count']}")
    print(f"Trades Executed: {summary['trades_executed']}")
    
    # Print fee summary
    fees = final_report['fees']
    print(f"\nFees Summary (24h):")
    print(f"  Total Fees: ${fees['total_fees']:.2f}")
    print(f"  Maker Fees: ${fees['maker_fees']:.2f}")
    print(f"  Taker Fees: ${fees['taker_fees']:.2f}")
    print(f"  Current Tier: {fees['current_tier']}")
    
    # Print positions
    positions = final_report['current_positions']
    if positions:
        print(f"\nCurrent Positions:")
        for symbol, pos in positions.items():
            pnl_sign = "+" if pos['unrealized_pnl'] >= 0 else ""
            print(f"  {symbol}: {pos['quantity']:+.3f} @ ${pos['avg_price']:,.2f} (Mark: ${pos['mark_price']:,.2f}) PnL: {pnl_sign}${pos['unrealized_pnl']:.2f}")
    
    # Save report
    output_file = Path("production_demo_report.json")
    with open(output_file, 'w') as f:
        json.dump(final_report, f, indent=2, default=str)
    
    print(f"\n💾 Full report saved to {output_file}")
    
    # Assess success
    success_criteria = [
        summary['trades_executed'] > 0,
        summary['current_equity'] > 0,  # System didn't blow up
        fees['total_fees'] > 0,  # Fees were calculated and recorded
        len(positions) > 0  # Have active positions
    ]
    
    if all(success_criteria):
        print(f"\n✅ PRODUCTION DEMO SUCCESSFUL")
        print(f"All components integrated and working correctly!")
        return 0
    else:
        print(f"\n❌ PRODUCTION DEMO HAD ISSUES")
        return 1


async def main():
    """Main execution."""
    try:
        return await run_production_demo()
    except Exception as e:
        print(f"\n💥 Demo crashed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)