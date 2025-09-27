#!/usr/bin/env python3
"""
Reconcile our P&L calculations with Coinbase's reported values.

This script connects to Coinbase API and compares our internal P&L
calculations with what Coinbase reports, highlighting any discrepancies.

Usage:
    poetry run python scripts/reconcile_pnl_with_coinbase.py [--live]
"""

import argparse
import json
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from bot_v2.features.live_trade.pnl_tracker import PnLTracker
from bot_v2.features.brokerages.coinbase.adapter import CoinbaseBrokerage
from bot_v2.features.brokerages.coinbase.client import CoinbaseClient, CoinbaseAuth
import os
from dotenv import load_dotenv


class PnLReconciler:
    """Reconcile P&L between our calculations and Coinbase."""
    
    def __init__(self, adapter: CoinbaseBrokerage):
        """
        Initialize reconciler.
        
        Args:
            adapter: Coinbase adapter for API access
        """
        self.adapter = adapter
        self.tracker = PnLTracker()
        self.discrepancies = []
        
    def fetch_coinbase_positions(self) -> List[Dict]:
        """Fetch current positions from Coinbase."""
        try:
            # Get CFM positions which include P&L data
            positions = self.adapter.list_positions()
            
            # Also try to get detailed CFM position data
            cfm_data = []
            for pos in positions:
                try:
                    detailed = self.adapter.client.cfm_position(pos.symbol)
                    if detailed:
                        cfm_data.append(detailed)
                except Exception as e:
                    print(f"Warning: Could not fetch CFM data for {pos.symbol}: {e}")
            
            return cfm_data if cfm_data else [p.__dict__ for p in positions]
        except Exception as e:
            print(f"Error fetching positions: {e}")
            return []
    
    def fetch_recent_fills(self, symbol: Optional[str] = None) -> List[Dict]:
        """Fetch recent fills to rebuild position history."""
        try:
            fills = self.adapter.list_fills(symbol=symbol, limit=100)
            return fills
        except Exception as e:
            print(f"Error fetching fills: {e}")
            return []
    
    def rebuild_position_from_fills(self, fills: List[Dict]) -> None:
        """
        Rebuild position state from fill history.
        
        Args:
            fills: List of fill dictionaries from Coinbase
        """
        # Sort fills by time (oldest first)
        sorted_fills = sorted(fills, key=lambda x: x.get('created_at', ''))
        
        for fill in sorted_fills:
            symbol = fill.get('product_id', '')
            side = fill.get('side', '').lower()  # 'buy' or 'sell'
            size = Decimal(str(fill.get('size', 0)))
            price = Decimal(str(fill.get('price', 0)))
            
            if symbol and size > 0:
                # Determine if this is a reduce order
                # This is simplified - in reality you'd check order flags
                is_reduce = False
                if symbol in self.tracker.positions:
                    pos = self.tracker.positions[symbol]
                    if (pos.side == 'long' and side == 'sell') or \
                       (pos.side == 'short' and side == 'buy'):
                        is_reduce = True
                
                self.tracker.update_position(symbol, side, size, price, is_reduce)
    
    def compare_positions(self, coinbase_positions: List[Dict]) -> Dict:
        """
        Compare our calculated positions with Coinbase's reported positions.
        
        Args:
            coinbase_positions: List of position data from Coinbase
            
        Returns:
            Comparison report dictionary
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'positions_compared': 0,
            'discrepancies': [],
            'summary': {
                'total_our_unrealized': Decimal('0'),
                'total_cb_unrealized': Decimal('0'),
                'total_our_realized': Decimal('0'),
                'total_cb_realized': Decimal('0'),
                'total_difference': Decimal('0')
            }
        }
        
        for cb_pos in coinbase_positions:
            symbol = cb_pos.get('product_id') or cb_pos.get('symbol')
            if not symbol:
                continue
                
            report['positions_compared'] += 1
            
            # Get Coinbase's reported P&L
            cb_unrealized = Decimal(str(cb_pos.get('unrealized_pnl', 0)))
            cb_realized = Decimal(str(cb_pos.get('realized_pnl', 0)))
            cb_mark = Decimal(str(cb_pos.get('mark_price', 0)))
            
            # Update our tracker with current mark
            if cb_mark and symbol in self.tracker.positions:
                self.tracker.update_marks({symbol: cb_mark})
            
            # Get our calculated P&L
            if symbol in self.tracker.positions:
                our_pos = self.tracker.positions[symbol]
                our_unrealized = our_pos.unrealized_pnl
                our_realized = our_pos.realized_pnl
                
                # Check for discrepancies
                unrealized_diff = abs(our_unrealized - cb_unrealized)
                realized_diff = abs(our_realized - cb_realized)
                
                # Tolerance for differences (0.1% or $0.10, whichever is larger)
                tolerance = max(Decimal('0.10'), abs(cb_unrealized) * Decimal('0.001'))
                
                if unrealized_diff > tolerance:
                    report['discrepancies'].append({
                        'symbol': symbol,
                        'type': 'unrealized_pnl',
                        'our_value': float(our_unrealized),
                        'coinbase_value': float(cb_unrealized),
                        'difference': float(our_unrealized - cb_unrealized),
                        'percentage_diff': float((unrealized_diff / abs(cb_unrealized) * 100) if cb_unrealized else 0)
                    })
                
                if realized_diff > tolerance:
                    report['discrepancies'].append({
                        'symbol': symbol,
                        'type': 'realized_pnl',
                        'our_value': float(our_realized),
                        'coinbase_value': float(cb_realized),
                        'difference': float(our_realized - cb_realized),
                        'percentage_diff': float((realized_diff / abs(cb_realized) * 100) if cb_realized else 0)
                    })
                
                # Update summary
                report['summary']['total_our_unrealized'] += our_unrealized
                report['summary']['total_our_realized'] += our_realized
                report['summary']['total_cb_unrealized'] += cb_unrealized
                report['summary']['total_cb_realized'] += cb_realized
            else:
                # Position exists in Coinbase but not in our tracker
                report['discrepancies'].append({
                    'symbol': symbol,
                    'type': 'missing_position',
                    'coinbase_unrealized': float(cb_unrealized),
                    'coinbase_realized': float(cb_realized),
                    'note': 'Position exists in Coinbase but not in our tracker'
                })
        
        # Calculate total difference
        our_total = report['summary']['total_our_unrealized'] + report['summary']['total_our_realized']
        cb_total = report['summary']['total_cb_unrealized'] + report['summary']['total_cb_realized']
        report['summary']['total_difference'] = our_total - cb_total
        
        # Convert Decimals to float for JSON
        report['summary'] = {k: float(v) for k, v in report['summary'].items()}
        
        return report
    
    def reconcile(self, rebuild_from_fills: bool = True) -> Dict:
        """
        Perform full reconciliation.
        
        Args:
            rebuild_from_fills: Whether to rebuild positions from fill history
            
        Returns:
            Reconciliation report
        """
        print("Starting P&L reconciliation with Coinbase...")
        
        # Fetch current positions from Coinbase
        print("Fetching Coinbase positions...")
        cb_positions = self.fetch_coinbase_positions()
        
        if not cb_positions:
            print("No positions found in Coinbase")
            return {'error': 'No positions found'}
        
        print(f"Found {len(cb_positions)} positions in Coinbase")
        
        # Optionally rebuild from fills
        if rebuild_from_fills:
            print("Fetching recent fills to rebuild position history...")
            fills = self.fetch_recent_fills()
            if fills:
                print(f"Rebuilding positions from {len(fills)} fills...")
                self.rebuild_position_from_fills(fills)
            else:
                print("No fills found, using empty tracker")
        
        # Compare positions
        print("Comparing P&L calculations...")
        report = self.compare_positions(cb_positions)
        
        return report
    
    def print_report(self, report: Dict) -> None:
        """Print a formatted reconciliation report."""
        print("\n" + "="*60)
        print("P&L RECONCILIATION REPORT")
        print("="*60)
        print(f"Timestamp: {report['timestamp']}")
        print(f"Positions compared: {report['positions_compared']}")
        
        print("\n--- SUMMARY ---")
        summary = report['summary']
        print(f"Our Total Unrealized P&L:      ${summary['total_our_unrealized']:,.2f}")
        print(f"Coinbase Total Unrealized P&L: ${summary['total_cb_unrealized']:,.2f}")
        print(f"Our Total Realized P&L:        ${summary['total_our_realized']:,.2f}")
        print(f"Coinbase Total Realized P&L:   ${summary['total_cb_realized']:,.2f}")
        print(f"Total Difference:              ${summary['total_difference']:,.2f}")
        
        if report['discrepancies']:
            print("\n--- DISCREPANCIES ---")
            for disc in report['discrepancies']:
                print(f"\n{disc['symbol']} - {disc['type']}:")
                if disc['type'] == 'missing_position':
                    print(f"  {disc['note']}")
                    print(f"  Coinbase Unrealized: ${disc['coinbase_unrealized']:,.2f}")
                    print(f"  Coinbase Realized: ${disc['coinbase_realized']:,.2f}")
                else:
                    print(f"  Our Value:      ${disc['our_value']:,.2f}")
                    print(f"  Coinbase Value: ${disc['coinbase_value']:,.2f}")
                    print(f"  Difference:     ${disc['difference']:,.2f} ({disc['percentage_diff']:.2f}%)")
        else:
            print("\n✅ No significant discrepancies found!")
        
        print("\n" + "="*60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Reconcile P&L with Coinbase')
    parser.add_argument('--live', action='store_true', help='Use live API (default is mock)')
    parser.add_argument('--no-fills', action='store_true', help='Skip rebuilding from fills')
    parser.add_argument('--output', help='Save report to JSON file')
    args = parser.parse_args()
    
    try:
        # Load environment variables
        load_dotenv()
        
        # Setup adapter
        if args.live:
            print("Using LIVE Coinbase API")
            # Initialize real Coinbase client with credentials from env
            auth = CoinbaseAuth(
                api_key=os.getenv('COINBASE_PROD_CDP_API_KEY', ''),
                api_secret=os.getenv('COINBASE_PROD_CDP_PRIVATE_KEY', ''),
                api_mode='advanced'
            )
            client = CoinbaseClient(
                base_url='https://api.coinbase.com',
                auth=auth,
                api_mode='advanced'
            )
            adapter = CoinbaseBrokerage(client=client)
        else:
            print("Using MOCK mode for demonstration")
            # Create mock adapter with sample data
            from unittest.mock import Mock
            adapter = Mock()
            
            # Mock positions
            adapter.list_positions.return_value = [
                Mock(
                    symbol='BTC-PERP',
                    qty=Decimal('0.05'),
                    entry_price=Decimal('94000'),
                    mark_price=Decimal('96000'),
                    unrealized_pnl=Decimal('100'),
                    realized_pnl=Decimal('50'),
                    side='long'
                )
            ]
            
            # Mock CFM position data
            adapter.client.cfm_position.return_value = {
                'product_id': 'BTC-PERP',
                'size': '0.05',
                'entry_price': '94000',
                'mark_price': '96000',
                'unrealized_pnl': '100',
                'realized_pnl': '50'
            }
            
            # Mock fills
            adapter.list_fills.return_value = [
                {
                    'product_id': 'BTC-PERP',
                    'side': 'buy',
                    'size': '0.05',
                    'price': '94000',
                    'created_at': '2024-01-01T00:00:00Z'
                }
            ]
        
        # Create reconciler
        reconciler = PnLReconciler(adapter)
        
        # Perform reconciliation
        report = reconciler.reconcile(rebuild_from_fills=not args.no_fills)
        
        # Print report
        reconciler.print_report(report)
        
        # Save to file if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"\nReport saved to {args.output}")
        
        # Exit with error if discrepancies found
        if report.get('discrepancies'):
            sys.exit(1)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
