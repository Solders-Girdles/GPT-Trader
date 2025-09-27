#!/usr/bin/env python3
"""
Monitor Paper Trading Results
Analyzes and displays paper trading performance from saved results.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List
# import pandas as pd  # Optional, not used
# from tabulate import tabulate  # Optional for better formatting


def load_results(results_dir: Path) -> List[Dict]:
    """Load all paper trading results from directory."""
    results = []
    
    for file in results_dir.glob("*.json"):
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                data['filename'] = file.name
                results.append(data)
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    return results


def analyze_trades(trades: List[Dict]) -> Dict:
    """Analyze trade performance."""
    if not trades:
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0
        }
    
    winning = [t for t in trades if t.get('pnl', 0) > 0]
    losing = [t for t in trades if t.get('pnl', 0) < 0]
    
    total_wins = sum(t.get('pnl', 0) for t in winning)
    total_losses = abs(sum(t.get('pnl', 0) for t in losing))
    
    return {
        'total_trades': len(trades),
        'winning_trades': len(winning),
        'losing_trades': len(losing),
        'win_rate': len(winning) / len(trades) * 100 if trades else 0,
        'total_pnl': sum(t.get('pnl', 0) for t in trades),
        'avg_win': total_wins / len(winning) if winning else 0,
        'avg_loss': total_losses / len(losing) if losing else 0,
        'profit_factor': total_wins / total_losses if total_losses > 0 else 0
    }


def display_session_summary(result: Dict):
    """Display summary for a single session."""
    metrics = result.get('metrics', {})
    trades = result.get('trades', [])
    
    # Parse timestamps
    start = result.get('start_time', '')
    end = result.get('end_time', '')
    
    if start and end:
        try:
            start_dt = datetime.fromisoformat(start)
            end_dt = datetime.fromisoformat(end)
            duration = end_dt - start_dt
        except:
            duration = "Unknown"
    else:
        duration = metrics.get('duration', 'Unknown')
    
    print(f"\n{'=' * 70}")
    print(f"SESSION: {result.get('filename', 'Unknown')}")
    print(f"{'=' * 70}")
    print(f"Duration: {duration}")
    print(f"Final Equity: ${metrics.get('equity', 0):.2f}")
    print(f"Total Return: {metrics.get('total_return', 0):.2f}%")
    print(f"Max Drawdown: {metrics.get('drawdown', 0):.2f}%")
    
    # Trade analysis
    trade_stats = analyze_trades(trades)
    print(f"\nTrade Statistics:")
    print(f"  Total Trades: {trade_stats['total_trades']}")
    print(f"  Win Rate: {trade_stats['win_rate']:.1f}%")
    print(f"  Total P&L: ${trade_stats['total_pnl']:.2f}")
    print(f"  Avg Win: ${trade_stats['avg_win']:.2f}")
    print(f"  Avg Loss: ${trade_stats['avg_loss']:.2f}")
    print(f"  Profit Factor: {trade_stats['profit_factor']:.2f}")
    
    # Show trades
    if trades:
        print(f"\nRecent Trades:")
        for trade in trades[-5:]:  # Last 5 trades
            symbol = trade.get('symbol', 'Unknown')
            side = trade.get('side', 'Unknown')
            pnl = trade.get('pnl', 0)
            if side == 'sell' and pnl != 0:
                emoji = "🟢" if pnl > 0 else "🔴"
                print(f"  {emoji} {symbol} {side}: P&L ${pnl:.2f}")


def compare_strategies(results_dir: Path):
    """Compare strategy performance from comparison files."""
    comparison_files = list(results_dir.glob("strategy_comparison_*.json"))
    
    if not comparison_files:
        print("No strategy comparison files found.")
        return
    
    # Get most recent comparison
    latest = max(comparison_files, key=lambda f: f.stat().st_mtime)
    
    with open(latest, 'r') as f:
        data = json.load(f)
    
    print(f"\n{'=' * 70}")
    print(f"STRATEGY COMPARISON - {latest.name}")
    print(f"{'=' * 70}")
    
    # Create comparison table
    rows = []
    for strategy, metrics in data.items():
        rows.append([
            strategy.upper(),
            f"{metrics.get('return', 0):.2f}%",
            metrics.get('trades', 0),
            f"{metrics.get('win_rate', 0):.1f}%",
            f"${metrics.get('final_equity', 0):.2f}"
        ])
    
    # Sort by return
    rows.sort(key=lambda x: float(x[1].rstrip('%')), reverse=True)
    
    # Print table manually without tabulate
    headers = ['Strategy', 'Return', 'Trades', 'Win Rate', 'Final Equity']
    
    # Print header
    print("\n" + "-" * 70)
    print(f"{'Strategy':<15} {'Return':<10} {'Trades':<10} {'Win Rate':<12} {'Final Equity':<15}")
    print("-" * 70)
    
    # Print rows
    for row in rows:
        print(f"{row[0]:<15} {row[1]:<10} {row[2]:<10} {row[3]:<12} {row[4]:<15}")
    print("-" * 70)
    
    # Best and worst
    if rows:
        print(f"\n🏆 Best Performer: {rows[0][0]} ({rows[0][1]})")
        print(f"📉 Worst Performer: {rows[-1][0]} ({rows[-1][1]})")


def get_aggregate_stats(results: List[Dict]) -> Dict:
    """Calculate aggregate statistics across all sessions."""
    if not results:
        return {}
    
    all_trades = []
    total_initial = 0
    total_final = 0
    
    for result in results:
        trades = result.get('trades', [])
        all_trades.extend(trades)
        
        metrics = result.get('metrics', {})
        # Estimate initial capital (assuming 10000 default)
        initial = 10000
        final = metrics.get('equity', initial)
        total_initial += initial
        total_final += final
    
    trade_stats = analyze_trades(all_trades)
    
    return {
        'total_sessions': len(results),
        'total_trades': trade_stats['total_trades'],
        'overall_return': (total_final - total_initial) / total_initial * 100 if total_initial > 0 else 0,
        'win_rate': trade_stats['win_rate'],
        'profit_factor': trade_stats['profit_factor'],
        'total_pnl': trade_stats['total_pnl']
    }


def main():
    """Main monitoring function."""
    results_dir = Path(__file__).parent.parent / 'results'
    
    if not results_dir.exists():
        print("Results directory not found. Run paper trading first.")
        return
    
    print("=" * 70)
    print("PAPER TRADING MONITOR")
    print("=" * 70)
    
    # Load all results
    results = load_results(results_dir)
    
    if not results:
        print("No results found.")
        return
    
    print(f"\nFound {len(results)} trading sessions")
    
    # Show aggregate stats
    agg_stats = get_aggregate_stats(results)
    if agg_stats:
        print(f"\n📊 AGGREGATE STATISTICS:")
        print(f"  Total Sessions: {agg_stats['total_sessions']}")
        print(f"  Total Trades: {agg_stats['total_trades']}")
        print(f"  Overall Return: {agg_stats['overall_return']:.2f}%")
        print(f"  Win Rate: {agg_stats['win_rate']:.1f}%")
        print(f"  Profit Factor: {agg_stats['profit_factor']:.2f}")
        print(f"  Total P&L: ${agg_stats['total_pnl']:.2f}")
    
    # Show recent sessions
    recent = sorted(results, key=lambda x: x.get('end_time', ''), reverse=True)[:3]
    
    print(f"\n📈 RECENT SESSIONS:")
    for result in recent:
        display_session_summary(result)
    
    # Compare strategies if available
    compare_strategies(results_dir)
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if agg_stats.get('total_trades', 0) < 20:
        print("  ⚠️ Need more trades for statistical significance (20+ recommended)")
    
    if agg_stats.get('win_rate', 0) < 40:
        print("  ⚠️ Low win rate - consider adjusting strategy parameters")
    elif agg_stats.get('win_rate', 0) > 60:
        print("  ✅ Good win rate - strategy showing promise")
    
    if agg_stats.get('profit_factor', 0) < 1:
        print("  ⚠️ Profit factor < 1 - losing more than winning")
    elif agg_stats.get('profit_factor', 0) > 1.5:
        print("  ✅ Strong profit factor - good risk/reward ratio")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()