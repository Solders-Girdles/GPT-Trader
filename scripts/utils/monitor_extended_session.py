#!/usr/bin/env python3
"""
Monitor Extended Paper Trading Session
Real-time monitoring of the extended trading session.
"""

import time
import json
from datetime import datetime
from pathlib import Path
import subprocess
import sys


def get_latest_results():
    """Get the latest session results."""
    results_dir = Path(__file__).parent.parent / 'results'
    
    if not results_dir.exists():
        return None
    
    # Find most recent extended session file
    session_files = list(results_dir.glob('extended_session_*.json'))
    if not session_files:
        return None
    
    latest_file = max(session_files, key=lambda f: f.stat().st_mtime)
    
    try:
        with open(latest_file) as f:
            return json.load(f)
    except:
        return None


def monitor_session():
    """Monitor the extended session."""
    print("\n" + "="*70)
    print("📊 EXTENDED SESSION MONITOR")
    print("="*70)
    print("Monitoring paper trading session in real-time...")
    print("Press Ctrl+C to stop monitoring")
    print("="*70)
    
    start_time = datetime.now()
    last_update = None
    
    try:
        while True:
            # Check if background process is still running
            try:
                result = subprocess.run(['pgrep', '-f', 'extended_simulation_session.py'], 
                                      capture_output=True, text=True)
                if not result.stdout.strip():
                    print("\n🛑 Extended session process not found - may have completed")
                    break
            except:
                pass
            
            # Get latest results
            results = get_latest_results()
            
            current_time = datetime.now()
            elapsed = (current_time - start_time).seconds / 60
            
            print(f"\n[{current_time.strftime('%H:%M:%S')}] Session Status (Elapsed: {elapsed:.0f}m)")
            print("-" * 50)
            
            if results:
                session_start = datetime.fromisoformat(results['start_time'])
                session_elapsed = (current_time - session_start).seconds / 60
                
                print(f"Session Start: {session_start.strftime('%H:%M:%S')}")
                print(f"Session Elapsed: {session_elapsed:.0f} minutes")
                
                if 'strategies' in results and results['strategies']:
                    print(f"\nCompleted Strategies:")
                    for strategy_name, strategy_data in results['strategies'].items():
                        print(f"  • {strategy_name}: {strategy_data['total_return']:+.2f}% "
                              f"({strategy_data['total_trades']} trades, "
                              f"{strategy_data['trades_per_hour']:.1f}/hr)")
                else:
                    print("No completed strategies yet...")
            else:
                print("No session results file found yet...")
            
            # Check results directory for any files
            results_dir = Path(__file__).parent.parent / 'results'
            if results_dir.exists():
                recent_files = sorted(results_dir.glob('*.json'), 
                                    key=lambda f: f.stat().st_mtime, reverse=True)[:3]
                if recent_files:
                    print(f"\nRecent result files:")
                    for file in recent_files:
                        mtime = datetime.fromtimestamp(file.stat().st_mtime)
                        age_minutes = (current_time - mtime).seconds / 60
                        print(f"  • {file.name} ({age_minutes:.0f}m ago)")
            
            print("\n" + "="*50)
            
            # Wait before next check
            time.sleep(60)  # Check every minute
            
    except KeyboardInterrupt:
        print(f"\n\n👋 Monitoring stopped by user")
    
    # Final status
    results = get_latest_results()
    if results and 'strategies' in results:
        print(f"\n📊 FINAL SESSION SUMMARY:")
        print("-" * 40)
        
        total_trades = 0
        for strategy_name, strategy_data in results['strategies'].items():
            total_trades += strategy_data.get('total_trades', 0)
            print(f"{strategy_name}: {strategy_data.get('total_return', 0):+.2f}% "
                  f"({strategy_data.get('total_trades', 0)} trades)")
        
        print(f"\nTotal trades across all strategies: {total_trades}")


if __name__ == "__main__":
    monitor_session()