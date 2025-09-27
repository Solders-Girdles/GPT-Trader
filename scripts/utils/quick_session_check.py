#!/usr/bin/env python3
"""
Quick Session Status Check
Get immediate status of the extended trading session.
"""

import subprocess
import json
from datetime import datetime
from pathlib import Path

def check_session_status():
    """Quick check of session status."""
    print("🔍 QUICK SESSION STATUS CHECK")
    print("="*50)
    
    # Check if process is running
    try:
        result = subprocess.run(['pgrep', '-f', 'extended_simulation_session.py'], 
                              capture_output=True, text=True)
        if result.stdout.strip():
            print("✅ Extended session process is RUNNING")
            pids = result.stdout.strip().split('\n')
            print(f"   Process ID(s): {', '.join(pids)}")
        else:
            print("❌ Extended session process NOT found")
            return
    except:
        print("⚠️ Cannot check process status")
    
    # Check for results files
    results_dir = Path(__file__).parent.parent / 'results'
    if results_dir.exists():
        all_files = list(results_dir.glob('*.json'))
        recent_files = sorted(all_files, key=lambda f: f.stat().st_mtime, reverse=True)[:5]
        
        print(f"\n📊 Recent result files ({len(all_files)} total):")
        for i, file in enumerate(recent_files):
            mtime = datetime.fromtimestamp(file.stat().st_mtime)
            age = (datetime.now() - mtime).seconds / 60
            print(f"   {i+1}. {file.name} ({age:.0f}m ago)")
        
        # Check for extended session file specifically
        extended_files = list(results_dir.glob('extended_session_*.json'))
        if extended_files:
            latest_extended = max(extended_files, key=lambda f: f.stat().st_mtime)
            print(f"\n🎯 Latest extended session file: {latest_extended.name}")
            
            try:
                with open(latest_extended) as f:
                    data = json.load(f)
                
                print(f"   Start time: {data.get('start_time', 'Unknown')}")
                print(f"   Duration: {data.get('duration_hours', 'Unknown')} hours planned")
                
                if 'strategies' in data and data['strategies']:
                    print(f"   Completed strategies: {len(data['strategies'])}")
                    for strategy, result in data['strategies'].items():
                        print(f"     • {strategy}: {result.get('total_return', 0):+.2f}% "
                              f"({result.get('total_trades', 0)} trades)")
                else:
                    print("   No completed strategies yet")
                    
            except Exception as e:
                print(f"   Error reading file: {e}")
        else:
            print("\n📊 No extended session results file found yet")
    else:
        print("\n📊 No results directory found")
    
    # Show what to expect
    print(f"\n⏱️ EXPECTED TIMELINE:")
    print(f"   Session started: ~05:55 AM")
    print(f"   Current phase: Momentum strategy (2 hours)")
    print(f"   Next: Mean reversion strategy (2 hours)")  
    print(f"   Final: Breakout strategy (1.5 hours)")
    print(f"   Total: 5.5 hours = ~11:25 AM completion")
    
    current_time = datetime.now()
    print(f"\n🕐 Current time: {current_time.strftime('%H:%M:%S')}")

if __name__ == "__main__":
    check_session_status()