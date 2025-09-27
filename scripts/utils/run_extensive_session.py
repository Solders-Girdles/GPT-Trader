#!/usr/bin/env python3
"""
Extensive Paper Trading Session Runner
Runs an extended session rotating through all strategies for comprehensive testing.
"""

import os
import sys
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path

def print_banner(text):
    """Print a formatted banner."""
    width = 80
    print("\n" + "=" * width)
    print(text.center(width))
    print("=" * width)

def run_strategy_session(strategy, duration_minutes, symbols):
    """Run a single strategy session."""
    cmd = [
        sys.executable,
        "scripts/paper_trade_live.py",
        "--strategy", strategy,
        "--duration", str(duration_minutes),
        "--symbols", symbols,
        "--mode", "balanced",
        "--capital", "10000"
    ]
    
    print(f"\n🚀 Starting {strategy.upper()} strategy for {duration_minutes} minutes...")
    print(f"   Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=duration_minutes*60+60)
        print(f"✅ {strategy.upper()} session completed")
        
        # Parse results from output if available
        output_lines = result.stdout.split('\n')
        for line in output_lines[-20:]:  # Check last 20 lines
            if 'Equity:' in line or 'Return:' in line or 'Trades:' in line:
                print(f"   {line.strip()}")
                
    except subprocess.TimeoutExpired:
        print(f"⚠️ {strategy.upper()} session timed out")
    except Exception as e:
        print(f"❌ Error running {strategy}: {e}")
    
    # Brief pause between strategies
    print(f"⏸️ 30 second break before next strategy...")
    time.sleep(30)

def main():
    """Run extensive paper trading session."""
    
    print_banner("EXTENSIVE PAPER TRADING SESSION")
    
    # Configuration
    strategies = ['momentum', 'mean_reversion', 'breakout', 'ma_crossover', 'volatility']
    symbols = "BTC-USD,ETH-USD,SOL-USD,LINK-USD,MATIC-USD,AVAX-USD"
    duration_per_strategy = 30  # minutes per strategy
    
    total_duration = len(strategies) * duration_per_strategy
    session_start = datetime.now()
    session_end = session_start + timedelta(minutes=total_duration)
    
    print(f"""
📊 SESSION CONFIGURATION
------------------------
Strategies: {', '.join(strategies)}
Duration per strategy: {duration_per_strategy} minutes
Total duration: {total_duration} minutes ({total_duration/60:.1f} hours)
Symbols: {symbols.replace(',', ', ')}
Initial capital: $10,000 per strategy

Start time: {session_start.strftime('%Y-%m-%d %H:%M:%S')}
Expected end: {session_end.strftime('%Y-%m-%d %H:%M:%S')}

📈 MONITORING OPTIONS:
------------------------
1. Terminal Monitor:
   python scripts/live_monitor.py

2. Web Dashboard:
   python scripts/dashboard_server.py
   Then open: http://localhost:8888

3. Check Results:
   python scripts/monitor_paper_trading.py
""")
    
    # Confirm start
    print("\n⚠️ This will run for approximately {:.1f} hours".format(total_duration/60))
    response = input("Do you want to start the extensive session? (yes/no): ")
    
    if response.lower() not in ['yes', 'y']:
        print("Session cancelled.")
        return
    
    print_banner("STARTING EXTENSIVE SESSION")
    
    # Create results directory
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    # Run each strategy
    session_results = {}
    
    for i, strategy in enumerate(strategies, 1):
        print_banner(f"STRATEGY {i}/{len(strategies)}: {strategy.upper()}")
        print(f"Progress: {(i-1)/len(strategies)*100:.0f}% complete")
        
        # Run the strategy
        run_strategy_session(strategy, duration_per_strategy, symbols)
        
        # Track completion
        session_results[strategy] = {
            'completed': True,
            'timestamp': datetime.now().isoformat()
        }
    
    # Session complete
    session_duration = datetime.now() - session_start
    
    print_banner("SESSION COMPLETE")
    
    print(f"""
📊 FINAL SUMMARY
----------------
Total Duration: {str(session_duration).split('.')[0]}
Strategies Tested: {len(strategies)}
Sessions Completed: {sum(1 for r in session_results.values() if r['completed'])}

📁 Results Location:
   results/live_*.json

📈 View Results:
   python scripts/monitor_paper_trading.py

🔍 Detailed Analysis:
   Check individual session files in results/ directory
""")
    
    # Save session summary
    import json
    summary_file = results_dir / f"extensive_session_{session_start.strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(summary_file, 'w') as f:
        json.dump({
            'start_time': session_start.isoformat(),
            'end_time': datetime.now().isoformat(),
            'duration_minutes': total_duration,
            'strategies': strategies,
            'symbols': symbols.split(','),
            'results': session_results
        }, f, indent=2)
    
    print(f"\n💾 Session summary saved to: {summary_file.name}")

if __name__ == "__main__":
    main()
