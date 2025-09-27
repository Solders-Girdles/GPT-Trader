#!/usr/bin/env python3
"""Quick test of the perps bot."""

import sys
import time
import subprocess
import threading

def run_bot():
    """Run the bot for a few seconds."""
    proc = subprocess.Popen([
        sys.executable,
        "scripts/run_perps_bot.py",
        "--profile", "dev",
        "--dry-run",
        "--interval", "1",
        "--symbols", "BTC-PERP"
    ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    # Let it run for 3 seconds
    time.sleep(3)
    
    # Terminate
    proc.terminate()
    
    # Get output
    output, _ = proc.communicate(timeout=1)
    print(output)
    
    return proc.returncode

if __name__ == "__main__":
    try:
        code = run_bot()
        print(f"\nBot test completed with code: {code}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)