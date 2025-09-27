#!/usr/bin/env python3
"""
Metrics Exporter CLI

Standalone Prometheus metrics exporter for monitoring.

Usage:
  poetry run python scripts/monitoring/metrics_exporter.py --port 9090
  
  Then access metrics at: http://localhost:9090/metrics
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.monitoring.dashboard_integration import MetricsExporter


async def run_exporter(port: int = 9090):
    """Run the metrics exporter server"""
    exporter = MetricsExporter(port=port)
    
    # Start the server
    await exporter.start()
    
    # Load initial metrics from latest canary monitor results if available
    results_dir = Path("results")
    if results_dir.exists():
        # Find most recent canary monitor file
        canary_files = sorted(results_dir.glob("canary_monitor_*.json"), reverse=True)
        if canary_files:
            latest_file = canary_files[0]
            print(f"📁 Loading metrics from: {latest_file}")
            
            with open(latest_file) as f:
                metrics = json.load(f)
                exporter.update_metrics_batch(metrics)
    
    print(f"\n✅ Metrics exporter running on http://0.0.0.0:{port}/metrics")
    print("Press Ctrl+C to stop")
    
    try:
        # Keep running until interrupted
        while True:
            await asyncio.sleep(60)  # Sleep for a minute
            
            # Periodically reload metrics from file
            if results_dir.exists():
                canary_files = sorted(results_dir.glob("canary_monitor_*.json"), reverse=True)
                if canary_files:
                    with open(canary_files[0]) as f:
                        metrics = json.load(f)
                        exporter.update_metrics_batch(metrics)
                        
    except KeyboardInterrupt:
        print("\n⚠️ Metrics exporter stopped")


def main():
    parser = argparse.ArgumentParser(description="Prometheus Metrics Exporter")
    parser.add_argument("--port", type=int, default=9090, help="Port to listen on (default: 9090)")
    
    args = parser.parse_args()
    
    try:
        asyncio.run(run_exporter(args.port))
    except KeyboardInterrupt:
        pass
    
    return 0


if __name__ == "__main__":
    sys.exit(main())