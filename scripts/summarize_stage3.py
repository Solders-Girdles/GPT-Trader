#!/usr/bin/env python3
"""
Generates a one-page summary report from Stage 3 artifacts.
"""

import json
from pathlib import Path
import argparse
from datetime import datetime
from decimal import Decimal

def read_json_artifact(path: Path, default: any = None):
    """Safely read a JSON artifact file."""
    if path.exists():
        with open(path, 'r') as f:
            return json.load(f)
    return default

def generate_report(artifacts_dir: Path):
    """Generate and print the Stage 3 summary report."""
    
    # --- Load Artifacts ---
    summary = read_json_artifact(artifacts_dir / 'stage3_summary.json', {})
    preflight = read_json_artifact(artifacts_dir / 'preflight_results.json', {})
    rejections = read_json_artifact(artifacts_dir / 'rejection_breakdown.json', {})
    sized_down = read_json_artifact(artifacts_dir / 'sized_down_events.json', [])
    reconciliation = read_json_artifact(artifacts_dir / 'reconciliation_history.json', [])
    stop_tests = read_json_artifact(artifacts_dir / 'stop_limit_tests.json', [])

    # --- Header ---
    print("# Stage 3 Execution Summary")
    print(f"Report Generated: {datetime.now().isoformat()}")
    print(f"Run Duration: {summary.get('duration_hours', 0):.2f} hours")
    print("-" * 60)

    # --- Success Criteria ---
    print("## 📊 Success Criteria")
    
    # Performance
    ws_reconnects = summary.get('statistics', {}).get('ws_reconnects', 0)
    ws_ok = ws_reconnects <= 1
    print(f"- **WebSocket Stability:** {'✅ PASS' if ws_ok else '❌ FAIL'} ({ws_reconnects} reconnects)")

    # Execution
    stop_placed = any(t.get('status') == 'simulated' for t in stop_tests)
    sized_down_logged = len(sized_down) > 0
    print(f"- **Stop-Limit Flow:** {'✅ PASS' if stop_placed else '❌ FAIL'}")
    print(f"- **SIZED_DOWN Events:** {'✅ PASS' if sized_down_logged else '⚠️  WARN'} ({len(sized_down)} events logged)")

    # Acceptance Rate
    total_orders = summary.get('statistics', {}).get('total_orders', 0)
    total_rejections = rejections.get('total', 0)
    if total_orders > 0:
        acceptance_rate = (total_orders - total_rejections) / total_orders * 100
        acceptance_ok = acceptance_rate >= 90
        print(f"- **Acceptance Rate:** {'✅ PASS' if acceptance_ok else '❌ FAIL'} ({acceptance_rate:.2f}%)")
    else:
        print("- **Acceptance Rate:**  N/A (No orders placed)")

    # Financials
    final_recon = reconciliation[-1] if reconciliation else {}
    recon_ok = final_recon.get('verification', {}).get('reconciled', False)
    print(f"- **Financial Reconciliation:** {'✅ PASS' if recon_ok else '❌ FAIL'}")

    print("-" * 60)

    # --- Key Metrics ---
    print("## 📈 Key Metrics & Artifacts")

    # Rejection Breakdown
    if rejections:
        print("\n### Order Rejection Breakdown")
        print(f"- **Total Rejections:** {rejections.get('total', 0)}")
        if rejections.get('by_reason'):
            for reason, count in rejections['by_reason'].items():
                print(f"  - {reason}: {count}")

    # SIZED_DOWN Events
    if sized_down:
        print("\n### SIZED_DOWN Events")
        for event in sized_down[:5]: # Show first 5
            print(f"- {event['timestamp']}: {event['message']}")

    # Final Reconciliation
    if final_recon:
        print("\n### Final Financial State")
        print(f"- **Total Equity:** ${final_recon.get('total_equity', 0):,.2f}")
        print(f"- **Realized PnL:** ${final_recon.get('totals', {}).get('realized_pnl', 0):,.2f}")
        print(f"- **Unrealized PnL:** ${final_recon.get('totals', {}).get('unrealized_pnl', 0):,.2f}")
        print(f"- **Fees Paid:** ${final_recon.get('totals', {}).get('fees_paid', 0):,.2f}")

    print("-" * 60)

def main():
    parser = argparse.ArgumentParser(description="Generate Stage 3 summary report.")
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts/stage3"),
        help="Directory containing Stage 3 artifacts."
    )
    args = parser.parse_args()

    if not args.artifacts_dir.exists():
        print(f"Error: Artifacts directory not found at '{args.artifacts_dir}'")
        return

    generate_report(args.artifacts_dir)

if __name__ == "__main__":
    main()
