#!/usr/bin/env python3
"""
Launch the GPT-Trader Streamlit dashboard
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Launch Streamlit dashboard"""
    # Get the dashboard app path
    project_root = Path(__file__).parent.parent
    dashboard_path = project_root / "src" / "bot" / "dashboard" / "app.py"
    
    if not dashboard_path.exists():
        print(f"Error: Dashboard not found at {dashboard_path}")
        sys.exit(1)
    
    # Launch Streamlit
    print("🚀 Launching GPT-Trader Dashboard...")
    print("=" * 50)
    print("Dashboard will open in your browser automatically")
    print("Press Ctrl+C to stop the dashboard")
    print("=" * 50)
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            str(dashboard_path),
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n✅ Dashboard stopped")
    except Exception as e:
        print(f"Error launching dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()