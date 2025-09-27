#!/usr/bin/env python3
"""
GPT-Trader Dashboard Launcher

Enhanced launcher script for GPT-Trader dashboards with multiple options.
"""

import argparse
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are available"""
    try:
        import streamlit
        import plotly
        import pandas
        print("✅ All dependencies found")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Install with: poetry install")
        return False

def launch_dashboard(
    dashboard_type: str = "standard",
    port: int = None,
    open_browser: bool = True, 
    headless: bool = False
):
    """Launch the specified GPT-Trader dashboard"""
    
    # Check dependencies
    if not check_dependencies():
        return False
    
    # Determine dashboard path and default port
    if dashboard_type == "realtime":
        dashboard_path = "src/bot/dashboard/realtime_dashboard.py"
        default_port = 8502
        dashboard_name = "Real-time Monitor"
    else:
        dashboard_path = "src/bot/dashboard/app.py"
        default_port = 8501
        dashboard_name = "Performance Dashboard"
    
    # Use provided port or default
    if port is None:
        port = default_port
    
    # Check if dashboard exists
    if not Path(dashboard_path).exists():
        print(f"❌ {dashboard_name} not found at {dashboard_path}")
        print("Make sure you're in the GPT-Trader root directory.")
        return False
    
    print(f"🚀 Launching GPT-Trader {dashboard_name}...")
    print(f"📊 Port: {port}")
    print(f"🌐 URL: http://localhost:{port}")
    
    if dashboard_type == "realtime":
        print("🔴 Features: Live monitoring, real-time P&L, emergency controls")
    else:
        print("📈 Features: Performance analysis, portfolio overview, strategy comparison")
    
    # Build streamlit command
    cmd = [
        "poetry", "run", "streamlit", "run", 
        dashboard_path,
        "--server.port", str(port),
        "--server.address", "localhost",
        "--browser.gatherUsageStats", "false"
    ]
    
    if headless:
        cmd.extend(["--server.headless", "true"])
    
    try:
        # Launch dashboard
        process = subprocess.Popen(cmd)
        
        if not headless:
            # Wait for server to start
            print("⏳ Waiting for dashboard to start...")
            time.sleep(4)
            
            if open_browser:
                print("🌐 Opening browser...")
                webbrowser.open(f"http://localhost:{port}")
        
        print(f"✅ {dashboard_name} launched successfully!")
        print("💡 Press Ctrl+C to stop the dashboard")
        
        # Wait for the process
        process.wait()
        
    except KeyboardInterrupt:
        print(f"\n🛑 Stopping {dashboard_name}...")
        process.terminate()
        print("✅ Dashboard stopped")
        
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")
        return False
    
    return True

def launch_both_dashboards():
    """Launch both dashboards simultaneously"""
    print("🚀 Launching both dashboards...")
    
    # Launch standard dashboard in background
    print("📊 Starting Performance Dashboard on port 8501...")
    std_cmd = [
        "poetry", "run", "streamlit", "run",
        "src/bot/dashboard/app.py",
        "--server.port", "8501",
        "--server.address", "localhost",
        "--server.headless", "true"
    ]
    
    try:
        std_process = subprocess.Popen(std_cmd)
        
        # Wait a moment then launch realtime
        time.sleep(3)
        print("🔴 Starting Real-time Monitor on port 8502...")
        
        rt_success = launch_dashboard("realtime", 8502, True, False)
        
        # Clean up background process
        std_process.terminate()
        
        return rt_success
        
    except Exception as e:
        print(f"❌ Error launching both dashboards: {e}")
        return False

def show_help():
    """Show available dashboard options"""
    print("""
🚀 GPT-Trader Dashboard Launcher
================================

Available Dashboards:
  standard    - Performance analysis and portfolio overview (port 8501)
  realtime    - Live monitoring with emergency controls (port 8502)
  both        - Launch both dashboards simultaneously

Usage Examples:
  python launch_dashboard.py                     # Standard dashboard
  python launch_dashboard.py --realtime          # Real-time monitor  
  python launch_dashboard.py --both              # Both dashboards
  python launch_dashboard.py --port 8503         # Custom port
  python launch_dashboard.py --headless          # No browser
""")

def main():
    """Main launcher function"""
    parser = argparse.ArgumentParser(
        description="Launch GPT-Trader Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Dashboard selection
    parser.add_argument("--realtime", action="store_true", 
                       help="Launch real-time monitoring dashboard")
    parser.add_argument("--standard", action="store_true",
                       help="Launch standard performance dashboard")
    parser.add_argument("--both", action="store_true",
                       help="Launch both dashboards")
    
    # Configuration
    parser.add_argument("--port", "-p", type=int, 
                       help="Port to run dashboard on")
    parser.add_argument("--no-browser", action="store_true", 
                       help="Don't open browser automatically")
    parser.add_argument("--headless", action="store_true", 
                       help="Run in headless mode")
    parser.add_argument("--help-dashboards", action="store_true",
                       help="Show dashboard information")
    
    args = parser.parse_args()
    
    # Show help if requested
    if args.help_dashboards:
        show_help()
        return 0
    
    # Determine dashboard type
    if args.both:
        success = launch_both_dashboards()
    elif args.realtime:
        success = launch_dashboard("realtime", args.port, not args.no_browser, args.headless)
    else:
        # Default to standard dashboard
        success = launch_dashboard("standard", args.port, not args.no_browser, args.headless)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())