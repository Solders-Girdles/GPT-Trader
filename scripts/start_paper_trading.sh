#!/bin/bash

# Start Paper Trading Session with Monitoring
# This script launches all components for a complete paper trading session

echo "=================================================="
echo "       PAPER TRADING SESSION LAUNCHER"
echo "=================================================="
echo ""
echo "This will start:"
echo "1. Web Dashboard (http://localhost:8888)"
echo "2. Paper Trading Session"
echo "3. Performance Monitor"
echo ""
echo "Choose an option:"
echo "1) Quick Test (5 minutes per strategy)"
echo "2) Standard Session (30 minutes per strategy)"
echo "3) Extended Session (60 minutes per strategy)"
echo "4) Custom Single Strategy"
echo "5) Start Dashboard Only"
echo ""
read -p "Enter choice [1-5]: " choice

# Function to start dashboard
start_dashboard() {
    echo "Starting dashboard server..."
    python scripts/dashboard_server.py &
    DASHBOARD_PID=$!
    echo "Dashboard PID: $DASHBOARD_PID"
    sleep 2
    echo "Dashboard available at: http://localhost:8888"
    
    # Try to open in browser
    if command -v open &> /dev/null; then
        open http://localhost:8888
    elif command -v xdg-open &> /dev/null; then
        xdg-open http://localhost:8888
    fi
}

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Stopping services..."
    if [ ! -z "$DASHBOARD_PID" ]; then
        kill $DASHBOARD_PID 2>/dev/null
    fi
    if [ ! -z "$TRADING_PID" ]; then
        kill $TRADING_PID 2>/dev/null
    fi
    echo "Cleanup complete."
}

# Set trap for cleanup
trap cleanup EXIT

case $choice in
    1)
        echo "Starting Quick Test Session (5 min/strategy)..."
        start_dashboard
        sleep 3
        echo ""
        echo "Starting paper trading..."
        python scripts/paper_trade_live.py --strategy momentum --duration 5 --symbols BTC-USD,ETH-USD,SOL-USD &
        TRADING_PID=$!
        ;;
        
    2)
        echo "Starting Standard Session (30 min/strategy)..."
        start_dashboard
        sleep 3
        echo ""
        echo "Starting extensive paper trading..."
        python scripts/run_extensive_session.py
        ;;
        
    3)
        echo "Starting Extended Session (60 min/strategy)..."
        start_dashboard
        sleep 3
        echo ""
        read -p "This will run for 5 hours. Continue? (y/n): " confirm
        if [ "$confirm" = "y" ]; then
            # Modify the extensive session script to use 60 minutes
            python scripts/paper_trade_live.py --strategy momentum --duration 60 --symbols BTC-USD,ETH-USD,SOL-USD,LINK-USD &
            TRADING_PID=$!
        else
            echo "Cancelled."
            exit 0
        fi
        ;;
        
    4)
        echo "Custom Single Strategy Session"
        echo "Available strategies: momentum, mean_reversion, breakout, ma_crossover, volatility"
        read -p "Enter strategy: " strategy
        read -p "Enter duration (minutes): " duration
        read -p "Enter symbols (comma-separated): " symbols
        
        start_dashboard
        sleep 3
        echo ""
        echo "Starting custom session..."
        python scripts/paper_trade_live.py --strategy $strategy --duration $duration --symbols $symbols &
        TRADING_PID=$!
        ;;
        
    5)
        echo "Starting Dashboard Only..."
        start_dashboard
        echo ""
        echo "Dashboard is running. Start paper trading in another terminal:"
        echo "  python scripts/paper_trade_live.py --strategy momentum --duration 30"
        ;;
        
    *)
        echo "Invalid choice."
        exit 1
        ;;
esac

# Keep script running
if [ "$choice" != "2" ]; then
    echo ""
    echo "=================================================="
    echo "Services are running. Press Ctrl+C to stop all."
    echo "=================================================="
    echo ""
    echo "Monitor URLs:"
    echo "  Web Dashboard: http://localhost:8888"
    echo "  Terminal Monitor: python scripts/live_monitor.py (in new terminal)"
    echo ""
    echo "Check results:"
    echo "  python scripts/monitor_paper_trading.py"
    echo ""
    
    # Wait for user interrupt
    wait
fi