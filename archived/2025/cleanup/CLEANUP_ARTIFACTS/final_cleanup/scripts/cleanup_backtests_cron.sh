#!/bin/bash
# Cron-compatible script for automated backtest cleanup
# SOT-PRE-009: Add file retention policy for backtests
#
# Add to crontab to run daily at 2 AM:
# 0 2 * * * /path/to/GPT-Trader/scripts/cleanup_backtests_cron.sh
#
# Or weekly on Sunday at 3 AM:
# 0 3 * * 0 /path/to/GPT-Trader/scripts/cleanup_backtests_cron.sh

set -euo pipefail  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CLEANUP_SCRIPT="$SCRIPT_DIR/cleanup_backtests.py"

# Create logs directory in project root, fallback to /tmp if needed
if [[ -w "$PROJECT_ROOT" ]]; then
    LOG_DIR="$PROJECT_ROOT/logs"
else
    LOG_DIR="/tmp/gpt-trader-logs"
fi
LOG_FILE="$LOG_DIR/backtest_cleanup.log"
PYTHON_CMD="python3"

# Ensure logs directory exists
if ! mkdir -p "$LOG_DIR" 2>/dev/null; then
    echo "Warning: Cannot create log directory $LOG_DIR, using /tmp"
    LOG_DIR="/tmp"
    LOG_FILE="$LOG_DIR/backtest_cleanup.log"
    mkdir -p "$LOG_DIR"
fi

# Function to log with timestamp
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> "$LOG_FILE"
}

# Function to send notification (customize as needed)
notify() {
    local status="$1"
    local message="$2"

    # Log the notification
    log "NOTIFICATION [$status]: $message"

    # Add your notification logic here:
    # - Email alerts
    # - Slack notifications
    # - System notifications
    # Example:
    # echo "$message" | mail -s "Backtest Cleanup $status" admin@example.com
}

# Main execution
main() {
    log "Starting automated backtest cleanup"

    # Check if cleanup script exists
    if [[ ! -f "$CLEANUP_SCRIPT" ]]; then
        notify "ERROR" "Cleanup script not found: $CLEANUP_SCRIPT"
        exit 1
    fi

    # Check if Python is available
    if ! command -v "$PYTHON_CMD" &> /dev/null; then
        notify "ERROR" "Python command not found: $PYTHON_CMD"
        exit 1
    fi

    # Change to project directory
    cd "$PROJECT_ROOT"

    # Run cleanup with default settings (7 days retention, keep top 10)
    if "$PYTHON_CMD" "$CLEANUP_SCRIPT" --retention-days 7 --keep-best 10 >> "$LOG_FILE" 2>&1; then
        # Get file counts for notification
        local file_count
        file_count=$(find "$PROJECT_ROOT/data/backtests" -name "*.csv" 2>/dev/null | wc -l || echo "0")

        notify "SUCCESS" "Backtest cleanup completed successfully. Remaining files: $file_count"
        log "Cleanup completed successfully"
    else
        local exit_code=$?

        # Exit code 2 means dry run (not an error)
        if [[ $exit_code -eq 2 ]]; then
            notify "INFO" "Dry run completed (exit code 2)"
        else
            notify "ERROR" "Cleanup failed with exit code: $exit_code"
            log "Cleanup failed with exit code: $exit_code"
            exit $exit_code
        fi
    fi

    log "Automated cleanup finished"
}

# Run with error handling
trap 'notify "ERROR" "Cleanup script failed unexpectedly"' ERR
main "$@"
