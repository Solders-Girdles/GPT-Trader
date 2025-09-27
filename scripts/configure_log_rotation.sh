#!/bin/bash
# Configure log rotation for demo/trade logs
# Keeps recent windows in /tmp, persists summaries to docs/ops/preflight/

set -e

echo "📝 CONFIGURING LOG ROTATION"
echo "============================"

# Log directories
LOG_DIR="/tmp/trading_logs"
ARCHIVE_DIR="docs/ops/preflight/logs"
SUMMARY_DIR="docs/ops/preflight/summaries"

# Create directories
mkdir -p "$LOG_DIR" "$ARCHIVE_DIR" "$SUMMARY_DIR"

# Log rotation configuration
MAX_LOG_SIZE="10M"  # Max size per log file
MAX_LOG_AGE="1d"    # Max age for active logs
KEEP_LOGS="10"      # Number of rotated logs to keep

# Create logrotate configuration
LOGROTATE_CONF="/tmp/trading_logrotate.conf"
cat > "$LOGROTATE_CONF" << EOF
# Trading System Log Rotation Configuration
# Generated: $(date)

# Trading logs
$LOG_DIR/*.log {
    size $MAX_LOG_SIZE
    maxage 1
    rotate $KEEP_LOGS
    compress
    delaycompress
    notifempty
    create 644 $(whoami) $(whoami)
    sharedscripts
    postrotate
        # Generate summary of rotated log
        if [ -f \$1.1 ]; then
            python -c "
import json
from datetime import datetime

log_file = '\$1.1'
summary = {
    'log_file': log_file,
    'rotated_at': datetime.utcnow().isoformat(),
    'size_bytes': 0,
    'line_count': 0,
    'error_count': 0,
    'warning_count': 0
}

try:
    with open(log_file, 'r') as f:
        lines = f.readlines()
        summary['line_count'] = len(lines)
        summary['error_count'] = sum(1 for l in lines if 'ERROR' in l)
        summary['warning_count'] = sum(1 for l in lines if 'WARNING' in l)
except:
    pass

# Save summary
summary_file = '$SUMMARY_DIR/summary_' + datetime.utcnow().strftime('%Y%m%d_%H%M%S') + '.json'
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)
" 2>/dev/null || true
        fi
        
        # Archive compressed logs older than 1 day
        find $LOG_DIR -name "*.gz" -mtime +1 -exec mv {} $ARCHIVE_DIR/ \; 2>/dev/null || true
    endscript
}

# WebSocket logs (higher rotation frequency)
$LOG_DIR/websocket*.log {
    size 5M
    hourly
    rotate 24
    compress
    delaycompress
    notifempty
    missingok
}

# Order logs (critical - keep longer)
$LOG_DIR/orders*.log {
    size 20M
    daily
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 $(whoami) $(whoami)
}

# Performance logs
$LOG_DIR/performance*.log {
    size 10M
    daily
    rotate 7
    compress
    notifempty
}
EOF

echo "✅ Logrotate configuration created: $LOGROTATE_CONF"

# Create log management script
LOG_MANAGER="scripts/manage_logs.py"
cat > "$LOG_MANAGER" << 'PYTHON'
#!/usr/bin/env python3
"""
Log management and rotation for trading system.
"""

import os
import json
import gzip
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

class LogManager:
    def __init__(self):
        self.log_dir = Path("/tmp/trading_logs")
        self.archive_dir = Path("docs/ops/preflight/logs")
        self.summary_dir = Path("docs/ops/preflight/summaries")
        
        # Create directories
        self.log_dir.mkdir(exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.summary_dir.mkdir(parents=True, exist_ok=True)
    
    def rotate_logs(self, max_size_mb: int = 10, max_age_hours: int = 24):
        """Rotate logs based on size and age."""
        
        rotated = []
        max_size_bytes = max_size_mb * 1024 * 1024
        max_age = datetime.now() - timedelta(hours=max_age_hours)
        
        for log_file in self.log_dir.glob("*.log"):
            stat = log_file.stat()
            
            # Check size
            if stat.st_size > max_size_bytes:
                rotated_name = f"{log_file.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
                rotated_path = log_file.parent / rotated_name
                log_file.rename(rotated_path)
                
                # Compress
                self.compress_log(rotated_path)
                rotated.append(rotated_name)
                
                # Create new empty log
                log_file.touch()
            
            # Check age
            elif datetime.fromtimestamp(stat.st_mtime) < max_age:
                # Archive old log
                archive_path = self.archive_dir / log_file.name
                shutil.move(str(log_file), str(archive_path))
                self.compress_log(archive_path)
                rotated.append(log_file.name)
        
        return rotated
    
    def compress_log(self, log_path: Path):
        """Compress a log file."""
        compressed_path = log_path.with_suffix('.log.gz')
        
        with open(log_path, 'rb') as f_in:
            with gzip.open(compressed_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # Remove original
        log_path.unlink()
        return compressed_path
    
    def generate_summary(self, log_path: Path) -> Dict:
        """Generate summary of a log file."""
        
        summary = {
            'file': str(log_path),
            'timestamp': datetime.now().isoformat(),
            'size_bytes': 0,
            'lines': 0,
            'errors': 0,
            'warnings': 0,
            'orders': 0,
            'trades': 0
        }
        
        try:
            # Handle compressed files
            if log_path.suffix == '.gz':
                open_func = gzip.open
                mode = 'rt'
            else:
                open_func = open
                mode = 'r'
            
            with open_func(log_path, mode) as f:
                for line in f:
                    summary['lines'] += 1
                    
                    # Count message types
                    if 'ERROR' in line:
                        summary['errors'] += 1
                    if 'WARNING' in line:
                        summary['warnings'] += 1
                    if 'order' in line.lower():
                        summary['orders'] += 1
                    if 'trade' in line.lower() or 'fill' in line.lower():
                        summary['trades'] += 1
            
            summary['size_bytes'] = log_path.stat().st_size
            
        except Exception as e:
            summary['error'] = str(e)
        
        return summary
    
    def cleanup_old_logs(self, keep_days: int = 7):
        """Clean up old archived logs."""
        
        cutoff = datetime.now() - timedelta(days=keep_days)
        removed = []
        
        for archive in self.archive_dir.glob("*.gz"):
            if datetime.fromtimestamp(archive.stat().st_mtime) < cutoff:
                archive.unlink()
                removed.append(archive.name)
        
        return removed
    
    def get_active_logs(self) -> List[Dict]:
        """Get list of active log files."""
        
        logs = []
        for log_file in self.log_dir.glob("*.log"):
            stat = log_file.stat()
            logs.append({
                'name': log_file.name,
                'size_mb': stat.st_size / (1024 * 1024),
                'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'lines': sum(1 for _ in open(log_file)) if stat.st_size < 1024*1024 else 'large'
            })
        
        return sorted(logs, key=lambda x: x['name'])

def main():
    """Run log management tasks."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Log Management")
    parser.add_argument("--rotate", action="store_true", help="Rotate logs")
    parser.add_argument("--cleanup", action="store_true", help="Clean old logs")
    parser.add_argument("--summary", action="store_true", help="Generate summaries")
    parser.add_argument("--status", action="store_true", help="Show log status")
    
    args = parser.parse_args()
    
    manager = LogManager()
    
    if args.rotate:
        rotated = manager.rotate_logs()
        print(f"Rotated {len(rotated)} logs")
    
    if args.cleanup:
        removed = manager.cleanup_old_logs()
        print(f"Removed {len(removed)} old logs")
    
    if args.summary:
        for log_file in manager.log_dir.glob("*.log"):
            summary = manager.generate_summary(log_file)
            summary_file = manager.summary_dir / f"summary_{log_file.stem}.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"Generated summary for {log_file.name}")
    
    if args.status:
        logs = manager.get_active_logs()
        print("\nActive Logs:")
        for log in logs:
            print(f"  {log['name']}: {log['size_mb']:.2f} MB, {log['lines']} lines")

if __name__ == "__main__":
    main()
PYTHON

chmod +x "$LOG_MANAGER"
echo "✅ Log manager created: $LOG_MANAGER"

# Create systemd timer for rotation (if systemd available)
if command -v systemctl &> /dev/null; then
    echo ""
    echo "📅 Creating systemd timer for log rotation..."
    
    # Service file
    cat > /tmp/trading-logs.service << EOF
[Unit]
Description=Trading Log Rotation
After=network.target

[Service]
Type=oneshot
User=$(whoami)
ExecStart=/usr/sbin/logrotate -f $LOGROTATE_CONF
ExecStartPost=$LOG_MANAGER --cleanup --summary
EOF
    
    # Timer file
    cat > /tmp/trading-logs.timer << EOF
[Unit]
Description=Trading Log Rotation Timer
Requires=trading-logs.service

[Timer]
OnCalendar=hourly
Persistent=true

[Install]
WantedBy=timers.target
EOF
    
    echo "  Systemd timer created (requires sudo to install)"
    echo "  To install: sudo cp /tmp/trading-logs.* /etc/systemd/system/"
    echo "  Then: sudo systemctl enable --now trading-logs.timer"
else
    # Fallback to cron
    echo ""
    echo "📅 Creating cron job for log rotation..."
    
    CRON_CMD="0 * * * * /usr/sbin/logrotate -f $LOGROTATE_CONF && $LOG_MANAGER --cleanup --summary"
    
    # Add to crontab if not already present
    (crontab -l 2>/dev/null | grep -v "logrotate.*trading"; echo "$CRON_CMD") | crontab -
    
    echo "✅ Cron job added for hourly rotation"
fi

# Test rotation
echo ""
echo "🧪 Testing log rotation..."

# Create test log
TEST_LOG="$LOG_DIR/test_rotation.log"
echo "Test log entry $(date)" > "$TEST_LOG"

# Run rotation
/usr/sbin/logrotate -f "$LOGROTATE_CONF" 2>/dev/null || logrotate -f "$LOGROTATE_CONF"

if [ -f "$TEST_LOG" ]; then
    echo "✅ Log rotation test successful"
else
    echo "⚠️  Log rotation test failed - check configuration"
fi

echo ""
echo "============================"
echo "✅ LOG ROTATION CONFIGURED"
echo ""
echo "Configuration:"
echo "  Active logs: $LOG_DIR"
echo "  Archives: $ARCHIVE_DIR"
echo "  Summaries: $SUMMARY_DIR"
echo "  Config: $LOGROTATE_CONF"
echo ""
echo "Management commands:"
echo "  Rotate: python $LOG_MANAGER --rotate"
echo "  Cleanup: python $LOG_MANAGER --cleanup"
echo "  Status: python $LOG_MANAGER --status"
echo ""