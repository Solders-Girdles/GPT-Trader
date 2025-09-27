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
