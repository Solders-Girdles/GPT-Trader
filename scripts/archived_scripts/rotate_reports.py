#!/usr/bin/env python3
"""
Report rotation and retention policy.
Keeps last N snapshots and archives older ones.
"""

import os
import shutil
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict

# Configuration
REPORT_DIR = "docs/ops/preflight"
ARCHIVE_DIR = "docs/ops/preflight/archive"
MAX_REPORTS_PER_TYPE = 10  # Keep last 10 of each type
MAX_AGE_DAYS = 30  # Archive reports older than 30 days


def get_report_info(filepath: Path) -> Dict:
    """Extract metadata from report file."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            return {
                'path': filepath,
                'timestamp': data.get('timestamp', ''),
                'version': data.get('version', 'unknown'),
                'environment': data.get('environment', 'unknown'),
                'size': filepath.stat().st_size
            }
    except:
        # Fallback to file stats
        stat = filepath.stat()
        return {
            'path': filepath,
            'timestamp': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'version': 'unknown',
            'environment': 'unknown',
            'size': stat.st_size
        }


def rotate_reports():
    """Rotate reports based on retention policy."""
    report_path = Path(REPORT_DIR)
    archive_path = Path(ARCHIVE_DIR)
    
    if not report_path.exists():
        print(f"Report directory {REPORT_DIR} does not exist")
        return
    
    # Create archive directory
    archive_path.mkdir(parents=True, exist_ok=True)
    
    # Group reports by type
    report_types = {
        'capability': [],
        'ws_probe': [],
        'preflight': [],
        'demo_validation': []
    }
    
    # Scan for reports
    for file in report_path.glob("*.json"):
        if file.is_file():
            info = get_report_info(file)
            
            # Categorize by filename pattern
            if 'capability' in file.name:
                report_types['capability'].append(info)
            elif 'ws_probe' in file.name:
                report_types['ws_probe'].append(info)
            elif 'preflight' in file.name:
                report_types['preflight'].append(info)
            elif 'demo_validation' in file.name:
                report_types['demo_validation'].append(info)
    
    # Process each type
    total_archived = 0
    total_deleted = 0
    
    for report_type, reports in report_types.items():
        if not reports:
            continue
        
        # Sort by timestamp (newest first)
        reports.sort(key=lambda x: x['timestamp'], reverse=True)
        
        print(f"\n📁 Processing {report_type} reports ({len(reports)} found)")
        
        # Keep recent, archive old
        for i, report in enumerate(reports):
            file_path = report['path']
            
            # Check age
            try:
                timestamp = datetime.fromisoformat(report['timestamp'].replace('Z', '+00:00'))
                age_days = (datetime.now(timestamp.tzinfo) - timestamp).days
            except:
                # Use file modification time
                age_days = (datetime.now() - datetime.fromtimestamp(file_path.stat().st_mtime)).days
            
            if i < MAX_REPORTS_PER_TYPE and age_days < MAX_AGE_DAYS:
                # Keep the file
                print(f"  ✅ Keep: {file_path.name} (position {i+1}, age {age_days}d)")
            elif age_days >= MAX_AGE_DAYS:
                # Archive old file
                archive_name = f"{file_path.stem}_{datetime.now().strftime('%Y%m%d')}{file_path.suffix}"
                archive_file = archive_path / archive_name
                shutil.move(str(file_path), str(archive_file))
                print(f"  📦 Archive: {file_path.name} -> archive/{archive_name}")
                total_archived += 1
            else:
                # Delete excess recent files
                file_path.unlink()
                print(f"  🗑️  Delete: {file_path.name} (excess, position {i+1})")
                total_deleted += 1
    
    # Clean up old archives (optional)
    archive_cutoff = datetime.now() - timedelta(days=90)  # Delete archives > 90 days
    for archive_file in archive_path.glob("*.json"):
        if archive_file.stat().st_mtime < archive_cutoff.timestamp():
            archive_file.unlink()
            print(f"  🗑️  Purged old archive: {archive_file.name}")
    
    print(f"\n📊 Summary:")
    print(f"  Archived: {total_archived} files")
    print(f"  Deleted: {total_deleted} files")
    print(f"  Archive location: {ARCHIVE_DIR}")


def generate_index():
    """Generate an index of current reports."""
    report_path = Path(REPORT_DIR)
    
    index = {
        'generated': datetime.now().isoformat(),
        'reports': []
    }
    
    for file in sorted(report_path.glob("*.json")):
        if file.is_file() and file.name != 'index.json':
            info = get_report_info(file)
            index['reports'].append({
                'filename': file.name,
                'timestamp': info['timestamp'],
                'version': info['version'],
                'environment': info['environment'],
                'size_bytes': info['size']
            })
    
    # Save index
    index_file = report_path / 'index.json'
    with open(index_file, 'w') as f:
        json.dump(index, f, indent=2)
    
    print(f"📄 Generated index with {len(index['reports'])} reports")


def main():
    """Run report rotation and indexing."""
    print("🔄 REPORT ROTATION")
    print("="*60)
    print(f"Policy: Keep {MAX_REPORTS_PER_TYPE} reports per type")
    print(f"Archive reports older than {MAX_AGE_DAYS} days")
    print(f"Purge archives older than 90 days")
    print("="*60)
    
    rotate_reports()
    generate_index()
    
    print("\n✅ Report rotation complete")


if __name__ == "__main__":
    main()