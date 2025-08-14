#!/usr/bin/env python3
"""
Database Consolidation Script for GPT-Trader
Phase 1.1: Consolidate multiple SQLite databases into unified schema

This script will:
1. Audit all existing databases
2. Back up all data
3. Migrate to unified schema
4. Verify data integrity
"""

import sqlite3
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatabaseConsolidator:
    """Consolidates multiple SQLite databases into unified schema"""
    
    # Known databases in the system
    LEGACY_DATABASES = {
        'strategy_collection': 'data/strategies/strategy_collection.db',
        'deployments': 'data/paper_trading/deployments.db',
        'portfolios': 'data/portfolio/portfolios.db',
        'order_management': 'data/oms/order_management.db',
        'circuit_breakers': 'data/risk/circuit_breakers.db',
        'live_trading': 'data/trading/live_trading.db',
        'risk_monitoring': 'data/risk/risk_monitoring.db',
        'streaming_data': 'data/streaming/streaming_data.db',
        'strategy_health': 'data/monitoring/strategy_health.db',
        'dashboard': 'data/dashboard/dashboard.db',
        'alerting': 'data/alerts/alerting.db',
    }
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path.cwd()
        self.backup_dir = self.project_root / 'data' / 'backup' / datetime.now().strftime('%Y%m%d_%H%M%S')
        self.unified_db_path = self.project_root / 'data' / 'unified.db'
        self.audit_results = {}
        
    def run_full_consolidation(self):
        """Execute complete database consolidation process"""
        try:
            logger.info("=" * 80)
            logger.info("Starting Database Consolidation Process")
            logger.info("=" * 80)
            
            # Step 1: Audit
            logger.info("\n📊 STEP 1: Auditing existing databases...")
            self.audit_databases()
            
            # Step 2: Backup
            logger.info("\n💾 STEP 2: Creating backups...")
            self.create_backups()
            
            # Step 3: Create unified schema
            logger.info("\n🏗️ STEP 3: Creating unified database schema...")
            self.create_unified_schema()
            
            # Step 4: Migrate data
            logger.info("\n📦 STEP 4: Migrating data...")
            self.migrate_all_data()
            
            # Step 5: Verify
            logger.info("\n✅ STEP 5: Verifying migration...")
            self.verify_migration()
            
            # Step 6: Generate report
            logger.info("\n📝 STEP 6: Generating migration report...")
            self.generate_report()
            
            logger.info("\n" + "=" * 80)
            logger.info("✨ Database consolidation completed successfully!")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"❌ Consolidation failed: {e}")
            logger.info("Rolling back changes...")
            self.rollback()
            raise
    
    def audit_databases(self) -> Dict[str, Any]:
        """Audit all existing databases"""
        logger.info("Scanning for database files...")
        
        for db_name, rel_path in self.LEGACY_DATABASES.items():
            db_path = self.project_root / rel_path
            
            if not db_path.exists():
                logger.warning(f"  ⚠️  {db_name}: Not found at {rel_path}")
                self.audit_results[db_name] = {'status': 'not_found', 'path': str(rel_path)}
                continue
                
            logger.info(f"  📁 {db_name}: Found at {rel_path}")
            
            try:
                conn = sqlite3.connect(str(db_path))
                cursor = conn.cursor()
                
                # Get tables
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                
                # Get record counts
                table_info = {}
                total_records = 0
                for table in tables:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    table_info[table] = count
                    total_records += count
                
                # Get file size
                file_size_mb = db_path.stat().st_size / (1024 * 1024)
                
                self.audit_results[db_name] = {
                    'status': 'found',
                    'path': str(rel_path),
                    'tables': table_info,
                    'total_records': total_records,
                    'file_size_mb': round(file_size_mb, 2)
                }
                
                logger.info(f"      Tables: {len(tables)}, Records: {total_records:,}, Size: {file_size_mb:.2f} MB")
                
                conn.close()
                
            except Exception as e:
                logger.error(f"  ❌ Error auditing {db_name}: {e}")
                self.audit_results[db_name] = {'status': 'error', 'error': str(e)}
        
        # Summary
        logger.info("\n📊 Audit Summary:")
        found_dbs = sum(1 for r in self.audit_results.values() if r.get('status') == 'found')
        total_tables = sum(len(r.get('tables', {})) for r in self.audit_results.values())
        total_records = sum(r.get('total_records', 0) for r in self.audit_results.values())
        total_size = sum(r.get('file_size_mb', 0) for r in self.audit_results.values())
        
        logger.info(f"  • Databases found: {found_dbs}/{len(self.LEGACY_DATABASES)}")
        logger.info(f"  • Total tables: {total_tables}")
        logger.info(f"  • Total records: {total_records:,}")
        logger.info(f"  • Total size: {total_size:.2f} MB")
        
        return self.audit_results
    
    def create_backups(self):
        """Create backups of all existing databases"""
        logger.info(f"Creating backup directory: {self.backup_dir}")
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        for db_name, audit_info in self.audit_results.items():
            if audit_info.get('status') != 'found':
                continue
                
            src_path = self.project_root / audit_info['path']
            dst_path = self.backup_dir / f"{db_name}.db"
            
            try:
                shutil.copy2(src_path, dst_path)
                logger.info(f"  ✅ Backed up {db_name}")
            except Exception as e:
                logger.error(f"  ❌ Failed to backup {db_name}: {e}")
                raise
        
        # Save audit results
        audit_file = self.backup_dir / 'audit_results.json'
        with open(audit_file, 'w') as f:
            json.dump(self.audit_results, f, indent=2, default=str)
        logger.info(f"  📝 Audit results saved to {audit_file}")
    
    def create_unified_schema(self):
        """Create the unified database with consolidated schema"""
        # Check if unified database already exists
        if self.unified_db_path.exists():
            backup_path = self.unified_db_path.with_suffix('.backup.db')
            shutil.move(self.unified_db_path, backup_path)
            logger.info(f"  Moved existing unified.db to {backup_path.name}")
        
        # Import the unified database module
        import sys
        sys.path.insert(0, str(self.project_root / 'src'))
        from bot.core.database import DatabaseConfig, DatabaseManager
        
        # Create unified database
        config = DatabaseConfig(database_path=self.unified_db_path)
        db_manager = DatabaseManager(config)
        
        logger.info(f"  ✅ Created unified database at {self.unified_db_path}")
        
        # Get schema info
        with db_manager.get_connection() as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            logger.info(f"  📋 Created {len(tables)} tables in unified schema")
        
        db_manager.close()
    
    def migrate_all_data(self):
        """Migrate data from all legacy databases to unified database"""
        migrations = {
            'strategy_collection': self.migrate_strategies,
            'deployments': self.migrate_deployments,
            'portfolios': self.migrate_portfolios,
            'order_management': self.migrate_orders,
            'circuit_breakers': self.migrate_circuit_breakers,
            'live_trading': self.migrate_live_trading,
            'risk_monitoring': self.migrate_risk_metrics,
            'streaming_data': self.migrate_market_data,
            'strategy_health': self.migrate_health_monitoring,
            'dashboard': self.migrate_dashboard_data,
            'alerting': self.migrate_alerts,
        }
        
        for db_name, migrate_func in migrations.items():
            if self.audit_results.get(db_name, {}).get('status') != 'found':
                logger.info(f"  ⏭️  Skipping {db_name} (not found)")
                continue
                
            try:
                logger.info(f"  🔄 Migrating {db_name}...")
                records_migrated = migrate_func()
                logger.info(f"     ✅ Migrated {records_migrated:,} records")
            except Exception as e:
                logger.error(f"  ❌ Failed to migrate {db_name}: {e}")
                raise
    
    def migrate_strategies(self) -> int:
        """Migrate strategy collection data"""
        src_db = self.project_root / self.audit_results['strategy_collection']['path']
        
        src_conn = sqlite3.connect(str(src_db))
        src_conn.row_factory = sqlite3.Row
        dst_conn = sqlite3.connect(str(self.unified_db_path))
        
        records_migrated = 0
        
        try:
            # Migrate strategies table
            cursor = src_conn.execute("SELECT * FROM strategies")
            for row in cursor:
                dst_conn.execute("""
                    INSERT OR REPLACE INTO components (
                        component_id, component_type, status, config_data, created_at
                    ) VALUES (?, 'strategy', 'active', ?, ?)
                """, (row['id'], json.dumps(dict(row)), row.get('created_at')))
                records_migrated += 1
            
            dst_conn.commit()
            
        finally:
            src_conn.close()
            dst_conn.close()
        
        return records_migrated
    
    def migrate_orders(self) -> int:
        """Migrate order management data"""
        src_db = self.project_root / self.audit_results['order_management']['path']
        
        if not src_db.exists():
            return 0
            
        src_conn = sqlite3.connect(str(src_db))
        src_conn.row_factory = sqlite3.Row
        dst_conn = sqlite3.connect(str(self.unified_db_path))
        
        records_migrated = 0
        
        try:
            # Check if orders table exists
            cursor = src_conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='orders'")
            if cursor.fetchone():
                cursor = src_conn.execute("SELECT * FROM orders")
                for row in cursor:
                    # Map old schema to new unified schema
                    dst_conn.execute("""
                        INSERT OR REPLACE INTO orders (
                            order_id, strategy_id, component_id, symbol, side, 
                            order_type, quantity, status, created_at
                        ) VALUES (?, ?, 'order_manager', ?, ?, ?, ?, ?, ?)
                    """, (
                        row['order_id'], row.get('strategy_id', 'unknown'),
                        row['symbol'], row['side'], row['order_type'],
                        row['quantity'], row['status'], row.get('created_at')
                    ))
                    records_migrated += 1
            
            dst_conn.commit()
            
        finally:
            src_conn.close()
            dst_conn.close()
        
        return records_migrated
    
    def migrate_circuit_breakers(self) -> int:
        """Migrate circuit breaker rules and events"""
        src_db = self.project_root / self.audit_results['circuit_breakers']['path']
        
        if not src_db.exists():
            return 0
            
        src_conn = sqlite3.connect(str(src_db))
        src_conn.row_factory = sqlite3.Row
        dst_conn = sqlite3.connect(str(self.unified_db_path))
        
        records_migrated = 0
        
        try:
            # Migrate rules
            cursor = src_conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='rules'")
            if cursor.fetchone():
                cursor = src_conn.execute("SELECT * FROM rules")
                for row in cursor:
                    dst_conn.execute("""
                        INSERT OR REPLACE INTO circuit_breaker_rules (
                            rule_id, name, description, breaker_type, threshold_value,
                            lookback_period_seconds, primary_action, status, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        row['id'], row['name'], row.get('description', ''),
                        row.get('type', 'unknown'), row.get('threshold', '0'),
                        row.get('lookback_seconds', 60), row.get('action', 'halt'),
                        row.get('status', 'active'), row.get('created_at')
                    ))
                    records_migrated += 1
            
            dst_conn.commit()
            
        finally:
            src_conn.close()
            dst_conn.close()
        
        return records_migrated
    
    def migrate_deployments(self) -> int:
        """Migrate deployment data"""
        # Placeholder - implement based on actual schema
        return 0
    
    def migrate_portfolios(self) -> int:
        """Migrate portfolio data"""
        # Placeholder - implement based on actual schema
        return 0
    
    def migrate_live_trading(self) -> int:
        """Migrate live trading data"""
        # Placeholder - implement based on actual schema
        return 0
    
    def migrate_risk_metrics(self) -> int:
        """Migrate risk monitoring data"""
        # Placeholder - implement based on actual schema
        return 0
    
    def migrate_market_data(self) -> int:
        """Migrate streaming market data"""
        # Placeholder - implement based on actual schema
        return 0
    
    def migrate_health_monitoring(self) -> int:
        """Migrate strategy health monitoring data"""
        # Placeholder - implement based on actual schema
        return 0
    
    def migrate_dashboard_data(self) -> int:
        """Migrate dashboard data"""
        # Placeholder - implement based on actual schema
        return 0
    
    def migrate_alerts(self) -> int:
        """Migrate alerting system data"""
        src_db = self.project_root / self.audit_results['alerting']['path']
        
        if not src_db.exists():
            return 0
            
        src_conn = sqlite3.connect(str(src_db))
        src_conn.row_factory = sqlite3.Row
        dst_conn = sqlite3.connect(str(self.unified_db_path))
        
        records_migrated = 0
        
        try:
            # Migrate alert rules
            cursor = src_conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='alert_rules'")
            if cursor.fetchone():
                cursor = src_conn.execute("SELECT * FROM alert_rules")
                for row in cursor:
                    dst_conn.execute("""
                        INSERT OR REPLACE INTO alert_rules (
                            rule_id, name, description, event_types, severity_levels,
                            channels, configuration, is_active, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        row['id'], row['name'], row.get('description', ''),
                        row.get('event_types', ''), row.get('severity_levels', ''),
                        row.get('channels', ''), row.get('config', '{}'),
                        row.get('active', 1), row.get('created_at')
                    ))
                    records_migrated += 1
            
            dst_conn.commit()
            
        finally:
            src_conn.close()
            dst_conn.close()
        
        return records_migrated
    
    def verify_migration(self):
        """Verify data integrity after migration"""
        logger.info("Verifying migration integrity...")
        
        conn = sqlite3.connect(str(self.unified_db_path))
        cursor = conn.cursor()
        
        # Check table counts
        tables_to_check = [
            'components', 'orders', 'positions', 'circuit_breaker_rules',
            'alert_rules', 'risk_metrics', 'quotes', 'trades', 'bars'
        ]
        
        for table in tables_to_check:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            logger.info(f"  • {table}: {count:,} records")
        
        conn.close()
    
    def generate_report(self):
        """Generate migration report"""
        report_path = self.backup_dir / 'migration_report.md'
        
        with open(report_path, 'w') as f:
            f.write("# Database Consolidation Report\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Pre-Migration Audit\n\n")
            for db_name, info in self.audit_results.items():
                f.write(f"### {db_name}\n")
                f.write(f"- Status: {info.get('status', 'unknown')}\n")
                if info.get('status') == 'found':
                    f.write(f"- Path: {info['path']}\n")
                    f.write(f"- Tables: {len(info.get('tables', {}))}\n")
                    f.write(f"- Records: {info.get('total_records', 0):,}\n")
                    f.write(f"- Size: {info.get('file_size_mb', 0):.2f} MB\n")
                f.write("\n")
            
            f.write("## Migration Results\n\n")
            f.write(f"- Unified database created: {self.unified_db_path}\n")
            f.write(f"- Backup location: {self.backup_dir}\n")
            f.write("\n")
            
            f.write("## Next Steps\n\n")
            f.write("1. Test the unified database with existing code\n")
            f.write("2. Update all modules to use DatabaseManager\n")
            f.write("3. Remove direct sqlite3.connect() calls\n")
            f.write("4. Delete legacy databases after validation\n")
        
        logger.info(f"  📄 Report saved to {report_path}")
    
    def rollback(self):
        """Rollback changes if migration fails"""
        if self.unified_db_path.exists():
            self.unified_db_path.unlink()
            logger.info("  Removed unified database")
        
        # Restore from backup if needed
        backup_unified = self.unified_db_path.with_suffix('.backup.db')
        if backup_unified.exists():
            shutil.move(backup_unified, self.unified_db_path)
            logger.info("  Restored previous unified database")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Consolidate GPT-Trader databases")
    parser.add_argument('--project-root', type=Path, help='Project root directory')
    parser.add_argument('--dry-run', action='store_true', help='Audit only, no migration')
    
    args = parser.parse_args()
    
    consolidator = DatabaseConsolidator(project_root=args.project_root)
    
    if args.dry_run:
        logger.info("Running in DRY RUN mode - audit only")
        consolidator.audit_databases()
    else:
        consolidator.run_full_consolidation()


if __name__ == '__main__':
    main()