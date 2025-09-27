#!/usr/bin/env python3
"""
SQLite to PostgreSQL Migration Script
Phase 2.5 - Day 1

Migrates data from multiple SQLite databases to unified PostgreSQL schema.
"""

import argparse
import json
import logging
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg2
from psycopg2.extras import execute_batch
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class MigrationConfig:
    """Migration configuration"""

    sqlite_dir: Path
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "gpt_trader"
    postgres_user: str = "trader"
    postgres_password: str = None  # Will be loaded from environment
    batch_size: int = 1000
    validate: bool = True
    dry_run: bool = False


class DatabaseMigrator:
    """Handles migration from SQLite to PostgreSQL"""

    def __init__(self, config: MigrationConfig) -> None:
        self.config = config
        self.sqlite_connections: dict[str, sqlite3.Connection] = {}
        self.pg_conn = None
        self.migration_stats = {
            "tables_migrated": 0,
            "records_migrated": 0,
            "errors": [],
            "warnings": [],
        }

    def connect_postgres(self) -> None:
        """Connect to PostgreSQL database"""
        try:
            self.pg_conn = psycopg2.connect(
                host=self.config.postgres_host,
                port=self.config.postgres_port,
                database=self.config.postgres_db,
                user=self.config.postgres_user,
                password=self.config.postgres_password,
            )
            self.pg_conn.autocommit = False
            logger.info("Connected to PostgreSQL")
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise

    def find_sqlite_databases(self) -> dict[str, Path]:
        """Find all SQLite databases in the project"""
        databases = {}

        # Expected database files
        expected_dbs = [
            "trading.db",
            "ml_features.db",
            "ml_models.db",
            "ml_predictions.db",
            "portfolio_optimization.db",
            "rebalancing_history.db",
        ]

        # Search for databases
        for db_name in expected_dbs:
            db_path = self.config.sqlite_dir / db_name
            if db_path.exists():
                databases[db_name] = db_path
                logger.info(f"Found database: {db_path}")
            else:
                # Try alternative locations
                alt_path = self.config.sqlite_dir / "data" / db_name
                if alt_path.exists():
                    databases[db_name] = alt_path
                    logger.info(f"Found database: {alt_path}")
                else:
                    logger.warning(f"Database not found: {db_name}")

        return databases

    def connect_sqlite(self, db_path: Path) -> sqlite3.Connection:
        """Connect to SQLite database"""
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def get_sqlite_schema(self, conn: sqlite3.Connection) -> dict[str, list[str]]:
        """Get schema information from SQLite database"""
        cursor = conn.cursor()

        # Get all tables
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        tables = cursor.fetchall()

        schema = {}
        for table in tables:
            table_name = table[0]
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            schema[table_name] = [col[1] for col in columns]

        return schema

    def migrate_trading_data(self) -> None:
        """Migrate trading-related data"""
        logger.info("Migrating trading data...")

        db_path = self.config.sqlite_dir / "trading.db"
        if not db_path.exists():
            logger.warning("Trading database not found, skipping")
            return

        conn = self.connect_sqlite(db_path)
        pg_cursor = self.pg_conn.cursor()

        try:
            # Migrate positions
            self._migrate_table(
                conn,
                pg_cursor,
                source_table="positions",
                target_table="trading.positions",
                column_mapping={
                    "id": None,  # Skip, use PostgreSQL serial
                    "symbol": "symbol",
                    "quantity": "quantity",
                    "entry_price": "entry_price",
                    "current_price": "current_price",
                    "opened_at": "opened_at",
                    "status": "status",
                    "strategy_id": "strategy_id",
                },
            )

            # Migrate orders
            self._migrate_table(
                conn,
                pg_cursor,
                source_table="orders",
                target_table="trading.orders",
                column_mapping={
                    "symbol": "symbol",
                    "side": "side",
                    "order_type": "order_type",
                    "quantity": "quantity",
                    "limit_price": "limit_price",
                    "status": "status",
                    "submitted_at": "submitted_at",
                },
            )

            # Migrate trades
            self._migrate_table(
                conn,
                pg_cursor,
                source_table="trades",
                target_table="trading.trades",
                column_mapping={
                    "symbol": "symbol",
                    "side": "side",
                    "quantity": "quantity",
                    "price": "price",
                    "commission": "commission",
                    "executed_at": "executed_at",
                },
            )

            self.pg_conn.commit()
            logger.info("Trading data migration completed")

        except Exception as e:
            self.pg_conn.rollback()
            logger.error(f"Error migrating trading data: {e}")
            self.migration_stats["errors"].append(str(e))
        finally:
            conn.close()

    def migrate_ml_data(self) -> None:
        """Migrate ML-related data"""
        logger.info("Migrating ML data...")

        pg_cursor = self.pg_conn.cursor()

        # Migrate ML features
        db_path = self.config.sqlite_dir / "ml_features.db"
        if db_path.exists():
            conn = self.connect_sqlite(db_path)
            try:
                # Migrate feature sets
                self._migrate_table(
                    conn,
                    pg_cursor,
                    source_table="feature_sets",
                    target_table="ml.feature_sets",
                    column_mapping={
                        "name": "name",
                        "description": "description",
                        "feature_count": "feature_count",
                        "feature_names": ("feature_names", self._convert_to_json),
                        "configuration": ("configuration", self._convert_to_json),
                    },
                )

                # Migrate feature values
                self._migrate_table(
                    conn,
                    pg_cursor,
                    source_table="feature_values",
                    target_table="ml.feature_values",
                    column_mapping={
                        "symbol": "symbol",
                        "timestamp": "timestamp",
                        "features": ("features", self._convert_to_json),
                    },
                )

                self.pg_conn.commit()
            except Exception as e:
                self.pg_conn.rollback()
                logger.error(f"Error migrating ML features: {e}")
                self.migration_stats["errors"].append(str(e))
            finally:
                conn.close()

        # Migrate ML models
        db_path = self.config.sqlite_dir / "ml_models.db"
        if db_path.exists():
            conn = self.connect_sqlite(db_path)
            try:
                self._migrate_table(
                    conn,
                    pg_cursor,
                    source_table="models",
                    target_table="ml.models",
                    column_mapping={
                        "model_type": "model_type",
                        "model_name": "model_name",
                        "model_path": "model_path",
                        "training_date": "training_date",
                        "performance_metrics": ("performance_metrics", self._convert_to_json),
                        "is_active": "is_active",
                    },
                )

                self.pg_conn.commit()
            except Exception as e:
                self.pg_conn.rollback()
                logger.error(f"Error migrating ML models: {e}")
                self.migration_stats["errors"].append(str(e))
            finally:
                conn.close()

    def migrate_portfolio_data(self) -> None:
        """Migrate portfolio-related data"""
        logger.info("Migrating portfolio data...")

        pg_cursor = self.pg_conn.cursor()

        # Migrate portfolio optimization
        db_path = self.config.sqlite_dir / "portfolio_optimization.db"
        if db_path.exists():
            conn = self.connect_sqlite(db_path)
            try:
                self._migrate_table(
                    conn,
                    pg_cursor,
                    source_table="optimization_runs",
                    target_table="portfolio.optimization_runs",
                    column_mapping={
                        "timestamp": "timestamp",
                        "optimization_type": "optimization_type",
                        "objective": "objective",
                        "optimal_weights": ("optimal_weights", self._convert_to_json),
                        "expected_return": "expected_return",
                        "expected_risk": "expected_risk",
                        "sharpe_ratio": "sharpe_ratio",
                    },
                )

                self.pg_conn.commit()
            except Exception as e:
                self.pg_conn.rollback()
                logger.error(f"Error migrating portfolio data: {e}")
                self.migration_stats["errors"].append(str(e))
            finally:
                conn.close()

    def _migrate_table(
        self,
        source_conn: sqlite3.Connection,
        target_cursor,
        source_table: str,
        target_table: str,
        column_mapping: dict,
    ):
        """Migrate a single table"""
        logger.info(f"Migrating {source_table} to {target_table}")

        # Get source data
        source_cursor = source_conn.cursor()
        source_cursor.execute(f"SELECT COUNT(*) FROM {source_table}")
        total_rows = source_cursor.fetchone()[0]

        if total_rows == 0:
            logger.info(f"No data in {source_table}, skipping")
            return

        source_cursor.execute(f"SELECT * FROM {source_table}")

        # Process in batches
        batch = []
        columns_to_insert = []

        with tqdm(total=total_rows, desc=f"Migrating {source_table}") as pbar:
            for row in source_cursor:
                # Convert row to dict
                row_dict = dict(row)

                # Transform data according to mapping
                transformed_row = {}
                for source_col, target_info in column_mapping.items():
                    if target_info is None:
                        continue

                    if isinstance(target_info, tuple):
                        target_col, transform_func = target_info
                        if source_col in row_dict:
                            transformed_row[target_col] = transform_func(row_dict[source_col])
                    else:
                        if source_col in row_dict:
                            transformed_row[target_info] = row_dict[source_col]

                if not columns_to_insert:
                    columns_to_insert = list(transformed_row.keys())

                batch.append(tuple(transformed_row[col] for col in columns_to_insert))

                if len(batch) >= self.config.batch_size:
                    self._insert_batch(target_cursor, target_table, columns_to_insert, batch)
                    batch = []

                pbar.update(1)

        # Insert remaining batch
        if batch:
            self._insert_batch(target_cursor, target_table, columns_to_insert, batch)

        self.migration_stats["tables_migrated"] += 1
        self.migration_stats["records_migrated"] += total_rows
        logger.info(f"Migrated {total_rows} rows from {source_table}")

    def _insert_batch(self, cursor, table: str, columns: list[str], batch: list[tuple]):
        """Insert a batch of records"""
        if self.config.dry_run:
            logger.info(f"[DRY RUN] Would insert {len(batch)} records into {table}")
            return

        placeholders = ",".join(["%s"] * len(columns))
        columns_str = ",".join(columns)
        query = f"INSERT INTO {table} ({columns_str}) VALUES ({placeholders})"

        try:
            execute_batch(cursor, query, batch, page_size=self.config.batch_size)
        except Exception as e:
            logger.error(f"Error inserting batch into {table}: {e}")
            raise

    def _convert_to_json(self, value: Any) -> str:
        """Convert value to JSON string"""
        if value is None:
            return None
        if isinstance(value, str):
            try:
                # Try to parse as JSON to validate
                json.loads(value)
                return value
            except:
                # If not valid JSON, convert to JSON
                return json.dumps(value)
        return json.dumps(value)

    def validate_migration(self):
        """Validate the migration by comparing row counts"""
        logger.info("Validating migration...")

        pg_cursor = self.pg_conn.cursor()

        # Check row counts for key tables
        validations = [
            ("trading.positions", "positions"),
            ("trading.orders", "orders"),
            ("trading.trades", "trades"),
            ("ml.feature_sets", "feature_sets"),
            ("ml.models", "models"),
        ]

        for pg_table, sqlite_table in validations:
            # Get PostgreSQL count
            pg_cursor.execute(f"SELECT COUNT(*) FROM {pg_table}")
            pg_count = pg_cursor.fetchone()[0]

            # Get SQLite count (if database exists)
            sqlite_count = 0
            for db_name, db_path in self.find_sqlite_databases().items():
                conn = self.connect_sqlite(db_path)
                cursor = conn.cursor()
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {sqlite_table}")
                    sqlite_count += cursor.fetchone()[0]
                except:
                    pass
                conn.close()

            if pg_count != sqlite_count:
                logger.warning(
                    f"Row count mismatch for {pg_table}: PostgreSQL={pg_count}, SQLite={sqlite_count}"
                )
                self.migration_stats["warnings"].append(f"Row count mismatch for {pg_table}")
            else:
                logger.info(f"✓ {pg_table}: {pg_count} rows")

    def generate_report(self):
        """Generate migration report"""
        report = f"""
        ========================================
        Migration Report
        ========================================

        Tables Migrated: {self.migration_stats['tables_migrated']}
        Records Migrated: {self.migration_stats['records_migrated']}

        Errors: {len(self.migration_stats['errors'])}
        Warnings: {len(self.migration_stats['warnings'])}

        """

        if self.migration_stats["errors"]:
            report += "Errors:\n"
            for error in self.migration_stats["errors"]:
                report += f"  - {error}\n"

        if self.migration_stats["warnings"]:
            report += "\nWarnings:\n"
            for warning in self.migration_stats["warnings"]:
                report += f"  - {warning}\n"

        logger.info(report)

        # Save report to file
        report_path = Path("migration_report.txt")
        report_path.write_text(report)
        logger.info(f"Report saved to {report_path}")

    def run(self):
        """Run the complete migration"""
        logger.info("Starting SQLite to PostgreSQL migration...")

        try:
            # Connect to PostgreSQL
            self.connect_postgres()

            # Find SQLite databases
            databases = self.find_sqlite_databases()
            if not databases:
                logger.error("No SQLite databases found")
                return

            # Run migrations
            self.migrate_trading_data()
            self.migrate_ml_data()
            self.migrate_portfolio_data()

            # Validate if requested
            if self.config.validate:
                self.validate_migration()

            # Generate report
            self.generate_report()

            logger.info("Migration completed successfully!")

        except Exception as e:
            logger.error(f"Migration failed: {e}")
            if self.pg_conn:
                self.pg_conn.rollback()
            raise
        finally:
            if self.pg_conn:
                self.pg_conn.close()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Migrate SQLite to PostgreSQL")
    parser.add_argument(
        "--sqlite-dir", type=Path, default=Path("."), help="Directory containing SQLite databases"
    )
    parser.add_argument("--postgres-host", default="localhost", help="PostgreSQL host")
    parser.add_argument("--postgres-port", type=int, default=5432, help="PostgreSQL port")
    parser.add_argument("--postgres-db", default="gpt_trader", help="PostgreSQL database name")
    parser.add_argument("--postgres-user", default="trader", help="PostgreSQL username")
    parser.add_argument(
        "--postgres-password",
        default=os.getenv("DATABASE_PASSWORD"),
        help="PostgreSQL password (or set DATABASE_PASSWORD env var)",
    )
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for inserts")
    parser.add_argument(
        "--validate", action="store_true", help="Validate migration after completion"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Perform dry run without actual migration"
    )

    args = parser.parse_args()

    # Validate password is provided
    postgres_password = args.postgres_password or os.getenv("DATABASE_PASSWORD")
    if not postgres_password:
        parser.error(
            "PostgreSQL password must be provided via --postgres-password or DATABASE_PASSWORD environment variable"
        )

    config = MigrationConfig(
        sqlite_dir=args.sqlite_dir,
        postgres_host=args.postgres_host,
        postgres_port=args.postgres_port,
        postgres_db=args.postgres_db,
        postgres_user=args.postgres_user,
        postgres_password=postgres_password,
        batch_size=args.batch_size,
        validate=args.validate,
        dry_run=args.dry_run,
    )

    migrator = DatabaseMigrator(config)
    migrator.run()


if __name__ == "__main__":
    main()
