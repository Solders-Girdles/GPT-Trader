#!/usr/bin/env python3
"""
Test PostgreSQL Connection and Setup
Phase 2.5 - Day 2

Verifies database connectivity and creates initial schema.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
from datetime import datetime
from decimal import Decimal

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from sqlalchemy import text

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_direct_connection():
    """Test direct PostgreSQL connection"""
    logger.info("Testing direct PostgreSQL connection...")

    try:
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database="postgres",  # Connect to default database first
            user="trader",
            password=os.getenv("DATABASE_PASSWORD", "trader_password_dev"),
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()

        # Check if database exists
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = 'gpt_trader'")
        exists = cursor.fetchone()

        if not exists:
            logger.info("Creating gpt_trader database...")
            cursor.execute("CREATE DATABASE gpt_trader")
            logger.info("Database created successfully")
        else:
            logger.info("Database gpt_trader already exists")

        cursor.close()
        conn.close()

        # Now connect to gpt_trader database
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database="gpt_trader",
            user="trader",
            password=os.getenv("DATABASE_PASSWORD", "trader_password_dev"),
        )

        cursor = conn.cursor()
        cursor.execute("SELECT version()")
        version = cursor.fetchone()[0]
        logger.info(f"✓ Connected to PostgreSQL: {version}")

        cursor.close()
        conn.close()

        return True

    except Exception as e:
        logger.error(f"✗ Direct connection failed: {e}")
        return False


def test_sqlalchemy_connection():
    """Test SQLAlchemy connection with our models"""
    logger.info("\nTesting SQLAlchemy connection...")

    try:
        from src.bot.database.manager import DatabaseConfig, DatabaseManager
        from src.bot.database.models import (
            Position,
        )

        # Create database manager
        config = DatabaseConfig()
        db_manager = DatabaseManager(config)

        # Test health check
        if db_manager.health_check():
            logger.info("✓ Database health check passed")
        else:
            logger.error("✗ Database health check failed")
            return False

        # Get pool status
        pool_status = db_manager.get_pool_status()
        logger.info(f"✓ Connection pool status: {pool_status}")

        # Test basic operations
        with db_manager.session_scope() as session:
            # Count tables
            result = session.execute(
                text(
                    """
                SELECT COUNT(*)
                FROM information_schema.tables
                WHERE table_schema IN ('trading', 'ml', 'portfolio', 'monitoring')
            """
                )
            )
            table_count = result.scalar()
            logger.info(f"✓ Found {table_count} tables in database")

            # Check schemas
            result = session.execute(
                text(
                    """
                SELECT schema_name
                FROM information_schema.schemata
                WHERE schema_name IN ('trading', 'ml', 'portfolio', 'monitoring')
                ORDER BY schema_name
            """
                )
            )
            schemas = [row[0] for row in result]
            logger.info(f"✓ Schemas present: {schemas}")

        # Test model operations
        logger.info("\nTesting model operations...")

        # Create test position
        test_position = db_manager.create(
            Position,
            symbol="TEST",
            quantity=Decimal("100"),
            entry_price=Decimal("150.50"),
            opened_at=datetime.utcnow(),
            strategy_id="test_strategy",
        )
        logger.info(f"✓ Created test position: {test_position}")

        # Query position
        position = db_manager.get_one(Position, symbol="TEST")
        if position:
            logger.info(f"✓ Retrieved position: {position}")

        # Update position
        updated = db_manager.update(
            Position,
            {"symbol": "TEST"},
            current_price=Decimal("155.75"),
            unrealized_pnl=Decimal("525.00"),
        )
        logger.info(f"✓ Updated {updated} position(s)")

        # Delete test data
        deleted = db_manager.delete(Position, symbol="TEST")
        logger.info(f"✓ Deleted {deleted} test position(s)")

        # Close connections
        db_manager.close()

        return True

    except Exception as e:
        logger.error(f"✗ SQLAlchemy connection failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def create_initial_data():
    """Create initial test data"""
    logger.info("\nCreating initial test data...")

    try:
        from src.bot.database.manager import get_db_manager
        from src.bot.database.models import Alert, AlertSeverity, FeatureSet, Model

        db_manager = get_db_manager()

        # Create feature set
        feature_set = db_manager.create(
            FeatureSet,
            name="technical_indicators_v1",
            description="Technical indicators for ML models",
            feature_count=50,
            feature_names=["rsi_14", "macd", "bb_upper", "bb_lower"],
            version="1.0.0",
            is_active=True,
        )
        logger.info(f"✓ Created feature set: {feature_set}")

        # Create model
        model = db_manager.create(
            Model,
            model_type="XGBoost",
            model_name="strategy_selector_v1",
            version="1.0.0",
            model_path="/models/xgboost_v1.joblib",
            feature_set_id=feature_set.feature_set_id,
            performance_metrics={"accuracy": 0.65, "sharpe": 1.2},
            is_active=True,
        )
        logger.info(f"✓ Created model: {model}")

        # Create alert
        alert = db_manager.create(
            Alert,
            timestamp=datetime.utcnow(),
            alert_type="system_startup",
            severity=AlertSeverity.INFO,
            component="database",
            message="Database initialized successfully",
        )
        logger.info(f"✓ Created alert: {alert}")

        return True

    except Exception as e:
        logger.error(f"✗ Failed to create initial data: {e}")
        return False


def show_statistics():
    """Show database statistics"""
    logger.info("\n" + "=" * 60)
    logger.info("Database Statistics")
    logger.info("=" * 60)

    try:
        from src.bot.database.manager import get_db_manager

        db_manager = get_db_manager()

        with db_manager.session_scope() as session:
            # Get table sizes
            result = session.execute(
                text(
                    """
                SELECT
                    schemaname,
                    tablename,
                    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size,
                    n_live_tup AS row_count
                FROM pg_stat_user_tables
                WHERE schemaname IN ('trading', 'ml', 'portfolio', 'monitoring')
                ORDER BY schemaname, tablename
            """
                )
            )

            logger.info("\nTable Statistics:")
            logger.info(f"{'Schema':<15} {'Table':<30} {'Size':<10} {'Rows':<10}")
            logger.info("-" * 65)

            total_rows = 0
            for row in result:
                logger.info(f"{row[0]:<15} {row[1]:<30} {row[2]:<10} {row[3]:<10}")
                total_rows += row[3] or 0

            logger.info(f"\nTotal rows across all tables: {total_rows}")

            # Get index statistics
            result = session.execute(
                text(
                    """
                SELECT
                    schemaname,
                    COUNT(*) as index_count,
                    pg_size_pretty(SUM(pg_relation_size(indexrelid))) as total_size
                FROM pg_stat_user_indexes
                WHERE schemaname IN ('trading', 'ml', 'portfolio', 'monitoring')
                GROUP BY schemaname
                ORDER BY schemaname
            """
                )
            )

            logger.info("\nIndex Statistics:")
            logger.info(f"{'Schema':<15} {'Count':<10} {'Total Size':<15}")
            logger.info("-" * 40)

            for row in result:
                logger.info(f"{row[0]:<15} {row[1]:<10} {row[2]:<15}")

        # Show connection pool status
        pool_status = db_manager.get_pool_status()
        logger.info("\nConnection Pool Status:")
        for key, value in pool_status.items():
            logger.info(f"  {key}: {value}")

        return True

    except Exception as e:
        logger.error(f"✗ Failed to get statistics: {e}")
        return False


def main():
    """Main test function"""
    logger.info("=" * 60)
    logger.info("PostgreSQL Connection Test")
    logger.info("Phase 2.5 - Day 2")
    logger.info("=" * 60)

    # Check if Docker is running
    logger.info("\nChecking Docker status...")
    import subprocess

    try:
        result = subprocess.run(["docker", "ps"], capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            logger.warning("Docker is not running. Please start Docker and run:")
            logger.warning("  cd deploy/postgres && docker-compose up -d")
            logger.warning("\nContinuing with assumption that PostgreSQL is available...")
    except Exception as e:
        logger.warning(f"Could not check Docker status: {e}")

    # Run tests
    tests_passed = []

    # Test 1: Direct connection
    if test_direct_connection():
        tests_passed.append("Direct Connection")

    # Test 2: SQLAlchemy connection
    if test_sqlalchemy_connection():
        tests_passed.append("SQLAlchemy Connection")

    # Test 3: Create initial data
    if create_initial_data():
        tests_passed.append("Initial Data Creation")

    # Test 4: Show statistics
    if show_statistics():
        tests_passed.append("Statistics")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)

    if len(tests_passed) == 4:
        logger.info("✅ All tests passed!")
        logger.info("\nPostgreSQL is ready for production use.")
        logger.info("\nNext steps:")
        logger.info("1. Run data migration: python scripts/migrate_to_postgres.py")
        logger.info("2. Update application code to use DatabaseManager")
        logger.info("3. Run performance benchmarks")
    else:
        logger.error(f"❌ Some tests failed. Passed: {tests_passed}")
        logger.error("\nPlease ensure PostgreSQL is running:")
        logger.error("  cd deploy/postgres && docker-compose up -d")

    return len(tests_passed) == 4


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
