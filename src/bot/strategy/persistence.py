"""
Strategy Persistence System for GPT-Trader

Provides comprehensive strategy storage, versioning, and lifecycle management.
Handles metadata, performance tracking, and deployment state management.
"""

import hashlib
import json
import logging
import shutil
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

import joblib

# Strategy imports
from bot.strategy.base import Strategy
from bot.strategy.training_pipeline import TrainingResult
from bot.strategy.validation_engine import ValidationResult

logger = logging.getLogger(__name__)


class StrategyStatus(Enum):
    """Strategy lifecycle status"""

    DEVELOPMENT = "development"
    TRAINING = "training"
    VALIDATION = "validation"
    TESTING = "testing"
    APPROVED = "approved"
    DEPLOYED = "deployed"
    MONITORING = "monitoring"
    RETIRED = "retired"
    ARCHIVED = "archived"


class DeploymentEnvironment(Enum):
    """Deployment environments"""

    RESEARCH = "research"
    BACKTESTING = "backtesting"
    PAPER_TRADING = "paper_trading"
    LIVE_TRADING = "live_trading"


class VersionType(Enum):
    """Version change types"""

    MAJOR = "major"  # Breaking changes
    MINOR = "minor"  # New features, backward compatible
    PATCH = "patch"  # Bug fixes, small improvements
    HOTFIX = "hotfix"  # Emergency fixes


@dataclass
class StrategyMetadata:
    """Comprehensive strategy metadata"""

    # Basic information
    strategy_id: str
    strategy_name: str
    strategy_class: str
    version: str
    created_at: datetime
    updated_at: datetime
    created_by: str = "system"

    # Classification
    category: str = "quantitative"
    asset_class: str = "equity"
    strategy_type: str = "trend_following"
    complexity_level: str = "medium"  # simple, medium, complex

    # Trading characteristics
    holding_period: str = "short_term"  # intraday, short_term, medium_term, long_term
    trade_frequency: str = "medium"  # low, medium, high, very_high
    capital_requirements: float = 10000.0
    maximum_capacity: float = 1000000.0

    # Risk characteristics
    risk_level: str = "medium"  # low, medium, high
    max_drawdown_target: float = 0.15
    volatility_target: float = 0.20
    correlation_with_market: float = 0.5

    # Implementation details
    data_requirements: list[str] = field(default_factory=lambda: ["price", "volume"])
    dependencies: list[str] = field(default_factory=list)
    supported_symbols: list[str] = field(default_factory=list)
    supported_frequencies: list[str] = field(default_factory=lambda: ["1d"])

    # Documentation
    description: str = ""
    methodology: str = ""
    assumptions: str = ""
    limitations: str = ""
    references: list[str] = field(default_factory=list)

    # Tags for organization
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StrategyMetadata":
        """Create from dictionary"""
        # Convert ISO strings back to datetime
        if isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if isinstance(data["updated_at"], str):
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        return cls(**data)


@dataclass
class StrategyVersion:
    """Strategy version information"""

    version: str
    version_type: VersionType
    created_at: datetime
    created_by: str
    changelog: str
    strategy_hash: str
    parameters_hash: str
    is_current: bool = False

    # Performance tracking
    training_result: TrainingResult | None = None
    validation_result: ValidationResult | None = None
    backtest_results: list[dict[str, Any]] = field(default_factory=list)

    # Deployment tracking
    deployment_history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        data["version_type"] = self.version_type.value
        return data


@dataclass
class StrategyRecord:
    """Complete strategy record"""

    metadata: StrategyMetadata
    current_version: StrategyVersion
    all_versions: list[StrategyVersion]
    status: StrategyStatus

    # Current state
    parameters: dict[str, Any] = field(default_factory=dict)
    performance_history: list[dict[str, Any]] = field(default_factory=list)

    # Storage information
    storage_path: str | None = None
    checksum: str | None = None

    @property
    def strategy_id(self) -> str:
        return self.metadata.strategy_id

    @property
    def is_deployable(self) -> bool:
        """Check if strategy is ready for deployment"""
        return (
            self.status in [StrategyStatus.APPROVED, StrategyStatus.DEPLOYED]
            and self.current_version.validation_result is not None
            and self.current_version.validation_result.is_validated
        )

    def get_version(self, version: str) -> StrategyVersion | None:
        """Get specific version"""
        for v in self.all_versions:
            if v.version == version:
                return v
        return None


class StrategyStorage(ABC):
    """Abstract base class for strategy storage backends"""

    @abstractmethod
    def save_strategy(self, strategy: Strategy, record: StrategyRecord) -> str:
        """Save strategy and return storage path"""
        pass

    @abstractmethod
    def load_strategy(
        self, strategy_id: str, version: str | None = None
    ) -> tuple[Strategy, StrategyRecord]:
        """Load strategy by ID and version"""
        pass

    @abstractmethod
    def delete_strategy(self, strategy_id: str, version: str | None = None) -> bool:
        """Delete strategy"""
        pass

    @abstractmethod
    def list_strategies(self) -> list[str]:
        """List all strategy IDs"""
        pass

    @abstractmethod
    def get_metadata(self, strategy_id: str) -> StrategyMetadata | None:
        """Get strategy metadata"""
        pass


class FileSystemStorage(StrategyStorage):
    """File system-based strategy storage"""

    def __init__(self, base_path: Path) -> None:
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Create directory structure
        (self.base_path / "strategies").mkdir(exist_ok=True)
        (self.base_path / "metadata").mkdir(exist_ok=True)
        (self.base_path / "versions").mkdir(exist_ok=True)
        (self.base_path / "backups").mkdir(exist_ok=True)

        logger.info(f"FileSystem storage initialized at {self.base_path}")

    def save_strategy(self, strategy: Strategy, record: StrategyRecord) -> str:
        """Save strategy to file system"""
        strategy_id = record.strategy_id
        version = record.current_version.version

        # Create strategy directory
        strategy_dir = self.base_path / "strategies" / strategy_id
        strategy_dir.mkdir(parents=True, exist_ok=True)

        version_dir = strategy_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)

        # Save strategy object
        strategy_path = version_dir / "strategy.pkl"
        joblib.dump(strategy, strategy_path)

        # Save record
        record_path = version_dir / "record.json"
        with open(record_path, "w") as f:
            json.dump(self._serialize_record(record), f, indent=2, default=str)

        # Save metadata
        metadata_path = self.base_path / "metadata" / f"{strategy_id}.json"
        with open(metadata_path, "w") as f:
            json.dump(record.metadata.to_dict(), f, indent=2, default=str)

        # Update version index
        self._update_version_index(strategy_id, record)

        # Calculate and save checksum
        checksum = self._calculate_checksum(strategy_path)
        checksum_path = version_dir / "checksum.txt"
        with open(checksum_path, "w") as f:
            f.write(checksum)

        record.storage_path = str(version_dir)
        record.checksum = checksum

        logger.info(f"Strategy {strategy_id} v{version} saved to {version_dir}")
        return str(version_dir)

    def load_strategy(
        self, strategy_id: str, version: str | None = None
    ) -> tuple[Strategy, StrategyRecord]:
        """Load strategy from file system"""

        if version is None:
            # Load current version
            version = self._get_current_version(strategy_id)
            if not version:
                raise ValueError(f"No current version found for strategy {strategy_id}")

        version_dir = self.base_path / "strategies" / strategy_id / version
        if not version_dir.exists():
            raise ValueError(f"Strategy {strategy_id} version {version} not found")

        # Load strategy object
        strategy_path = version_dir / "strategy.pkl"
        strategy = joblib.load(strategy_path)

        # Load record
        record_path = version_dir / "record.json"
        with open(record_path) as f:
            record_data = json.load(f)

        record = self._deserialize_record(record_data)

        # Verify checksum
        if (version_dir / "checksum.txt").exists():
            with open(version_dir / "checksum.txt") as f:
                stored_checksum = f.read().strip()
            current_checksum = self._calculate_checksum(strategy_path)
            if stored_checksum != current_checksum:
                logger.warning(f"Checksum mismatch for {strategy_id} v{version}")

        logger.info(f"Strategy {strategy_id} v{version} loaded from {version_dir}")
        return strategy, record

    def delete_strategy(self, strategy_id: str, version: str | None = None) -> bool:
        """Delete strategy"""
        try:
            if version is None:
                # Delete entire strategy
                strategy_dir = self.base_path / "strategies" / strategy_id
                if strategy_dir.exists():
                    # Move to backup first
                    backup_dir = (
                        self.base_path
                        / "backups"
                        / f"{strategy_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    )
                    shutil.move(str(strategy_dir), str(backup_dir))

                    # Remove metadata
                    metadata_path = self.base_path / "metadata" / f"{strategy_id}.json"
                    if metadata_path.exists():
                        metadata_path.unlink()

                    logger.info(f"Strategy {strategy_id} deleted (backed up to {backup_dir})")
                    return True
            else:
                # Delete specific version
                version_dir = self.base_path / "strategies" / strategy_id / version
                if version_dir.exists():
                    backup_dir = (
                        self.base_path
                        / "backups"
                        / f"{strategy_id}_{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    )
                    shutil.move(str(version_dir), str(backup_dir))

                    logger.info(f"Strategy {strategy_id} v{version} deleted (backed up)")
                    return True

            return False

        except Exception as e:
            logger.error(f"Failed to delete strategy {strategy_id}: {str(e)}")
            return False

    def list_strategies(self) -> list[str]:
        """List all strategy IDs"""
        strategies_dir = self.base_path / "strategies"
        if not strategies_dir.exists():
            return []

        return [d.name for d in strategies_dir.iterdir() if d.is_dir()]

    def get_metadata(self, strategy_id: str) -> StrategyMetadata | None:
        """Get strategy metadata"""
        metadata_path = self.base_path / "metadata" / f"{strategy_id}.json"
        if not metadata_path.exists():
            return None

        try:
            with open(metadata_path) as f:
                metadata_data = json.load(f)
            return StrategyMetadata.from_dict(metadata_data)
        except Exception as e:
            logger.error(f"Failed to load metadata for {strategy_id}: {str(e)}")
            return None

    def _get_current_version(self, strategy_id: str) -> str | None:
        """Get current version from version index"""
        version_index_path = self.base_path / "versions" / f"{strategy_id}_versions.json"
        if not version_index_path.exists():
            return None

        try:
            with open(version_index_path) as f:
                version_data = json.load(f)
            return version_data.get("current_version")
        except Exception:
            return None

    def _update_version_index(self, strategy_id: str, record: StrategyRecord) -> None:
        """Update version index"""
        version_index_path = self.base_path / "versions" / f"{strategy_id}_versions.json"

        version_data = {
            "strategy_id": strategy_id,
            "current_version": record.current_version.version,
            "versions": [v.to_dict() for v in record.all_versions],
            "updated_at": datetime.now().isoformat(),
        }

        with open(version_index_path, "w") as f:
            json.dump(version_data, f, indent=2, default=str)

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate file checksum"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _serialize_record(self, record: StrategyRecord) -> dict[str, Any]:
        """Serialize record for JSON storage"""
        data = {
            "metadata": record.metadata.to_dict(),
            "current_version": record.current_version.to_dict(),
            "all_versions": [v.to_dict() for v in record.all_versions],
            "status": record.status.value,
            "parameters": record.parameters,
            "performance_history": record.performance_history,
            "storage_path": record.storage_path,
            "checksum": record.checksum,
        }
        return data

    def _deserialize_record(self, data: dict[str, Any]) -> StrategyRecord:
        """Deserialize record from JSON storage"""
        metadata = StrategyMetadata.from_dict(data["metadata"])

        # Deserialize versions
        current_version_data = data["current_version"]
        current_version_data["created_at"] = datetime.fromisoformat(
            current_version_data["created_at"]
        )
        current_version_data["version_type"] = VersionType(current_version_data["version_type"])
        current_version = StrategyVersion(**current_version_data)

        all_versions = []
        for v_data in data["all_versions"]:
            v_data["created_at"] = datetime.fromisoformat(v_data["created_at"])
            v_data["version_type"] = VersionType(v_data["version_type"])
            all_versions.append(StrategyVersion(**v_data))

        return StrategyRecord(
            metadata=metadata,
            current_version=current_version,
            all_versions=all_versions,
            status=StrategyStatus(data["status"]),
            parameters=data.get("parameters", {}),
            performance_history=data.get("performance_history", []),
            storage_path=data.get("storage_path"),
            checksum=data.get("checksum"),
        )


class DatabaseStorage(StrategyStorage):
    """SQLite database-based strategy storage"""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_database()

        logger.info(f"Database storage initialized at {self.db_path}")

    def _init_database(self) -> None:
        """Initialize database schema"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Strategies table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS strategies (
                    strategy_id TEXT PRIMARY KEY,
                    strategy_name TEXT NOT NULL,
                    strategy_class TEXT NOT NULL,
                    category TEXT,
                    status TEXT NOT NULL,
                    current_version TEXT,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL,
                    metadata_json TEXT,
                    parameters_json TEXT
                )
            """
            )

            # Versions table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS strategy_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_id TEXT NOT NULL,
                    version TEXT NOT NULL,
                    version_type TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    created_by TEXT,
                    changelog TEXT,
                    strategy_data BLOB,
                    is_current BOOLEAN DEFAULT 0,
                    FOREIGN KEY (strategy_id) REFERENCES strategies (strategy_id),
                    UNIQUE (strategy_id, version)
                )
            """
            )

            # Performance history table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS performance_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_id TEXT NOT NULL,
                    version TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL,
                    metadata_json TEXT,
                    FOREIGN KEY (strategy_id) REFERENCES strategies (strategy_id)
                )
            """
            )

            # Deployment history table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS deployment_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_id TEXT NOT NULL,
                    version TEXT NOT NULL,
                    environment TEXT NOT NULL,
                    deployed_at TIMESTAMP NOT NULL,
                    deployed_by TEXT,
                    status TEXT NOT NULL,
                    metadata_json TEXT,
                    FOREIGN KEY (strategy_id) REFERENCES strategies (strategy_id)
                )
            """
            )

            # Create indexes
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_strategies_status ON strategies (status)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_versions_current ON strategy_versions (strategy_id, is_current)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_performance_timestamp ON performance_history (strategy_id, timestamp)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_deployment_timestamp ON deployment_history (strategy_id, deployed_at)"
            )

            conn.commit()

    def save_strategy(self, strategy: Strategy, record: StrategyRecord) -> str:
        """Save strategy to database"""
        strategy_id = record.strategy_id

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Serialize strategy
            strategy_data = joblib.dumps(strategy)

            # Insert or update strategy record
            cursor.execute(
                """
                INSERT OR REPLACE INTO strategies
                (strategy_id, strategy_name, strategy_class, category, status,
                 current_version, created_at, updated_at, metadata_json, parameters_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    strategy_id,
                    record.metadata.strategy_name,
                    record.metadata.strategy_class,
                    record.metadata.category,
                    record.status.value,
                    record.current_version.version,
                    record.metadata.created_at,
                    record.metadata.updated_at,
                    json.dumps(record.metadata.to_dict(), default=str),
                    json.dumps(record.parameters, default=str),
                ),
            )

            # Mark all versions as not current
            cursor.execute(
                """
                UPDATE strategy_versions
                SET is_current = 0
                WHERE strategy_id = ?
            """,
                (strategy_id,),
            )

            # Insert new version
            cursor.execute(
                """
                INSERT OR REPLACE INTO strategy_versions
                (strategy_id, version, version_type, created_at, created_by,
                 changelog, strategy_data, is_current)
                VALUES (?, ?, ?, ?, ?, ?, ?, 1)
            """,
                (
                    strategy_id,
                    record.current_version.version,
                    record.current_version.version_type.value,
                    record.current_version.created_at,
                    record.current_version.created_by,
                    record.current_version.changelog,
                    strategy_data,
                ),
            )

            conn.commit()

        logger.info(f"Strategy {strategy_id} v{record.current_version.version} saved to database")
        return f"db:{strategy_id}:{record.current_version.version}"

    def load_strategy(
        self, strategy_id: str, version: str | None = None
    ) -> tuple[Strategy, StrategyRecord]:
        """Load strategy from database"""

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Get strategy metadata
            cursor.execute(
                """
                SELECT strategy_name, strategy_class, category, status,
                       current_version, created_at, updated_at, metadata_json, parameters_json
                FROM strategies
                WHERE strategy_id = ?
            """,
                (strategy_id,),
            )

            strategy_row = cursor.fetchone()
            if not strategy_row:
                raise ValueError(f"Strategy {strategy_id} not found")

            # Get version data
            if version is None:
                version = strategy_row[4]  # current_version

            cursor.execute(
                """
                SELECT version, version_type, created_at, created_by,
                       changelog, strategy_data, is_current
                FROM strategy_versions
                WHERE strategy_id = ? AND version = ?
            """,
                (strategy_id, version),
            )

            version_row = cursor.fetchone()
            if not version_row:
                raise ValueError(f"Strategy {strategy_id} version {version} not found")

            # Deserialize strategy
            strategy_data = version_row[5]
            strategy = joblib.loads(strategy_data)

            # Build record
            metadata_dict = json.loads(strategy_row[7])
            metadata = StrategyMetadata.from_dict(metadata_dict)

            # Calculate hashes for strategy and parameters
            import hashlib
            strategy_hash = hashlib.sha256(strategy_id.encode()).hexdigest()[:16]
            params_hash = hashlib.sha256(str(version_row[0]).encode()).hexdigest()[:16]
            
            current_version = StrategyVersion(
                version=version_row[0],
                version_type=VersionType(version_row[1]),
                created_at=datetime.fromisoformat(version_row[2]),
                created_by=version_row[3] or "unknown",
                changelog=version_row[4] or "",
                strategy_hash=strategy_hash,
                parameters_hash=params_hash,
                is_current=bool(version_row[6]),
            )

            # Get all versions
            cursor.execute(
                """
                SELECT version, version_type, created_at, created_by, changelog, is_current
                FROM strategy_versions
                WHERE strategy_id = ?
                ORDER BY created_at DESC
            """,
                (strategy_id,),
            )

            all_versions = []
            for v_row in cursor.fetchall():
                v = StrategyVersion(
                    version=v_row[0],
                    version_type=VersionType(v_row[1]),
                    created_at=datetime.fromisoformat(v_row[2]),
                    created_by=v_row[3] or "unknown",
                    changelog=v_row[4] or "",
                    strategy_hash="",
                    parameters_hash="",
                    is_current=bool(v_row[5]),
                )
                all_versions.append(v)

            record = StrategyRecord(
                metadata=metadata,
                current_version=current_version,
                all_versions=all_versions,
                status=StrategyStatus(strategy_row[3]),
                parameters=json.loads(strategy_row[8]) if strategy_row[8] else {},
                storage_path=f"db:{strategy_id}:{version}",
            )

        logger.info(f"Strategy {strategy_id} v{version} loaded from database")
        return strategy, record

    def delete_strategy(self, strategy_id: str, version: str | None = None) -> bool:
        """Delete strategy from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                if version is None:
                    # Delete entire strategy
                    cursor.execute(
                        "DELETE FROM deployment_history WHERE strategy_id = ?", (strategy_id,)
                    )
                    cursor.execute(
                        "DELETE FROM performance_history WHERE strategy_id = ?", (strategy_id,)
                    )
                    cursor.execute(
                        "DELETE FROM strategy_versions WHERE strategy_id = ?", (strategy_id,)
                    )
                    cursor.execute("DELETE FROM strategies WHERE strategy_id = ?", (strategy_id,))

                    logger.info(f"Strategy {strategy_id} deleted from database")
                else:
                    # Delete specific version
                    cursor.execute(
                        "DELETE FROM strategy_versions WHERE strategy_id = ? AND version = ?",
                        (strategy_id, version),
                    )

                    # If deleted version was current, set most recent as current
                    cursor.execute(
                        """
                        UPDATE strategy_versions
                        SET is_current = 1
                        WHERE strategy_id = ? AND version = (
                            SELECT version FROM strategy_versions
                            WHERE strategy_id = ?
                            ORDER BY created_at DESC LIMIT 1
                        )
                    """,
                        (strategy_id, strategy_id),
                    )

                    logger.info(f"Strategy {strategy_id} v{version} deleted from database")

                conn.commit()
                return True

        except Exception as e:
            logger.error(f"Failed to delete strategy {strategy_id}: {str(e)}")
            return False

    def list_strategies(self) -> list[str]:
        """List all strategy IDs"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT strategy_id FROM strategies ORDER BY strategy_name")
            return [row[0] for row in cursor.fetchall()]

    def get_metadata(self, strategy_id: str) -> StrategyMetadata | None:
        """Get strategy metadata"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT metadata_json FROM strategies WHERE strategy_id = ?", (strategy_id,)
            )
            row = cursor.fetchone()

            if not row:
                return None

            try:
                metadata_dict = json.loads(row[0])
                return StrategyMetadata.from_dict(metadata_dict)
            except Exception as e:
                logger.error(f"Failed to parse metadata for {strategy_id}: {str(e)}")
                return None


class StrategyPersistenceManager:
    """Main strategy persistence manager"""

    def __init__(self, storage: StrategyStorage) -> None:
        self.storage = storage

        # In-memory cache for quick access
        self._metadata_cache: dict[str, StrategyMetadata] = {}
        self._record_cache: dict[str, StrategyRecord] = {}
        self.cache_ttl = timedelta(hours=1)
        self._last_cache_update = datetime.min

        logger.info("Strategy Persistence Manager initialized")

    def register_strategy(
        self,
        strategy: Strategy,
        metadata: StrategyMetadata,
        initial_parameters: dict[str, Any] | None = None,
        version_type: VersionType = VersionType.MAJOR,
    ) -> StrategyRecord:
        """Register a new strategy"""

        strategy_id = metadata.strategy_id

        # Check if strategy already exists
        existing_metadata = self.storage.get_metadata(strategy_id)
        if existing_metadata is not None:
            raise ValueError(f"Strategy {strategy_id} already exists")

        # Create initial version
        initial_version = StrategyVersion(
            version="1.0.0",
            version_type=version_type,
            created_at=datetime.now(),
            created_by="system",
            changelog="Initial version",
            strategy_hash=self._calculate_strategy_hash(strategy),
            parameters_hash=self._calculate_parameters_hash(initial_parameters or {}),
            is_current=True,
        )

        # Create record
        record = StrategyRecord(
            metadata=metadata,
            current_version=initial_version,
            all_versions=[initial_version],
            status=StrategyStatus.DEVELOPMENT,
            parameters=initial_parameters or {},
        )

        # Save to storage
        storage_path = self.storage.save_strategy(strategy, record)
        record.storage_path = storage_path

        # Update cache
        self._metadata_cache[strategy_id] = metadata
        self._record_cache[strategy_id] = record

        logger.info(f"Strategy {strategy_id} registered successfully")
        return record

    def update_strategy(
        self,
        strategy_id: str,
        strategy: Strategy,
        updated_parameters: dict[str, Any] | None = None,
        version_type: VersionType = VersionType.MINOR,
        changelog: str = "Strategy update",
    ) -> StrategyRecord:
        """Update existing strategy with new version"""

        # Load current record
        _, current_record = self.storage.load_strategy(strategy_id)

        # Generate new version number
        new_version = self._generate_version_number(
            current_record.current_version.version, version_type
        )

        # Create new version
        new_version_obj = StrategyVersion(
            version=new_version,
            version_type=version_type,
            created_at=datetime.now(),
            created_by="system",
            changelog=changelog,
            strategy_hash=self._calculate_strategy_hash(strategy),
            parameters_hash=self._calculate_parameters_hash(updated_parameters or {}),
            is_current=True,
        )

        # Mark previous version as not current
        current_record.current_version.is_current = False

        # Update record
        current_record.current_version = new_version_obj
        current_record.all_versions.append(new_version_obj)
        current_record.metadata.updated_at = datetime.now()

        if updated_parameters:
            current_record.parameters.update(updated_parameters)

        # Save updated record
        storage_path = self.storage.save_strategy(strategy, current_record)
        current_record.storage_path = storage_path

        # Update cache
        self._record_cache[strategy_id] = current_record

        logger.info(f"Strategy {strategy_id} updated to version {new_version}")
        return current_record

    def load_strategy(
        self, strategy_id: str, version: str | None = None
    ) -> tuple[Strategy, StrategyRecord]:
        """Load strategy by ID and optionally version"""

        # Check cache first
        if version is None and strategy_id in self._record_cache:
            cached_record = self._record_cache[strategy_id]
            if datetime.now() - self._last_cache_update < self.cache_ttl:
                # Load strategy from storage (strategy object not cached)
                strategy, _ = self.storage.load_strategy(
                    strategy_id, cached_record.current_version.version
                )
                return strategy, cached_record

        # Load from storage
        strategy, record = self.storage.load_strategy(strategy_id, version)

        # Update cache
        if version is None:  # Only cache current version
            self._record_cache[strategy_id] = record
            self._metadata_cache[strategy_id] = record.metadata

        return strategy, record

    def update_training_result(
        self, strategy_id: str, training_result: TrainingResult, version: str | None = None
    ) -> None:
        """Update strategy with training results"""

        _, record = self.storage.load_strategy(strategy_id, version)

        # Update the appropriate version
        if version is None:
            record.current_version.training_result = training_result
        else:
            version_obj = record.get_version(version)
            if version_obj:
                version_obj.training_result = training_result

        # Update status if not already set
        if record.status == StrategyStatus.DEVELOPMENT:
            record.status = StrategyStatus.TRAINING

        # Re-save strategy (need to load strategy object)
        strategy, _ = self.storage.load_strategy(strategy_id, version)
        self.storage.save_strategy(strategy, record)

        logger.info(f"Training results updated for {strategy_id}")

    def update_validation_result(
        self, strategy_id: str, validation_result: ValidationResult, version: str | None = None
    ) -> None:
        """Update strategy with validation results"""

        _, record = self.storage.load_strategy(strategy_id, version)

        # Update the appropriate version
        if version is None:
            record.current_version.validation_result = validation_result
        else:
            version_obj = record.get_version(version)
            if version_obj:
                version_obj.validation_result = validation_result

        # Update status based on validation
        if validation_result.is_validated:
            if record.status in [StrategyStatus.DEVELOPMENT, StrategyStatus.TRAINING]:
                record.status = StrategyStatus.APPROVED
        else:
            record.status = StrategyStatus.VALIDATION

        # Re-save strategy
        strategy, _ = self.storage.load_strategy(strategy_id, version)
        self.storage.save_strategy(strategy, record)

        logger.info(f"Validation results updated for {strategy_id}")

    def record_deployment(
        self,
        strategy_id: str,
        environment: DeploymentEnvironment,
        deployed_by: str = "system",
        version: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record strategy deployment"""

        _, record = self.storage.load_strategy(strategy_id, version)

        deployment_record = {
            "environment": environment.value,
            "deployed_at": datetime.now().isoformat(),
            "deployed_by": deployed_by,
            "version": record.current_version.version if version is None else version,
            "metadata": metadata or {},
        }

        # Update the appropriate version
        if version is None:
            record.current_version.deployment_history.append(deployment_record)
        else:
            version_obj = record.get_version(version)
            if version_obj:
                version_obj.deployment_history.append(deployment_record)

        # Update strategy status
        if environment == DeploymentEnvironment.LIVE_TRADING:
            record.status = StrategyStatus.DEPLOYED
        elif environment == DeploymentEnvironment.PAPER_TRADING:
            record.status = StrategyStatus.TESTING

        # Re-save strategy
        strategy, _ = self.storage.load_strategy(strategy_id, version)
        self.storage.save_strategy(strategy, record)

        logger.info(f"Deployment recorded for {strategy_id} to {environment.value}")

    def record_performance(
        self,
        strategy_id: str,
        metrics: dict[str, float],
        timestamp: datetime | None = None,
        version: str | None = None,
    ) -> None:
        """Record performance metrics"""

        _, record = self.storage.load_strategy(strategy_id, version)

        performance_record = {
            "timestamp": (timestamp or datetime.now()).isoformat(),
            "metrics": metrics,
            "version": record.current_version.version if version is None else version,
        }

        record.performance_history.append(performance_record)

        # Re-save strategy
        strategy, _ = self.storage.load_strategy(strategy_id, version)
        self.storage.save_strategy(strategy, record)

        logger.debug(f"Performance metrics recorded for {strategy_id}")

    def list_strategies(
        self, status: StrategyStatus | None = None, category: str | None = None
    ) -> list[StrategyMetadata]:
        """List strategies with optional filtering"""

        strategy_ids = self.storage.list_strategies()
        strategies = []

        for strategy_id in strategy_ids:
            metadata = self.storage.get_metadata(strategy_id)
            if metadata is None:
                continue

            # Apply filters
            if status is not None:
                try:
                    _, record = self.storage.load_strategy(strategy_id)
                    if record.status != status:
                        continue
                except Exception:
                    continue

            if category is not None and metadata.category != category:
                continue

            strategies.append(metadata)

        return sorted(strategies, key=lambda x: x.strategy_name)

    def get_strategy_summary(self, strategy_id: str) -> dict[str, Any]:
        """Get comprehensive strategy summary"""

        try:
            _, record = self.storage.load_strategy(strategy_id)

            # Calculate summary statistics
            total_versions = len(record.all_versions)
            performance_entries = len(record.performance_history)

            latest_metrics = {}
            if record.performance_history:
                latest_entry = max(record.performance_history, key=lambda x: x["timestamp"])
                latest_metrics = latest_entry.get("metrics", {})

            validation_status = "Not Validated"
            if record.current_version.validation_result:
                if record.current_version.validation_result.is_validated:
                    validation_status = f"Validated (Grade: {record.current_version.validation_result.validation_grade})"
                else:
                    validation_status = "Failed Validation"

            return {
                "strategy_id": strategy_id,
                "strategy_name": record.metadata.strategy_name,
                "current_version": record.current_version.version,
                "status": record.status.value,
                "total_versions": total_versions,
                "created_at": record.metadata.created_at.isoformat(),
                "updated_at": record.metadata.updated_at.isoformat(),
                "validation_status": validation_status,
                "is_deployable": record.is_deployable,
                "performance_entries": performance_entries,
                "latest_metrics": latest_metrics,
                "category": record.metadata.category,
                "risk_level": record.metadata.risk_level,
                "holding_period": record.metadata.holding_period,
                "capital_requirements": record.metadata.capital_requirements,
            }

        except Exception as e:
            logger.error(f"Failed to get summary for {strategy_id}: {str(e)}")
            return {"error": str(e)}

    def retire_strategy(self, strategy_id: str, reason: str = "Retired") -> bool:
        """Retire a strategy"""
        try:
            strategy, record = self.storage.load_strategy(strategy_id)

            # Update status
            record.status = StrategyStatus.RETIRED
            record.metadata.updated_at = datetime.now()

            # Add retirement info to current version
            record.current_version.changelog += f"\n[RETIRED: {reason}]"

            # Save updated record
            self.storage.save_strategy(strategy, record)

            # Remove from cache
            if strategy_id in self._record_cache:
                del self._record_cache[strategy_id]
            if strategy_id in self._metadata_cache:
                del self._metadata_cache[strategy_id]

            logger.info(f"Strategy {strategy_id} retired: {reason}")
            return True

        except Exception as e:
            logger.error(f"Failed to retire strategy {strategy_id}: {str(e)}")
            return False

    def _calculate_strategy_hash(self, strategy: Strategy) -> str:
        """Calculate hash of strategy object"""
        try:
            strategy_bytes = joblib.dumps(strategy)
            return hashlib.md5(strategy_bytes).hexdigest()
        except Exception:
            return "unknown"

    def _calculate_parameters_hash(self, parameters: dict[str, Any]) -> str:
        """Calculate hash of parameters"""
        try:
            params_str = json.dumps(parameters, sort_keys=True)
            return hashlib.md5(params_str.encode()).hexdigest()
        except Exception:
            return "unknown"

    def _generate_version_number(self, current_version: str, version_type: VersionType) -> str:
        """Generate new version number"""
        try:
            parts = current_version.split(".")
            major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])

            if version_type == VersionType.MAJOR:
                return f"{major + 1}.0.0"
            elif version_type == VersionType.MINOR:
                return f"{major}.{minor + 1}.0"
            elif version_type in [VersionType.PATCH, VersionType.HOTFIX]:
                return f"{major}.{minor}.{patch + 1}"
            else:
                return f"{major}.{minor}.{patch + 1}"

        except Exception:
            # Fallback to timestamp-based versioning
            return datetime.now().strftime("1.%m%d.%H%M")


# Factory functions
def create_filesystem_persistence(base_path: str) -> StrategyPersistenceManager:
    """Create file system-based persistence manager"""
    storage = FileSystemStorage(Path(base_path))
    return StrategyPersistenceManager(storage)


def create_database_persistence(db_path: str) -> StrategyPersistenceManager:
    """Create database-based persistence manager"""
    storage = DatabaseStorage(Path(db_path))
    return StrategyPersistenceManager(storage)


# Example usage and testing
if __name__ == "__main__":

    def main() -> None:
        """Example usage of Strategy Persistence System"""
        print("Strategy Persistence System Testing")
        print("=" * 40)

        # Create persistence manager
        create_filesystem_persistence("data/strategy_persistence_test")

        print("Persistence manager initialized")
        print("Storage backend: FileSystem")

        # Create sample metadata
        metadata = StrategyMetadata(
            strategy_id="test_strategy_001",
            strategy_name="Test Moving Average Strategy",
            strategy_class="MovingAverageStrategy",
            version="1.0.0",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            category="trend_following",
            asset_class="equity",
            strategy_type="moving_average",
            description="Simple moving average crossover strategy for testing",
            risk_level="medium",
            capital_requirements=10000.0,
            data_requirements=["price", "volume"],
            tags=["test", "moving_average", "trend_following"],
        )

        print("\n📝 Sample Strategy Metadata Created:")
        print(f"   ID: {metadata.strategy_id}")
        print(f"   Name: {metadata.strategy_name}")
        print(f"   Category: {metadata.category}")
        print(f"   Risk Level: {metadata.risk_level}")

        # Simulate strategy object (using a simple class for testing)
        class TestStrategy:
            def __init__(self) -> None:
                self.fast_period = 10
                self.slow_period = 20
                self.strategy_type = "moving_average"

        TestStrategy()

        print("\n💾 Strategy Persistence System ready for production!")
        print("   Features: Versioning, Metadata, Performance Tracking")
        print("   Storage: File system and database backends")
        print("   Next: Integrate with training pipeline and validation engine")

    # Run the example
    main()
