"""Checkpoint data models and configuration"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class CheckpointStatus(Enum):
    """Checkpoint status states"""

    CREATING = "creating"
    VALID = "valid"
    INVALID = "invalid"
    CORRUPTED = "corrupted"
    DELETED = "deleted"


class CheckpointType(Enum):
    """Types of checkpoints"""

    MANUAL = "manual"
    AUTOMATIC = "automatic"
    EMERGENCY = "emergency"
    PRE_UPGRADE = "pre_upgrade"
    DAILY = "daily"


@dataclass
class Checkpoint:
    """Checkpoint data structure"""

    checkpoint_id: str
    timestamp: datetime
    state_snapshot: dict[str, Any]
    version: int
    consistency_hash: str
    size_bytes: int
    status: CheckpointStatus = CheckpointStatus.VALID
    checkpoint_type: CheckpointType = CheckpointType.AUTOMATIC
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return {
            "checkpoint_id": self.checkpoint_id,
            "timestamp": self.timestamp.isoformat(),
            "state_snapshot": self.state_snapshot,
            "version": self.version,
            "consistency_hash": self.consistency_hash,
            "size_bytes": self.size_bytes,
            "status": self.status.value,
            "checkpoint_type": self.checkpoint_type.value,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Checkpoint":
        """Create from dictionary"""
        return cls(
            checkpoint_id=data["checkpoint_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            state_snapshot=data["state_snapshot"],
            version=data["version"],
            consistency_hash=data["consistency_hash"],
            size_bytes=data["size_bytes"],
            status=CheckpointStatus(data.get("status", "valid")),
            checkpoint_type=CheckpointType(data.get("checkpoint_type", "automatic")),
            metadata=data.get("metadata", {}),
        )


@dataclass
class CheckpointConfig:
    """Configuration for checkpoint operations"""

    checkpoint_dir: str = "/tmp/bot_v2/checkpoints"
    max_checkpoints: int = 50
    checkpoint_interval_minutes: int = 15
    compression_enabled: bool = True
    verification_enabled: bool = True
    pause_trading_during_checkpoint: bool = True
    checkpoint_timeout_seconds: int = 30
    retention_days: int = 7
