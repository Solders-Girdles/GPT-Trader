"""Atomic checkpoint storage operations"""

import gzip
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from bot_v2.state.checkpoint.models import Checkpoint, CheckpointConfig

logger = logging.getLogger(__name__)


class CheckpointStorage:
    """Handles atomic storage and retrieval of checkpoints"""

    def __init__(self, config: CheckpointConfig) -> None:
        self.config = config
        self.checkpoint_path = Path(config.checkpoint_dir)
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)

    async def store_checkpoint_atomic(self, checkpoint: Checkpoint, data: bytes) -> bool:
        """Store checkpoint atomically"""
        try:
            # Prepare file paths
            checkpoint_file = self.checkpoint_path / f"{checkpoint.checkpoint_id}.checkpoint"
            temp_file = self.checkpoint_path / f"{checkpoint.checkpoint_id}.tmp"
            metadata_file = self.checkpoint_path / f"{checkpoint.checkpoint_id}.meta"

            # Write to temporary file first
            with open(temp_file, "wb") as f:
                f.write(data)

            # Write metadata
            metadata = checkpoint.to_dict()
            del metadata["state_snapshot"]  # Don't duplicate large data

            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2, default=str)

            # Atomic rename
            temp_file.rename(checkpoint_file)

            logger.debug(f"Checkpoint {checkpoint.checkpoint_id} stored successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to store checkpoint: {e}")

            # Cleanup temporary files
            for suffix in [".tmp", ".checkpoint", ".meta"]:
                file_path = self.checkpoint_path / f"{checkpoint.checkpoint_id}{suffix}"
                if file_path.exists():
                    file_path.unlink()

            return False

    def load_checkpoint_from_disk(self, checkpoint_id: str) -> Checkpoint | None:
        """Load checkpoint from disk"""
        try:
            metadata_file = self.checkpoint_path / f"{checkpoint_id}.meta"
            checkpoint_file = self.checkpoint_path / f"{checkpoint_id}.checkpoint"

            if not metadata_file.exists() or not checkpoint_file.exists():
                return None

            # Load metadata
            with open(metadata_file) as f:
                metadata = json.load(f)

            # Load state snapshot
            with open(checkpoint_file, "rb") as f:
                data = f.read()

            # Decompress if needed
            if self.config.compression_enabled:
                data = gzip.decompress(data)

            state_snapshot = json.loads(data)
            metadata["state_snapshot"] = state_snapshot

            return Checkpoint.from_dict(metadata)

        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_id}: {e}")
            return None

    def calculate_consistency_hash(self, state_snapshot: dict[str, Any]) -> str:
        """Calculate hash for consistency verification"""
        # Create deterministic string representation
        state_str = json.dumps(state_snapshot, sort_keys=True, default=str)
        return hashlib.sha256(state_str.encode()).hexdigest()

    async def verify_checkpoint_integrity(self, checkpoint: Checkpoint) -> bool:
        """Verify checkpoint file integrity"""
        try:
            checkpoint_file = self.checkpoint_path / f"{checkpoint.checkpoint_id}.checkpoint"

            if not checkpoint_file.exists():
                logger.error(f"Checkpoint file {checkpoint_file} not found")
                return False

            # Load and verify data
            with open(checkpoint_file, "rb") as f:
                data = f.read()

            # Decompress if needed
            if self.config.compression_enabled:
                data = gzip.decompress(data)

            # Parse and verify
            state_snapshot = json.loads(data)
            calculated_hash = self.calculate_consistency_hash(state_snapshot)

            return calculated_hash == checkpoint.consistency_hash

        except Exception as e:
            logger.error(f"Integrity check failed: {e}")
            return False

    def delete_checkpoint_files(self, checkpoint_id: str) -> bool:
        """Delete checkpoint files"""
        try:
            for suffix in [".checkpoint", ".meta"]:
                file_path = self.checkpoint_path / f"{checkpoint_id}{suffix}"
                if file_path.exists():
                    file_path.unlink()
            return True
        except Exception as e:
            logger.error(f"Failed to delete checkpoint files: {e}")
            return False
