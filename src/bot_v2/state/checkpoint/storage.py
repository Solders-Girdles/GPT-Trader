"""Atomic checkpoint storage operations"""

import json
import logging
from pathlib import Path
from typing import Any

from bot_v2.state.checkpoint.models import Checkpoint, CheckpointConfig
from bot_v2.state.utils import (
    AtomicFileStorage,
    calculate_data_hash,
    decompress_data,
    deserialize_from_json,
)

logger = logging.getLogger(__name__)


class CheckpointStorage:
    """Handles atomic storage and retrieval of checkpoints"""

    def __init__(self, config: CheckpointConfig) -> None:
        self.config = config
        self.checkpoint_path = Path(config.checkpoint_dir)
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        # Use shared atomic storage utility
        self.storage = AtomicFileStorage(self.checkpoint_path)

    async def store_checkpoint_atomic(self, checkpoint: Checkpoint, data: bytes) -> bool:
        """Store checkpoint atomically"""
        try:
            # Use AtomicFileStorage for atomic write
            checkpoint_filename = f"{checkpoint.checkpoint_id}.checkpoint"
            self.storage.write_atomic(checkpoint_filename, data)

            # Write metadata using AtomicFileStorage
            metadata = checkpoint.to_dict()
            del metadata["state_snapshot"]  # Don't duplicate large data
            metadata_bytes = json.dumps(metadata, indent=2, default=str).encode("utf-8")
            metadata_filename = f"{checkpoint.checkpoint_id}.meta"
            self.storage.write_atomic(metadata_filename, metadata_bytes)

            logger.debug(f"Checkpoint {checkpoint.checkpoint_id} stored successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to store checkpoint: {e}")

            # Cleanup any partially written files
            for suffix in [".checkpoint", ".meta"]:
                filename = f"{checkpoint.checkpoint_id}{suffix}"
                self.storage.delete(filename)

            return False

    def load_checkpoint_from_disk(self, checkpoint_id: str) -> Checkpoint | None:
        """Load checkpoint from disk"""
        try:
            metadata_filename = f"{checkpoint_id}.meta"
            checkpoint_filename = f"{checkpoint_id}.checkpoint"

            if not self.storage.exists(metadata_filename) or not self.storage.exists(
                checkpoint_filename
            ):
                return None

            # Load metadata using AtomicFileStorage
            metadata_bytes = self.storage.read(metadata_filename)
            metadata = json.loads(metadata_bytes.decode("utf-8"))

            # Load state snapshot using AtomicFileStorage
            data = self.storage.read(checkpoint_filename)

            # Decompress if needed
            if self.config.compression_enabled:
                data = decompress_data(data)

            state_snapshot = json.loads(data)
            metadata["state_snapshot"] = state_snapshot

            return Checkpoint.from_dict(metadata)

        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_id}: {e}")
            return None

    def calculate_consistency_hash(self, state_snapshot: dict[str, Any]) -> str:
        """Calculate hash for consistency verification"""
        # Use shared hash calculation utility
        return calculate_data_hash(state_snapshot)

    async def verify_checkpoint_integrity(self, checkpoint: Checkpoint) -> bool:
        """Verify checkpoint file integrity"""
        try:
            checkpoint_filename = f"{checkpoint.checkpoint_id}.checkpoint"

            if not self.storage.exists(checkpoint_filename):
                logger.error(f"Checkpoint file {checkpoint_filename} not found")
                return False

            # Load and verify data using AtomicFileStorage
            data = self.storage.read(checkpoint_filename)

            # Decompress if needed
            if self.config.compression_enabled:
                data = decompress_data(data)

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
                filename = f"{checkpoint_id}{suffix}"
                self.storage.delete(filename)
            return True
        except Exception as e:
            logger.error(f"Failed to delete checkpoint files: {e}")
            return False
