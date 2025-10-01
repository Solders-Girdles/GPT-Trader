"""Transport service for multi-tier backup storage.

Handles storage and retrieval of backups across local, network, and cloud tiers.
"""

import logging
import shutil
from pathlib import Path
from typing import Any

from bot_v2.state.backup.models import StorageTier

logger = logging.getLogger(__name__)

# Optional cloud storage support
try:
    import boto3

    S3_AVAILABLE = True
except ImportError:
    boto3 = None  # type: ignore[assignment]
    S3_AVAILABLE = False
    logger.warning("Boto3 not available, S3 backup disabled")


class TransportService:
    """Service for transporting backup data across storage tiers."""

    def __init__(
        self,
        local_path: Path,
        backup_path: Path,
        network_path: Path | None = None,
        s3_bucket: str | None = None,
        enable_s3: bool = True,
    ) -> None:
        """
        Initialize transport service.

        Args:
            local_path: Local storage path
            backup_path: Primary backup directory path
            network_path: Network storage path (optional)
            s3_bucket: S3 bucket name (optional)
            enable_s3: Whether to enable S3 storage
        """
        self.local_path = local_path
        self.backup_path = backup_path
        self.network_path = network_path
        self.s3_bucket = s3_bucket
        self._s3_client: Any | None = None

        # Initialize storage paths
        self._init_paths()

        # Initialize S3 if available
        if enable_s3 and s3_bucket and S3_AVAILABLE:
            self._init_s3()

    def _init_paths(self) -> None:
        """Initialize storage paths."""
        self.backup_path.mkdir(parents=True, exist_ok=True)
        self.local_path.mkdir(parents=True, exist_ok=True)

        if self.network_path:
            try:
                self.network_path.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                logger.warning(f"Unable to initialize network storage path: {exc}")
                self.network_path = None

    def _init_s3(self) -> None:
        """Initialize S3 client."""
        if boto3 is None:
            self._s3_client = None
            return

        try:
            self._s3_client = boto3.client("s3")
            # Verify bucket exists
            self._s3_client.head_bucket(Bucket=self.s3_bucket)
            logger.info(f"S3 backup initialized with bucket {self.s3_bucket}")

        except Exception as e:
            logger.warning(f"S3 initialization failed: {e}")
            self._s3_client = None

    async def store(self, backup_id: str, data: bytes, tier: StorageTier) -> str:
        """
        Store backup data to appropriate tier.

        Args:
            backup_id: Unique backup identifier
            data: Backup data
            tier: Storage tier

        Returns:
            Storage path/location
        """
        if tier == StorageTier.LOCAL:
            return await self._store_local(backup_id, data)

        elif tier == StorageTier.NETWORK:
            if self.network_path:
                return await self._store_network(backup_id, data)
            else:
                logger.warning("Network storage not available, falling back to local")
                return await self._store_local(backup_id, data)

        elif tier == StorageTier.CLOUD:
            if self._s3_client:
                return await self._store_s3(backup_id, data)
            else:
                logger.warning("S3 storage not available, falling back to local")
                return await self._store_local(backup_id, data)

        elif tier == StorageTier.ARCHIVE:
            if self._s3_client:
                return await self._store_s3_archive(backup_id, data)
            else:
                logger.warning("S3 archive not available, falling back to local")
                return await self._store_local(backup_id, data)

        return await self._store_local(backup_id, data)

    async def _store_local(self, backup_id: str, data: bytes) -> str:
        """Store backup locally."""
        primary_path = self.backup_path / f"{backup_id}.backup"

        with open(primary_path, "wb") as f:
            f.write(data)

        # Create mirror copy if local path differs
        if self.local_path != self.backup_path:
            mirror_path = self.local_path / f"{backup_id}.backup"
            try:
                shutil.copyfile(primary_path, mirror_path)
            except Exception as exc:
                logger.debug(f"Failed to mirror backup file: {exc}")

        return str(primary_path)

    async def _store_network(self, backup_id: str, data: bytes) -> str:
        """Store backup to network storage."""
        if not self.network_path:
            raise ValueError("Network storage path not configured")

        file_path = self.network_path / f"{backup_id}.backup"

        with open(file_path, "wb") as f:
            f.write(data)

        return str(file_path)

    async def _store_s3(self, backup_id: str, data: bytes) -> str:
        """Store backup to S3."""
        try:
            key = f"backups/{backup_id}.backup"

            self._s3_client.put_object(
                Bucket=self.s3_bucket, Key=key, Body=data, StorageClass="STANDARD_IA"
            )

            return f"s3://{self.s3_bucket}/{key}"

        except Exception as e:
            logger.error(f"S3 storage failed: {e}")
            # Fallback to local
            return await self._store_local(backup_id, data)

    async def _store_s3_archive(self, backup_id: str, data: bytes) -> str:
        """Store backup to S3 Glacier."""
        try:
            key = f"archive/{backup_id}.backup"

            self._s3_client.put_object(
                Bucket=self.s3_bucket, Key=key, Body=data, StorageClass="GLACIER"
            )

            return f"s3://{self.s3_bucket}/{key}"

        except Exception as e:
            logger.error(f"S3 archive storage failed: {e}")
            return await self._store_local(backup_id, data)

    async def retrieve(self, backup_id: str, tier: StorageTier) -> bytes | None:
        """
        Retrieve backup data from storage.

        Args:
            backup_id: Unique backup identifier
            tier: Storage tier to retrieve from

        Returns:
            Backup data or None if not found
        """
        if tier == StorageTier.LOCAL:
            return await self._retrieve_local(backup_id)

        elif tier == StorageTier.NETWORK:
            return await self._retrieve_network(backup_id)

        elif tier in [StorageTier.CLOUD, StorageTier.ARCHIVE]:
            return await self._retrieve_s3(backup_id, tier)

        return None

    async def _retrieve_local(self, backup_id: str) -> bytes | None:
        """Retrieve backup from local storage."""
        primary_path = self.backup_path / f"{backup_id}.backup"

        if primary_path.exists():
            with open(primary_path, "rb") as f:
                return f.read()

        logger.warning(f"Local backup {backup_id} not found")
        return None

    async def _retrieve_network(self, backup_id: str) -> bytes | None:
        """Retrieve backup from network storage."""
        if not self.network_path:
            return None

        file_path = self.network_path / f"{backup_id}.backup"

        if file_path.exists():
            with open(file_path, "rb") as f:
                return f.read()

        return None

    async def _retrieve_s3(self, backup_id: str, tier: StorageTier) -> bytes | None:
        """Retrieve backup from S3."""
        if not self._s3_client:
            return None

        try:
            prefix = "archive" if tier == StorageTier.ARCHIVE else "backups"
            key = f"{prefix}/{backup_id}.backup"

            response = self._s3_client.get_object(Bucket=self.s3_bucket, Key=key)
            return response["Body"].read()

        except Exception as e:
            logger.error(f"S3 retrieval failed: {e}")
            return None

    async def delete(self, backup_id: str, tier: StorageTier) -> bool:
        """
        Delete backup from storage.

        Args:
            backup_id: Unique backup identifier
            tier: Storage tier

        Returns:
            True if successful
        """
        try:
            if tier == StorageTier.LOCAL:
                paths = [
                    self.local_path / f"{backup_id}.backup",
                    self.backup_path / f"{backup_id}.backup",
                ]
                for file_path in paths:
                    if file_path.exists():
                        file_path.unlink()

            elif tier == StorageTier.NETWORK:
                if self.network_path:
                    file_path = self.network_path / f"{backup_id}.backup"
                    if file_path.exists():
                        file_path.unlink()

            elif tier in [StorageTier.CLOUD, StorageTier.ARCHIVE]:
                if self._s3_client:
                    prefix = "archive" if tier == StorageTier.ARCHIVE else "backups"
                    key = f"{prefix}/{backup_id}.backup"
                    self._s3_client.delete_object(Bucket=self.s3_bucket, Key=key)

            return True

        except Exception as e:
            logger.error(f"Failed to delete backup: {e}")
            return False

    def upload_to_s3(self, backup_id: str) -> None:
        """
        Upload existing local backup to S3.

        Args:
            backup_id: Backup identifier
        """
        if not self._s3_client or not self.s3_bucket:
            return

        local_candidates = [
            self.backup_path / f"{backup_id}.backup",
            self.local_path / f"{backup_id}.backup",
        ]

        source_path = next((path for path in local_candidates if path.exists()), None)
        if source_path is None:
            logger.warning(f"No local artifact found for backup {backup_id}")
            return

        try:
            self._s3_client.upload_file(
                str(source_path), self.s3_bucket, f"{backup_id}.backup"
            )
            logger.info(f"Uploaded {backup_id} to S3")

        except Exception as exc:
            logger.error(f"Failed to upload {backup_id} to S3: {exc}")

    @property
    def has_network_storage(self) -> bool:
        """Check if network storage is available."""
        return self.network_path is not None

    @property
    def has_cloud_storage(self) -> bool:
        """Check if cloud storage is available."""
        return self._s3_client is not None
