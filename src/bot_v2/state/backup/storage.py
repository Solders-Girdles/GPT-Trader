"""Storage backends for backup data."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .models import BackupConfig, BackupMetadata, StorageTier

try:  # optional dependency for cloud storage
    import boto3

    S3_AVAILABLE = True
except ImportError:  # pragma: no cover - cloud storage optional
    boto3 = None
    S3_AVAILABLE = False

logger = logging.getLogger(__name__)


class BackupStorage:
    """Handles persistence of backup payloads to different storage tiers."""

    def __init__(self, config: BackupConfig) -> None:
        self.config = config
        self._init_paths()
        self.s3_client = self._init_s3()

    def _init_paths(self) -> None:
        Path(self.config.backup_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.local_storage_path).mkdir(parents=True, exist_ok=True)
        network_root = self.config.network_storage_path
        if network_root:
            Path(network_root).mkdir(parents=True, exist_ok=True)

    def _init_s3(self) -> Any | None:  # pragma: no cover - exercised in integration
        if not (S3_AVAILABLE and self.config.s3_bucket):
            return None
        try:
            client = boto3.client("s3", region_name=self.config.s3_region)
            client.head_bucket(Bucket=self.config.s3_bucket)
            logger.info("S3 backup initialized with bucket %s", self.config.s3_bucket)
            return client
        except Exception as exc:  # pragma: no cover - depends on AWS credentials
            logger.warning("S3 initialization failed: %s", exc)
            return None

    async def store(self, backup_id: str, data: bytes, tier: StorageTier) -> str:
        if tier == StorageTier.LOCAL:
            return self._store_local(backup_id, data)
        if tier == StorageTier.NETWORK and self.config.network_storage_path:
            return self._store_network(backup_id, data)
        if tier == StorageTier.CLOUD and self.s3_client:
            return await self._store_s3(backup_id, data, "backups")
        if tier == StorageTier.ARCHIVE and self.s3_client:
            return await self._store_s3(backup_id, data, "archive", storage_class="GLACIER")
        # fallback
        return self._store_local(backup_id, data)

    async def retrieve(self, metadata: BackupMetadata) -> bytes | None:
        tier = metadata.storage_tier
        if tier == StorageTier.LOCAL:
            return self._read_local(self.config.local_storage_path, metadata.backup_id)
        if tier == StorageTier.NETWORK and self.config.network_storage_path:
            return self._read_local(self.config.network_storage_path, metadata.backup_id)
        if tier in (StorageTier.CLOUD, StorageTier.ARCHIVE) and self.s3_client:
            prefix = "archive" if tier == StorageTier.ARCHIVE else "backups"
            return await self._read_s3(metadata.backup_id, prefix)
        return None

    async def delete(self, metadata: BackupMetadata) -> bool:
        tier = metadata.storage_tier
        if tier == StorageTier.LOCAL:
            return self._delete_local(self.config.local_storage_path, metadata.backup_id)
        if tier == StorageTier.NETWORK and self.config.network_storage_path:
            return self._delete_local(self.config.network_storage_path, metadata.backup_id)
        if tier in (StorageTier.CLOUD, StorageTier.ARCHIVE) and self.s3_client:
            prefix = "archive" if tier == StorageTier.ARCHIVE else "backups"
            key = f"{prefix}/{metadata.backup_id}.backup"
            try:
                self.s3_client.delete_object(Bucket=self.config.s3_bucket, Key=key)
                return True
            except Exception as exc:  # pragma: no cover - network behavior
                logger.error("Failed to delete S3 object %s: %s", key, exc)
                return False
        return False

    def _store_local(self, backup_id: str, data: bytes) -> str:
        file_path = Path(self.config.local_storage_path) / f"{backup_id}.backup"
        with open(file_path, "wb") as handle:
            handle.write(data)
        return str(file_path)

    def _store_network(self, backup_id: str, data: bytes) -> str:
        network_root = self.config.network_storage_path
        if network_root is None:
            raise ValueError("Network storage path not configured")
        file_path = Path(network_root) / f"{backup_id}.backup"
        with open(file_path, "wb") as handle:
            handle.write(data)
        return str(file_path)

    async def _store_s3(
        self,
        backup_id: str,
        data: bytes,
        prefix: str,
        *,
        storage_class: str = "STANDARD_IA",
    ) -> str:
        client = self.s3_client
        if client is None:
            return self._store_local(backup_id, data)
        key = f"{prefix}/{backup_id}.backup"
        try:
            client.put_object(
                Bucket=self.config.s3_bucket,
                Key=key,
                Body=data,
                StorageClass=storage_class,
            )
            return f"s3://{self.config.s3_bucket}/{key}"
        except Exception as exc:  # pragma: no cover - depends on AWS
            logger.error("S3 storage failed: %s", exc)
            return self._store_local(backup_id, data)

    def _read_local(self, root: str | Path, backup_id: str) -> bytes | None:
        file_path = Path(root) / f"{backup_id}.backup"
        if not file_path.exists():
            return None
        with open(file_path, "rb") as handle:
            return handle.read()

    async def _read_s3(self, backup_id: str, prefix: str) -> bytes | None:
        client = self.s3_client
        if client is None:
            return None
        key = f"{prefix}/{backup_id}.backup"
        try:
            response = client.get_object(Bucket=self.config.s3_bucket, Key=key)
            body = response.get("Body")
            if body is None:
                return None
            data = body.read()
            return data if isinstance(data, bytes) else bytes(data)
        except Exception as exc:  # pragma: no cover - network behavior
            logger.error("S3 retrieval failed: %s", exc)
            return None

    def _delete_local(self, root: str | Path, backup_id: str) -> bool:
        file_path = Path(root) / f"{backup_id}.backup"
        try:
            if file_path.exists():
                file_path.unlink()
            return True
        except Exception as exc:  # pragma: no cover - filesystem issues
            logger.error("Failed to delete backup file %s: %s", file_path, exc)
            return False
