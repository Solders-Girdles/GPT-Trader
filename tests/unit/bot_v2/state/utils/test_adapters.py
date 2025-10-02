"""Unit tests for storage adapter implementations."""

from __future__ import annotations

import builtins
import sys
from types import ModuleType
from typing import Any

import pytest

from bot_v2.state.utils.adapters import (
    DefaultPostgresAdapter,
    DefaultRedisAdapter,
    DefaultS3Adapter,
    PostgresAdapter,
    RedisAdapter,
    S3Adapter,
)


def test_redis_adapter_base_methods_are_noops() -> None:
    """Calling abstract base methods should not raise and improves coverage."""
    RedisAdapter.ping(object())
    RedisAdapter.get(object(), "key")
    RedisAdapter.setex(object(), "key", 1, "value")
    RedisAdapter.delete(object(), "key")
    RedisAdapter.keys(object(), "*")
    RedisAdapter.dbsize(object())
    RedisAdapter.close(object())


def test_postgres_adapter_base_methods_are_noops() -> None:
    PostgresAdapter.execute(object(), "SELECT 1")
    PostgresAdapter.commit(object())
    PostgresAdapter.rollback(object())
    PostgresAdapter.close(object())


def test_s3_adapter_base_methods_are_noops() -> None:
    S3Adapter.head_bucket(object(), "bucket")
    S3Adapter.get_object(object(), "bucket", "key")
    S3Adapter.put_object(object(), "bucket", "key", b"data")
    S3Adapter.delete_object(object(), "bucket", "key")
    S3Adapter.list_objects_v2(object(), "bucket")


def _install_fake_redis(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    instances: list[Any] = []

    class FakeRedisClient:
        def __init__(self, **_kwargs):
            self.store: dict[str, str] = {}
            self.closed = False
            instances.append(self)

        def ping(self) -> None:
            return None

        def get(self, key: str) -> str | None:
            return self.store.get(key)

        def setex(self, key: str, ttl_seconds: int, value: str) -> None:  # noqa: ARG002
            self.store[key] = value

        def delete(self, key: str) -> int:
            return 1 if self.store.pop(key, None) is not None else 0

        def keys(self, pattern: str) -> list[str]:  # noqa: ARG002
            return sorted(self.store.keys())

        def dbsize(self) -> int:
            return len(self.store)

        def close(self) -> None:
            self.closed = True

    module = ModuleType("redis")
    module.Redis = FakeRedisClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "redis", module)
    return instances


def _install_fake_psycopg2(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    cursors: list[Any] = []

    class FakeCursor:
        def __init__(self) -> None:
            self._result: list[dict[str, Any]] = []
            cursors.append(self)

        def __enter__(self) -> "FakeCursor":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001, D401
            return None

        def execute(self, query: str, params: tuple = ()) -> None:
            normalized = query.strip().upper()
            if normalized.startswith("SELECT"):
                self._result = [{"result": 1, "params": params}]
            else:
                self._result = []

        def fetchall(self) -> list[dict[str, Any]]:
            return self._result

    class FakeConnection:
        def __init__(self) -> None:
            self.commits = 0
            self.rollbacks = 0
            self.closed = False

        def cursor(self) -> FakeCursor:
            return FakeCursor()

        def commit(self) -> None:
            self.commits += 1

        def rollback(self) -> None:
            self.rollbacks += 1

        def close(self) -> None:
            self.closed = True

    psycopg2_module = ModuleType("psycopg2")
    extras_module = ModuleType("psycopg2.extras")
    psycopg2_module.connect = lambda **_kwargs: FakeConnection()  # type: ignore[attr-defined]
    extras_module.RealDictCursor = object()
    monkeypatch.setitem(sys.modules, "psycopg2", psycopg2_module)
    monkeypatch.setitem(sys.modules, "psycopg2.extras", extras_module)
    return cursors


def _install_fake_boto3(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    clients: list[Any] = []

    class FakeS3Client:
        def __init__(self) -> None:
            self.storage: dict[tuple[str, str], bytes] = {}
            self.closed = False
            clients.append(self)

        def head_bucket(self, Bucket: str) -> dict[str, str]:  # noqa: N803
            return {"Bucket": Bucket}

        def get_object(self, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803
            body = self.storage.get((Bucket, Key), b"")
            return {"Body": body}

        def put_object(
            self,
            Bucket: str,
            Key: str,
            Body: bytes,
            StorageClass: str = "STANDARD",
            Metadata: dict[str, str] | None = None,
        ) -> dict[str, Any]:  # noqa: N803
            self.storage[(Bucket, Key)] = Body
            return {
                "StorageClass": StorageClass,
                "Metadata": Metadata or {},
            }

        def delete_object(self, Bucket: str, Key: str) -> dict[str, bool]:  # noqa: N803
            self.storage.pop((Bucket, Key), None)
            return {"Deleted": True}

        def list_objects_v2(self, Bucket: str, Prefix: str = "") -> dict[str, Any]:  # noqa: N803
            contents = [
                {"Key": key}
                for (bucket, key), _value in self.storage.items()
                if bucket == Bucket and key.startswith(Prefix)
            ]
            return {"Contents": contents}

    module = ModuleType("boto3")
    module.client = lambda service, region_name=None: FakeS3Client()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "boto3", module)
    return clients


def test_default_redis_adapter_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    instances = _install_fake_redis(monkeypatch)

    adapter = DefaultRedisAdapter("localhost", 6379, 0)
    assert adapter.ping() is True

    assert adapter.setex("key", 60, "value") is True
    assert adapter.get("key") == "value"
    assert adapter.keys("*") == ["key"]
    assert adapter.dbsize() == 1
    assert adapter.delete("key") == 1

    adapter.close()
    assert instances[0].closed is True


def test_default_redis_adapter_handles_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_redis(monkeypatch)
    adapter = DefaultRedisAdapter("localhost", 6379, 0)

    def raise_runtime(*_args, **_kwargs):
        raise RuntimeError("boom")

    adapter._client.ping = raise_runtime
    assert adapter.ping() is False

    adapter._client.setex = raise_runtime
    assert adapter.setex("key", 1, "value") is False

    adapter._client = None
    assert adapter.ping() is False
    assert adapter.get("key") is None
    assert adapter.setex("key", 1, "value") is False
    assert adapter.delete("key") == 0
    assert adapter.keys("*") == []
    assert adapter.dbsize() == 0
    adapter.close()  # Should be a no-op when client is None


def test_default_redis_adapter_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "redis", raising=False)
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "redis":
            raise ImportError("redis missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="Redis library not available"):
        DefaultRedisAdapter("localhost", 6379, 0)


def test_default_postgres_adapter_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    cursors = _install_fake_psycopg2(monkeypatch)

    adapter = DefaultPostgresAdapter("localhost", 5432, "db", "user", "pass")

    select_result = adapter.execute("SELECT * FROM table", (1,))
    assert select_result == [{"result": 1, "params": (1,)}]

    update_result = adapter.execute("UPDATE table SET value=1", ())
    assert update_result == []

    adapter.commit()
    adapter.rollback()
    adapter.close()

    assert cursors, "Cursor should have been created"


def test_default_postgres_adapter_handles_absent_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_psycopg2(monkeypatch)
    adapter = DefaultPostgresAdapter("localhost", 5432, "db", "user", "pass")

    adapter._conn = None
    assert adapter.execute("SELECT 1") == []
    adapter.commit()
    adapter.rollback()
    adapter.close()


def test_default_postgres_adapter_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "psycopg2", raising=False)
    monkeypatch.delitem(sys.modules, "psycopg2.extras", raising=False)
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("psycopg2"):
            raise ImportError("psycopg2 missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="PostgreSQL library not available"):
        DefaultPostgresAdapter("localhost", 5432, "db", "user", "pass")


def test_default_s3_adapter_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    clients = _install_fake_boto3(monkeypatch)

    adapter = DefaultS3Adapter("us-east-1")

    assert adapter.head_bucket("bucket") == {"Bucket": "bucket"}
    adapter.put_object("bucket", "key", b"value", metadata={"meta": "1"})
    assert adapter.get_object("bucket", "key")["Body"] == b"value"
    listing = adapter.list_objects_v2("bucket")
    assert listing == {"Contents": [{"Key": "key"}]}
    assert adapter.delete_object("bucket", "key") == {"Deleted": True}

    assert clients, "S3 client should have been created"


def test_default_s3_adapter_handles_missing_client(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_boto3(monkeypatch)
    adapter = DefaultS3Adapter("us-east-1")
    adapter._client = None

    with pytest.raises(Exception, match="S3 client not available"):
        adapter.head_bucket("bucket")

    with pytest.raises(Exception, match="S3 client not available"):
        adapter.get_object("bucket", "key")

    with pytest.raises(Exception, match="S3 client not available"):
        adapter.put_object("bucket", "key", b"value")

    with pytest.raises(Exception, match="S3 client not available"):
        adapter.delete_object("bucket", "key")

    with pytest.raises(Exception, match="S3 client not available"):
        adapter.list_objects_v2("bucket")


def test_default_s3_adapter_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "boto3", raising=False)
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "boto3":
            raise ImportError("boto3 missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="S3 library not available"):
        DefaultS3Adapter("us-east-1")
