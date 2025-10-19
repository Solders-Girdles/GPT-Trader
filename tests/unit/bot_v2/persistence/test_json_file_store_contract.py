"""Contract tests for JsonFileStore persistence layer."""

from __future__ import annotations

import json
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pytest

from bot_v2.persistence.json_file_store import JsonFileStore


@dataclass
class TestDataClass:
    """Test dataclass for serialization."""

    name: str
    value: int
    timestamp: datetime


class TestJsonFileStoreContract:
    """Test JsonFileStore contract with comprehensive failure and locking scenarios."""

    @pytest.fixture
    def temp_file(self):
        """Create a temporary file for testing."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as f:
            path = Path(f.name)
        yield path
        path.unlink(missing_ok=True)

    @pytest.fixture
    def temp_jsonl_file(self):
        """Create a temporary JSONL file for testing."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as f:
            path = Path(f.name)
        yield path
        path.unlink(missing_ok=True)

    @pytest.fixture
    def store(self, temp_file):
        """Create a JsonFileStore instance."""
        return JsonFileStore(temp_file, create=True)

    def test_initialization_creates_parent_directories(self, tmp_path):
        """Test that initialization creates parent directories."""
        nested_path = tmp_path / "nested" / "deep" / "file.json"
        _ = JsonFileStore(nested_path, create=True)

        assert nested_path.parent.exists()
        assert nested_path.exists()

    def test_initialization_without_create_does_not_touch_file(self, temp_file):
        """Test initialization without create flag doesn't create file."""
        temp_file.unlink(missing_ok=True)
        _ = JsonFileStore(temp_file, create=False)

        assert not temp_file.exists()

    def test_write_json_atomic_operation(self, store, temp_file):
        """Test that write_json is atomic and handles concurrent access."""
        test_data = {"key": "value", "number": 42}

        store.write_json(test_data)

        # Verify file contains correct JSON
        with temp_file.open("r") as f:
            content = json.load(f)
            assert content == test_data

        # Verify trailing newline for indented JSON
        with temp_file.open("r") as f:
            text = f.read()
            assert text.endswith("\n")

    def test_write_json_without_indent(self, store, temp_file):
        """Test write_json without indentation."""
        test_data = {"compact": True}

        store.write_json(test_data, indent=None)

        with temp_file.open("r") as f:
            content = f.read()
            # Should be compact JSON without trailing newline
            assert content == '{"compact": true}'

    def test_read_json_returns_default_on_missing_file(self, tmp_path):
        """Test read_json returns default when file doesn't exist."""
        missing_file = tmp_path / "missing.json"
        store = JsonFileStore(missing_file, create=False)

        result = store.read_json(default="default_value")
        assert result == "default_value"

    def test_read_json_returns_default_on_corrupt_json(self, store, temp_file):
        """Test read_json returns default on JSON decode errors."""
        # Write invalid JSON
        temp_file.write_text("{invalid json content")

        result = store.read_json(default="fallback")
        assert result == "fallback"

    def test_read_json_successful_parsing(self, store, temp_file):
        """Test successful JSON reading."""
        test_data = {"success": True, "nested": {"key": "value"}}
        temp_file.write_text(json.dumps(test_data))

        result = store.read_json()
        assert result == test_data

    def test_append_jsonl_thread_safety(self, store, temp_file):
        """Test that append_jsonl is thread-safe."""
        results = []

        def append_worker(worker_id: int):
            for i in range(10):
                store.append_jsonl(
                    {"worker": worker_id, "sequence": i, "timestamp": datetime.now().isoformat()}
                )
            results.append(f"worker_{worker_id}_done")

        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=append_worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Verify all operations completed
        assert len(results) == 3
        assert all("done" in r for r in results)

        # Verify all records were written
        records = list(store.iter_jsonl())
        assert len(records) == 30  # 3 workers * 10 records each

        # Verify no corruption (all records are valid JSON)
        for record in records:
            assert "worker" in record
            assert "sequence" in record
            assert "timestamp" in record

    def test_iter_jsonl_handles_missing_file(self, tmp_path):
        """Test iter_jsonl returns empty iterator for missing files."""
        missing_file = tmp_path / "missing.jsonl"
        store = JsonFileStore(missing_file, create=False)

        records = list(store.iter_jsonl())
        assert records == []

    def test_iter_jsonl_skips_malformed_lines(self, store, temp_file):
        """Test iter_jsonl skips malformed JSON lines."""
        # Write mixed valid and invalid JSON lines
        content = """{"valid": true}
invalid json line
{"also_valid": 123}
{"incomplete": }
more invalid content
{"final": "record"}
"""
        temp_file.write_text(content)

        records = list(store.iter_jsonl())

        # Should only return valid records
        assert len(records) == 3
        assert records[0] == {"valid": True}
        assert records[1] == {"also_valid": 123}
        assert records[2] == {"final": "record"}

    def test_iter_jsonl_skips_empty_lines(self, store, temp_file):
        """Test iter_jsonl skips empty and whitespace-only lines."""
        content = """{"first": 1}

{"second": 2}

{"third": 3}
"""
        temp_file.write_text(content)

        records = list(store.iter_jsonl())
        assert len(records) == 3
        assert records[0]["first"] == 1
        assert records[1]["second"] == 2
        assert records[2]["third"] == 3

    def test_iter_jsonl_filters_non_dict_records(self, store, temp_file):
        """Test iter_jsonl only yields dict records."""
        content = """["array", "not", "dict"]
"string record"
42
{"valid_dict": true}
null
"""
        temp_file.write_text(content)

        records = list(store.iter_jsonl())
        assert len(records) == 1
        assert records[0] == {"valid_dict": True}

    def test_replace_jsonl_atomic_replacement(self, store, temp_file):
        """Test replace_jsonl atomically replaces all content."""
        # First add some records
        store.append_jsonl({"original": 1})
        store.append_jsonl({"original": 2})

        # Replace with new records
        new_records = [{"new": "a"}, {"new": "b"}, {"new": "c"}]
        store.replace_jsonl(new_records)

        # Verify only new records exist
        records = list(store.iter_jsonl())
        assert len(records) == 3
        assert all("new" in r for r in records)
        assert all("original" not in r for r in records)

    def test_default_serializer_handles_dataclasses(self):
        """Test _default_serializer handles dataclass instances."""
        test_obj = TestDataClass("test", 42, datetime(2023, 1, 1, 12, 0, 0))

        result = JsonFileStore._default_serializer(test_obj)

        expected = {"name": "test", "value": 42, "timestamp": "2023-01-01T12:00:00"}
        assert result == expected

    def test_default_serializer_handles_datetime(self):
        """Test _default_serializer handles datetime objects."""
        dt = datetime(2023, 12, 25, 15, 30, 45, 123456)

        result = JsonFileStore._default_serializer(dt)
        assert result == "2023-12-25T15:30:45.123456"

    def test_default_serializer_handles_unknown_types(self):
        """Test _default_serializer converts unknown types to strings."""

        class CustomClass:
            def __str__(self):
                return "custom_string"

        obj = CustomClass()
        result = JsonFileStore._default_serializer(obj)
        assert result == "custom_string"

    def test_lock_reentrancy(self, store):
        """Test that the lock is re-entrant (RLock behavior)."""
        # This should not deadlock
        with store._lock:
            with store._lock:
                store.write_json({"nested": "locks"})
                result = store.read_json()
                assert result == {"nested": "locks"}

    def test_concurrent_read_write_operations(self, store, temp_file):
        """Test concurrent read/write operations don't corrupt data."""
        results = []
        errors = []

        def writer_worker(worker_id: int):
            try:
                for i in range(20):
                    store.write_json(
                        {"writer": worker_id, "write_count": i, "data": f"content_{worker_id}_{i}"}
                    )
                    time.sleep(0.001)  # Small delay to encourage interleaving
                results.append(f"writer_{worker_id}_success")
            except Exception as e:
                errors.append(f"writer_{worker_id}_error: {e}")

        def reader_worker(worker_id: int):
            try:
                for i in range(10):
                    data = store.read_json(default={})
                    # Just verify we can read something
                    assert isinstance(data, dict)
                    time.sleep(0.001)
                results.append(f"reader_{worker_id}_success")
            except Exception as e:
                errors.append(f"reader_{worker_id}_error: {e}")

        # Start concurrent operations
        threads = []
        for i in range(2):
            threads.append(threading.Thread(target=writer_worker, args=(i,)))
            threads.append(threading.Thread(target=reader_worker, args=(i,)))

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join(timeout=10)

        # Verify no errors occurred
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 4, f"Expected 4 successful operations, got {len(results)}"

    def test_large_jsonl_file_handling(self, store, temp_file):
        """Test handling of larger JSONL files."""
        # Create a file with many records
        large_records = [{"id": i, "data": f"value_{i}"} for i in range(1000)]

        for record in large_records:
            store.append_jsonl(record)

        # Read back and verify
        read_records = list(store.iter_jsonl())
        assert len(read_records) == 1000

        # Verify first and last records
        assert read_records[0] == {"id": 0, "data": "value_0"}
        assert read_records[-1] == {"id": 999, "data": "value_999"}

    def test_jsonl_with_complex_nested_data(self, store, temp_file):
        """Test JSONL with complex nested structures."""
        complex_record = {
            "metadata": {
                "version": "1.0",
                "timestamp": datetime.now().isoformat(),
                "tags": ["test", "complex"],
            },
            "data": {"nested": {"deeply": {"embedded": [1, 2, {"key": "value"}]}}},
            "primitives": {"string": "test", "number": 42, "boolean": True, "null": None},
        }

        store.append_jsonl(complex_record)

        records = list(store.iter_jsonl())
        assert len(records) == 1

        read_record = records[0]
        assert read_record["metadata"]["version"] == "1.0"
        assert read_record["data"]["nested"]["deeply"]["embedded"][2]["key"] == "value"
        assert read_record["primitives"]["number"] == 42
        assert read_record["primitives"]["boolean"] is True
        assert read_record["primitives"]["null"] is None

    def test_file_permission_error_handling(self, store, temp_file):
        """Test graceful handling of file permission errors."""
        # Make file unreadable
        temp_file.chmod(0o000)

        try:
            # Should handle permission error gracefully
            result = store.read_json(default="permission_denied")
            assert result == "permission_denied"

            # Should handle write permission error
            with pytest.raises(OSError):
                store.write_json({"test": "data"})
        finally:
            # Restore permissions for cleanup
            temp_file.chmod(0o644)

    def test_empty_jsonl_file_handling(self, store, temp_file):
        """Test handling of completely empty JSONL files."""
        # Create empty file
        temp_file.touch()

        records = list(store.iter_jsonl())
        assert records == []

    def test_mixed_encoding_content(self, store, temp_file):
        """Test handling of files with mixed encoding scenarios."""
        # Write valid UTF-8 content
        temp_file.write_text('{"message": "café", "emoji": "🚀"}\n', encoding="utf-8")

        records = list(store.iter_jsonl())
        assert len(records) == 1
        assert records[0]["message"] == "café"
        assert records[0]["emoji"] == "🚀"
