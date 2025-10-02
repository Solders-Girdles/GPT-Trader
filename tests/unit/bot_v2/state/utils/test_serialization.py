
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum


from bot_v2.state.utils.serialization import (
    SerializableMixin,
    calculate_data_hash,
    compress_data,
    decompress_data,
    deserialize_datetime,
    deserialize_from_json,
    prepare_compressed_payload,
    serialize_datetime,
    serialize_decimal,
    serialize_enum,
    serialize_to_json,
)


class Status(Enum):
    READY = "ready"
    ERROR = "error"


@dataclass
class Step(SerializableMixin):
    name: str


@dataclass
class SampleRecord(SerializableMixin):
    identifier: str
    created_at: datetime
    balance: Decimal
    metadata: dict[str, datetime]
    steps: list[Step] = field(default_factory=list)
    _internal: str = "hidden"


def test_datetime_serialization_round_trip():
    dt = datetime(2024, 5, 3, 12, 34, 56, tzinfo=timezone.utc)
    serialized = serialize_datetime(dt)
    assert deserialize_datetime(serialized) == dt


def test_decimal_and_enum_serialization():
    value = Decimal("120.500")
    assert serialize_decimal(value) == "120.500"
    assert serialize_enum(Status.READY) == "ready"


def test_calculate_data_hash_is_deterministic():
    payload_a = {"b": [1, 2, 3], "a": "value"}
    payload_b = {"a": "value", "b": [1, 2, 3]}
    assert calculate_data_hash(payload_a) == calculate_data_hash(payload_b)


def test_compress_and_decompress_round_trip():
    original = b"critical trading state snapshot"
    compressed = compress_data(original, compression_level=9)
    assert decompress_data(compressed) == original


def test_serializable_mixin_handles_nested_and_private_fields():
    now = datetime(2024, 6, 1, 8, 15, tzinfo=timezone.utc)
    record = SampleRecord(
        identifier="acct-1",
        created_at=now,
        balance=Decimal("1500.25"),
        metadata={"last_sync": now},
        steps=[Step("bootstrap"), Step("rebalance")],
    )

    as_dict = record.to_dict()
    assert as_dict["created_at"] == now.isoformat()
    assert as_dict["balance"] == "1500.25"
    assert as_dict["metadata"]["last_sync"] == now.isoformat()
    assert as_dict["steps"] == [{"name": "bootstrap"}, {"name": "rebalance"}]
    assert "_internal" not in as_dict

    restored = SampleRecord.from_dict(as_dict)
    assert restored.identifier == record.identifier
    assert restored.created_at == now
    assert restored.metadata["last_sync"] == now.isoformat()
    assert restored.steps == as_dict["steps"]


def test_json_serialization_helpers():
    payload = {"status": Status.ERROR, "amount": Decimal("1.5")}
    json_bytes = serialize_to_json(payload)
    decoded = deserialize_from_json(json_bytes)

    assert decoded == {"amount": "1.5", "status": "Status.ERROR"}


def test_prepare_compressed_payload_toggle():
    payload = {"symbol": "BTC-PERP", "status": "active"}

    compressed, original_size, compressed_size = prepare_compressed_payload(
        payload, enable_compression=True, compression_level=9
    )
    assert decompress_data(compressed) == serialize_to_json(payload)
    assert original_size == len(serialize_to_json(payload))

    raw, raw_original_size, raw_compressed_size = prepare_compressed_payload(
        payload, enable_compression=False
    )
    assert raw == serialize_to_json(payload)
    assert raw_original_size == raw_compressed_size
