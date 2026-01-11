from __future__ import annotations

from pathlib import Path

from gpt_trader.persistence.event_store import EventStore


def test_get_recent_limits_and_zero() -> None:
    store = EventStore()
    for i in range(5):
        store.append("evt", {"i": i})

    assert [e["data"]["i"] for e in store.get_recent(2)] == [3, 4]
    assert store.get_recent(0) == []


def test_cache_metrics_zero_size_discards() -> None:
    store = EventStore(max_cache_size=0)
    store.append("evt", {"i": 1})

    assert store.get_cache_size() == 0
    assert store.get_cache_fill_ratio() == 0.0


def test_store_wraps_non_dict_payload() -> None:
    store = EventStore()
    store.store("payload")

    assert store.events == [{"type": "event", "data": {"payload": "payload"}}]


def test_append_error_prefers_message_and_context() -> None:
    store = EventStore()
    store.append_error(
        error="fallback",
        details={"ignored": True},
        bot_id="bot-1",
        message="use-this",
        context={"ok": True},
    )

    event = store.events[0]
    assert event["data"]["error"] == "use-this"
    assert event["data"]["details"] == {"ok": True}


def test_persistent_mode_extracts_nested_bot_id(tmp_path: Path) -> None:
    store = EventStore(root=tmp_path)
    store.append("position", {"position": {"bot_id": "bot-xyz"}})

    assert store._database is not None
    events = store._database.read_events_by_bot("bot-xyz")

    assert len(events) == 1
    assert events[0]["type"] == "position"
    store.close()


def test_get_recent_by_type_in_memory() -> None:
    store = EventStore()
    store.append("price_tick", {"i": 1})
    store.append("error", {"i": 2})
    store.append("price_tick", {"i": 3})

    recent = store.get_recent_by_type("price_tick", count=2)

    assert [e["data"]["i"] for e in recent] == [1, 3]


def test_get_recent_by_type_persistent(tmp_path: Path) -> None:
    store = EventStore(root=tmp_path)
    store.append("price_tick", {"i": 1})
    store.append("metric", {"i": 2})
    store.append("price_tick", {"i": 3})
    store.close()

    reopened = EventStore(root=tmp_path)
    recent = reopened.get_recent_by_type("price_tick", count=1)
    assert [e["data"]["i"] for e in recent] == [3]
    reopened.close()
