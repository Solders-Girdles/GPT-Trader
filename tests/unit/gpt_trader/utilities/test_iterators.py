"""Tests for iterator utilities."""

from __future__ import annotations

from collections.abc import Iterator

from gpt_trader.utilities.iterators import empty_stream


class TestEmptyStream:
    """Tests for empty_stream function."""

    def test_returns_iterator(self) -> None:
        result = empty_stream()
        assert isinstance(result, Iterator)

    def test_yields_nothing(self) -> None:
        result = list(empty_stream())
        assert result == []

    def test_is_exhausted_immediately(self) -> None:
        stream = empty_stream()
        items = []
        for item in stream:
            items.append(item)
        assert items == []

    def test_multiple_calls_independent(self) -> None:
        stream1 = empty_stream()
        stream2 = empty_stream()
        assert list(stream1) == []
        assert list(stream2) == []
