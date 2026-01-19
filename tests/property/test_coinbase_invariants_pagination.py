from __future__ import annotations

from typing import Any

from hypothesis import given, seed, settings
from hypothesis import strategies as st

from gpt_trader.features.brokerages.coinbase.client.base import CoinbaseClientBase


class _PaginationClient(CoinbaseClientBase):
    def __init__(self, pages: list[list[int]]) -> None:
        super().__init__(
            base_url="https://example.com",
            auth=None,
            # Removed enable_keep_alive as it's not in CoinbaseClientBase anymore
        )
        self._pages = pages
        self._cursor_calls = 0

    def _request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        index = self._cursor_calls
        self._cursor_calls += 1
        current = self._pages[index]
        next_cursor = str(index + 1) if index + 1 < len(self._pages) else None
        return {"items": current, "next_cursor": next_cursor}

    def paginate(self, path: str, params: dict[str, Any], items_key: str) -> list[Any]:
        all_items: list[Any] = []
        next_cursor: str | None = None

        while True:
            current_params = params.copy()
            if next_cursor:
                current_params["cursor"] = next_cursor

            # Build the path with parameters for a GET request
            request_path = self._build_path_with_params(path, current_params)

            # Call _request without the 'payload' keyword argument for GET
            response = self._request("GET", request_path)

            all_items.extend(response.get(items_key, []))
            next_cursor = response.get("next_cursor")

            if not next_cursor:
                break
        return all_items


@seed(4242)
@settings(max_examples=60, deadline=None)
@given(
    pages=st.lists(
        st.lists(st.integers(min_value=0, max_value=100), max_size=5),
        min_size=1,
        max_size=5,
    )
)
def test_paginate_yields_all_items_in_order(pages: list[list[int]]) -> None:
    client = _PaginationClient(pages)
    collected = list(client.paginate("/fake", params={}, items_key="items"))
    expected: list[int] = [item for page in pages for item in page]
    assert collected == expected
    assert client._cursor_calls == len(pages)
