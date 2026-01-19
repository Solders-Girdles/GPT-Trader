"""Tests for CoinbaseClientBase.paginate."""

from unittest.mock import Mock

from gpt_trader.features.brokerages.coinbase.auth import SimpleAuth
from gpt_trader.features.brokerages.coinbase.client.base import CoinbaseClientBase


class TestCoinbaseClientBasePagination:
    """Test CoinbaseClientBase pagination helper."""

    def setup_method(self) -> None:
        self.base_url = "https://api.coinbase.com"
        self.auth = Mock(spec=SimpleAuth)
        self.auth.get_headers.return_value = {"Authorization": "Bearer test-token"}

    def test_paginate_success(self) -> None:
        """Test successful pagination."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)

        # Mock responses for multiple pages
        page1 = {"data": [{"id": 1}, {"id": 2}], "cursor": "page2"}
        page2 = {"data": [{"id": 3}], "cursor": None}

        client._request = Mock(side_effect=[page1, page2])

        results = list(client.paginate("/api/v3/test", {}, "data"))

        assert len(results) == 3
        assert results == [{"id": 1}, {"id": 2}, {"id": 3}]
        assert client._request.call_count == 2

        # Check that cursor was passed in second request
        second_call_args = client._request.call_args_list[1]
        assert "cursor=page2" in second_call_args[0][1]

    def test_paginate_custom_cursor_params(self) -> None:
        """Test pagination with custom cursor parameters."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)

        pages = [
            {"items": [{"id": 1}], "next_cursor": "next"},
            {"items": [], "next_cursor": None},
        ]
        client._request = Mock(side_effect=pages)

        list(
            client.paginate(
                "/api/v3/test",
                {"limit": "100"},
                "items",
                cursor_param="page_token",
                cursor_field="next_cursor",
            )
        )

        # Check that custom cursor param was used
        call_args = client._request.call_args
        assert "page_token=next" in call_args[0][1]

    def test_paginate_with_pagination_object(self) -> None:
        """Test pagination with nested pagination cursor."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)

        page1 = {"items": [{"id": 1}], "pagination": {"next_cursor": "next"}}
        page2 = {"items": [], "pagination": {"next_cursor": None}}
        client._request = Mock(side_effect=[page1, page2])

        list(client.paginate("/api/v3/test", {}, "items"))

        second_call_args = client._request.call_args_list[1]
        assert "cursor=next" in second_call_args[0][1]

    def test_paginate_no_cursor(self) -> None:
        """Test pagination when no cursor is returned."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)

        page = {"data": [{"id": 1}]}
        client._request = Mock(return_value=page)

        results = list(client.paginate("/api/v3/test", {}, "data"))

        assert len(results) == 1
        assert client._request.call_count == 1

    def test_paginate_non_list_item_yields_single(self) -> None:
        """Test pagination yields single non-list item."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)
        client._request = Mock(return_value={"id": 1})

        results = list(client.paginate("/api/v3/test"))

        assert results == [{"id": 1}]
        assert client._request.call_count == 1

    def test_paginate_non_dict_response_stops(self) -> None:
        """Test pagination stops when response is not a dict."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)
        client._request = Mock(return_value="payload")

        results = list(client.paginate("/api/v3/test"))

        assert results == ["payload"]
        assert client._request.call_count == 1

    def test_paginate_empty_items(self) -> None:
        """Test pagination when items are empty."""
        client = CoinbaseClientBase(base_url=self.base_url, auth=self.auth)

        page = {"data": [], "cursor": None}
        client._request = Mock(return_value=page)

        results = list(client.paginate("/api/v3/test", {}, "data"))

        assert len(results) == 0
        # Should still perform the initial request even with no items
        assert client._request.call_count == 1
