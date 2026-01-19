"""Core unit tests for OrderSubmitter client order IDs."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from gpt_trader.features.live_trade.execution.order_submission import OrderSubmitter


class TestGenerateSubmitId:
    """Tests for _generate_submit_id method."""

    def test_uses_provided_client_order_id(
        self,
        submitter: OrderSubmitter,
    ) -> None:
        """Test that provided client order ID is used."""
        result = submitter._generate_submit_id("custom-id-123")
        assert result == "custom-id-123"

    def test_generates_id_when_none(
        self,
        submitter: OrderSubmitter,
    ) -> None:
        """Test that ID is generated when None provided."""
        result = submitter._generate_submit_id(None)
        assert result.startswith("test-bot-123_")
        assert len(result) > len("test-bot-123_")

    @patch.dict("os.environ", {"INTEGRATION_TEST_ORDER_ID": "forced-id"})
    def test_integration_mode_uses_env_override(
        self,
        mock_broker: MagicMock,
        mock_event_store: MagicMock,
        open_orders: list[str],
    ) -> None:
        """Test that integration mode uses environment override."""
        submitter = OrderSubmitter(
            broker=mock_broker,
            event_store=mock_event_store,
            bot_id="test-bot",
            open_orders=open_orders,
            integration_mode=True,
        )

        result = submitter._generate_submit_id(None)
        assert result == "forced-id"
