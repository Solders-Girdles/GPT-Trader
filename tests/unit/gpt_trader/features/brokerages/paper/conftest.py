from __future__ import annotations

from unittest.mock import MagicMock, Mock

import pytest

import gpt_trader.features.brokerages.paper.hybrid as hybrid_module
from gpt_trader.features.brokerages.paper.hybrid import HybridPaperBroker


@pytest.fixture
def broker_factory(monkeypatch: pytest.MonkeyPatch):
    def _factory(
        *,
        client: Mock | MagicMock | None = None,
        **kwargs,
    ) -> HybridPaperBroker:
        monkeypatch.setattr(hybrid_module, "CoinbaseClient", MagicMock())
        monkeypatch.setattr(hybrid_module, "SimpleAuth", MagicMock())
        broker = HybridPaperBroker(
            api_key="test_key",
            private_key="test_private_key",
            **kwargs,
        )
        broker._client = client or Mock()
        return broker

    return _factory
