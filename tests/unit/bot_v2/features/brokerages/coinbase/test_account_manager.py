"""Tests for CoinbaseAccountManager high-level helpers."""

from decimal import Decimal

import pytest

from bot_v2.features.brokerages.coinbase.account_manager import CoinbaseAccountManager


class StubBroker:
    def __init__(self):
        self.calls = []

    def get_key_permissions(self):  # pragma: no cover - simple stub
        self.calls.append('key_permissions')
        return {'can_trade': True}

    def get_fee_schedule(self):
        self.calls.append('fee_schedule')
        return {'tier': 'Advanced'}

    def get_account_limits(self):
        self.calls.append('limits')
        return {'max_order': '100000'}

    def get_transaction_summary(self):
        self.calls.append('transaction_summary')
        return {'total_volume': '12345'}

    def list_payment_methods(self):
        self.calls.append('payment_methods')
        return [{'id': 'pm-1'}]

    def list_portfolios(self):
        self.calls.append('portfolios')
        return [{'uuid': 'pf-1'}]

    def create_convert_quote(self, payload):
        self.calls.append(('convert_quote', payload))
        return {'trade_id': 'trade-1', 'quote_id': 'q-1'}

    def commit_convert_trade(self, trade_id, payload):
        self.calls.append(('commit_trade', trade_id, payload))
        return {'trade_id': trade_id, 'status': 'pending'}

    def move_portfolio_funds(self, payload):
        self.calls.append(('move_funds', payload))
        return {'status': 'ok', **payload}


class StubEventStore:
    def __init__(self):
        self.metrics = []

    def append_metric(self, bot_id, metrics):
        self.metrics.append((bot_id, metrics))


def test_snapshot_collects_all_sections():
    broker = StubBroker()
    store = StubEventStore()
    manager = CoinbaseAccountManager(broker, event_store=store)

    snapshot = manager.snapshot()

    assert snapshot['key_permissions']['can_trade'] is True
    assert snapshot['fee_schedule']['tier'] == 'Advanced'
    assert snapshot['limits']['max_order'] == '100000'
    assert snapshot['transaction_summary']['total_volume'] == '12345'
    assert snapshot['payment_methods'][0]['id'] == 'pm-1'
    assert snapshot['portfolios'][0]['uuid'] == 'pf-1'
    assert any(m[1].get('event_type') == 'account_manager_snapshot' for m in store.metrics)


def test_convert_commits_when_requested():
    broker = StubBroker()
    store = StubEventStore()
    manager = CoinbaseAccountManager(broker, event_store=store)

    result = manager.convert({'from': 'USD', 'to': 'USDC', 'amount': '100'}, commit=True)

    assert result['trade_id'] == 'trade-1'
    assert any(call[0] == 'convert_quote' for call in broker.calls)
    assert any(call[0] == 'commit_trade' for call in broker.calls)
    assert any(m[1].get('event_type') == 'convert_commit' for m in store.metrics)


def test_move_funds_delegates_to_broker():
    broker = StubBroker()
    store = StubEventStore()
    manager = CoinbaseAccountManager(broker, event_store=store)

    payload = {'from_portfolio': 'pf-1', 'to_portfolio': 'pf-2', 'amount': '5'}
    result = manager.move_funds(payload)

    assert result['status'] == 'ok'
    assert ('move_funds', payload) in broker.calls
    assert any(m[1].get('event_type') == 'portfolio_move' for m in store.metrics)
