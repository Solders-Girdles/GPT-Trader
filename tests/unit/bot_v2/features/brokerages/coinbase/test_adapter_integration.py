"""
Integration tests for Coinbase adapter with SequenceGuard.
Verifies stream_user_events properly annotates messages with gap detection.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from bot_v2.features.brokerages.coinbase.adapter import CoinbaseBrokerage
from bot_v2.features.brokerages.coinbase.ws import SequenceGuard


class TestAdapterSequenceGuard:
    """Test adapter integration with SequenceGuard for gap detection."""
    
    def test_stream_user_events_annotates_gaps(self):
        """Verify stream_user_events uses SequenceGuard.annotate() and detects gaps."""
        # Mock WebSocket messages with a sequence gap
        mock_messages = [
            {"type": "heartbeat", "sequence": 1, "time": "2024-01-01T00:00:00Z"},
            {"type": "received", "sequence": 2, "order_id": "123", "side": "buy"},
            {"type": "done", "sequence": 4, "order_id": "124"},  # Gap here! Missing seq 3
            {"type": "match", "sequence": 5, "trade_id": "456"},
        ]
        
        # Create adapter with mocked WebSocket
        with patch('bot_v2.features.brokerages.coinbase.adapter.CoinbaseWebSocket') as MockWS:
            mock_ws = MockWS.return_value
            mock_ws.stream_messages.return_value = iter(mock_messages)
            
            # Build minimal API config
            from bot_v2.features.brokerages.coinbase.models import APIConfig
            cfg = APIConfig(
                api_key="k",
                api_secret="s",
                passphrase="p",
                base_url="https://test",
                api_mode="exchange"
            )
            adapter = CoinbaseBrokerage(config=cfg)
            
            # Collect annotated messages
            annotated_messages = list(adapter.stream_user_events(product_ids=[]))
            
            # Verify all messages were returned
            assert len(annotated_messages) == 4
            
            # Check first two messages have no gap
            assert "gap_detected" not in annotated_messages[0]
            assert "gap_detected" not in annotated_messages[1]
            
            # Check third message has gap detected
            assert annotated_messages[2].get("gap_detected") == True
            assert annotated_messages[2].get("last_seq") == 2
            assert annotated_messages[2].get("sequence") == 4
            
            # Fourth message should not have gap (continues from 4)
            assert "gap_detected" not in annotated_messages[3]
    
    def test_sequence_guard_different_sequence_fields(self):
        """Test SequenceGuard handles different sequence field names."""
        guard = SequenceGuard()
        
        # Test 'sequence' field
        msg1 = {"sequence": 10}
        result1 = guard.annotate(msg1)
        assert "gap_detected" not in result1
        
        # Test 'seq' field with gap
        msg2 = {"seq": 12}
        result2 = guard.annotate(msg2)
        assert result2.get("gap_detected") == True
        assert result2.get("last_seq") == 10
        
        # Test 'sequence_num' field
        msg3 = {"sequence_num": 13}
        result3 = guard.annotate(msg3)
        assert "gap_detected" not in result3
    
    def test_sequence_guard_handles_non_numeric(self):
        """Test SequenceGuard gracefully handles non-numeric sequences."""
        guard = SequenceGuard()
        
        # Non-numeric sequence should be ignored
        msg1 = {"sequence": "abc"}
        result1 = guard.annotate(msg1)
        assert result1 == msg1  # Unchanged
        assert "gap_detected" not in result1
        
        # Subsequent numeric sequence starts fresh
        msg2 = {"sequence": 100}
        result2 = guard.annotate(msg2)
        assert "gap_detected" not in result2
        
        # Now gaps are detected normally
        msg3 = {"sequence": 102}
        result3 = guard.annotate(msg3)
        assert result3.get("gap_detected") == True


class TestAPIModeBehavior:
    """Test adapter behaves correctly in different API modes."""
    
    @pytest.mark.parametrize("api_mode,expected_behavior", [
        ("advanced", "all_features"),
        ("exchange", "limited_features"),
    ])
    def test_mode_specific_features(self, api_mode, expected_behavior):
        """Test that features are available/blocked based on API mode."""
        from bot_v2.features.brokerages.coinbase.models import APIConfig
        
        config = APIConfig(
            api_key="test",
            api_secret="test",
            passphrase="test" if api_mode == "exchange" else None,
            base_url="https://test.com",
            api_mode=api_mode
        )
        
        adapter = CoinbaseBrokerage(config=config)
        
        # Mock the client
        with patch.object(adapter, 'client') as mock_client:
            if api_mode == "advanced":
                # All methods should work
                mock_client.list_portfolios.return_value = {"portfolios": []}
                result = adapter.client.list_portfolios()
                assert result == {"portfolios": []}
            else:
                # Some methods should raise InvalidRequestError
                from bot_v2.features.brokerages.coinbase.errors import InvalidRequestError
                mock_client.list_portfolios.side_effect = InvalidRequestError(
                    "list_portfolios not available in exchange mode"
                )
                with pytest.raises(InvalidRequestError) as exc:
                    adapter.client.list_portfolios()
                assert "not available in exchange mode" in str(exc.value)
