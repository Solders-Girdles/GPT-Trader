"""Unit tests for Coinbase client performance optimizations."""

import time
from unittest.mock import MagicMock, patch, call
from decimal import Decimal

import pytest

from bot_v2.features.brokerages.coinbase.client import CoinbaseClient, CoinbaseAuth


def test_keep_alive_header_added():
    """Test that keep-alive header is added by the real transport when enabled."""
    client = CoinbaseClient(
        base_url="https://api.coinbase.com",
        enable_keep_alive=True
    )

    # Verify keep-alive is enabled and opener exists
    assert client.enable_keep_alive is True
    assert client._opener is not None

    # Mock opener.open to avoid real network and capture Request interactions
    from unittest.mock import MagicMock

    mock_opener = MagicMock()
    mock_response = MagicMock()
    mock_response.getcode.return_value = 200
    mock_response.headers.items.return_value = []
    mock_response.read.return_value = b'{"success": true}'
    mock_response.__enter__ = MagicMock(return_value=mock_response)
    mock_response.__exit__ = MagicMock(return_value=None)
    mock_opener.open.return_value = mock_response
    client._opener = mock_opener

    with patch('urllib.request.Request') as mock_request:
        mock_req_obj = MagicMock()
        mock_request.return_value = mock_req_obj

        # Call the real transport
        status, headers, text = client._urllib_transport(
            "GET", "https://api.coinbase.com/test",
            {"Content-Type": "application/json"},
            None, 30
        )

        # Verify opener was used and Connection header was set by transport
        mock_opener.open.assert_called_once()
        calls = [c.args for c in mock_req_obj.add_header.call_args_list]
        assert ("Connection", "keep-alive") in calls
        assert status == 200 and text == '{"success": true}'


def test_keep_alive_disabled():
    """Test that keep-alive header is not added by the real transport when disabled."""
    client = CoinbaseClient(
        base_url="https://api.coinbase.com",
        enable_keep_alive=False
    )

    # Verify keep-alive is disabled and no opener
    assert client.enable_keep_alive is False
    assert client._opener is None

    from unittest.mock import MagicMock
    mock_response = MagicMock()
    mock_response.getcode.return_value = 200
    mock_response.headers.items.return_value = []
    mock_response.read.return_value = b'{"success": true}'
    mock_response.__enter__ = MagicMock(return_value=mock_response)
    mock_response.__exit__ = MagicMock(return_value=None)

    with patch('urllib.request.Request') as mock_request, \
         patch('urllib.request.urlopen') as mock_urlopen:
        mock_req_obj = MagicMock()
        mock_request.return_value = mock_req_obj
        mock_urlopen.return_value = mock_response

        status, headers, text = client._urllib_transport(
            "GET", "https://api.coinbase.com/test",
            {"Content-Type": "application/json"},
            None, 30
        )

        # Verify urlopen was used and no Connection header set
        mock_urlopen.assert_called_once()
        calls = [c.args for c in mock_req_obj.add_header.call_args_list]
        assert ("Connection", "keep-alive") not in calls
        assert status == 200 and text == '{"success": true}'


def test_shared_opener_created():
    """Test that shared opener is created when keep-alive is enabled."""
    client = CoinbaseClient(
        base_url="https://api.coinbase.com",
        enable_keep_alive=True
    )
    
    # Verify opener was created
    assert client._opener is not None
    
    # Verify opener is not created when disabled
    client_no_keepalive = CoinbaseClient(
        base_url="https://api.coinbase.com",
        enable_keep_alive=False
    )
    assert client_no_keepalive._opener is None


def test_backoff_jitter_deterministic(fake_clock):
    """Test that backoff jitter is deterministic based on attempt number."""
    with patch('bot_v2.features.brokerages.coinbase.client.get_config') as mock_config:
        mock_config.return_value = {
            'max_retries': 3,
            'retry_delay': 1.0,
            'jitter_factor': 0.1
        }
        
        client = CoinbaseClient(base_url="https://api.coinbase.com")
        
        # Track sleep calls
        sleep_calls = []
        
        def mock_transport(method, url, headers, body, timeout):
            # Return 429 to trigger retry
            if len(sleep_calls) < 2:
                return (429, {}, '{"error": "rate limited"}')
            return (200, {}, '{"success": true}')
        
        client._transport = mock_transport
        
        def capture_sleep(duration):
            sleep_calls.append(duration)
            fake_clock.sleep(duration)

        with patch('time.sleep', side_effect=capture_sleep):
            
            # Make request that will retry
            result = client._request("GET", "/test")
            
            # Verify deterministic jitter
            assert len(sleep_calls) == 2
            
            # First retry: base_delay * 2^0 + jitter
            # jitter = 1.0 * 0.1 * (1/10) = 0.01
            expected_first = 1.0 + 0.01
            assert abs(sleep_calls[0] - expected_first) < 0.001
            
            # Second retry: base_delay * 2^1 + jitter
            # jitter = 2.0 * 0.1 * (2/10) = 0.04
            expected_second = 2.0 + 0.04
            assert abs(sleep_calls[1] - expected_second) < 0.001


def test_jitter_disabled(fake_clock):
    """Test that jitter can be disabled by setting factor to 0."""
    with patch('bot_v2.features.brokerages.coinbase.client.get_config') as mock_config:
        mock_config.return_value = {
            'max_retries': 3,
            'retry_delay': 1.0,
            'jitter_factor': 0  # No jitter
        }
        
        client = CoinbaseClient(base_url="https://api.coinbase.com")
        
        # Track sleep calls
        sleep_calls = []
        
        def mock_transport(method, url, headers, body, timeout):
            # Return 429 to trigger retry
            if len(sleep_calls) < 2:
                return (429, {}, '{"error": "rate limited"}')
            return (200, {}, '{"success": true}')
        
        client._transport = mock_transport
        
        def capture_sleep(duration):
            sleep_calls.append(duration)
            fake_clock.sleep(duration)

        with patch('time.sleep', side_effect=capture_sleep):
            
            # Make request that will retry
            result = client._request("GET", "/test")
            
            # Verify no jitter added
            assert len(sleep_calls) == 2
            assert sleep_calls[0] == 1.0  # Exact base delay
            assert sleep_calls[1] == 2.0  # Exact doubled delay


def test_connection_reuse_with_opener():
    """Test that opener is used for requests when keep-alive is enabled."""
    client = CoinbaseClient(
        base_url="https://api.coinbase.com",
        enable_keep_alive=True
    )
    
    # Mock the opener
    mock_opener = MagicMock()
    mock_response = MagicMock()
    mock_response.getcode.return_value = 200
    mock_response.headers.items.return_value = []
    mock_response.read.return_value = b'{"success": true}'
    mock_response.__enter__ = MagicMock(return_value=mock_response)
    mock_response.__exit__ = MagicMock(return_value=None)
    mock_opener.open.return_value = mock_response
    
    client._opener = mock_opener
    
    # Make a request using real transport
    with patch('urllib.request.Request') as mock_request:
        mock_req_obj = MagicMock()
        mock_request.return_value = mock_req_obj
        
        # Call the transport directly
        status, headers, text = client._urllib_transport(
            "GET", "https://api.coinbase.com/test",
            {"Content-Type": "application/json"},
            None, 30
        )
        
        # Verify opener was used
        mock_opener.open.assert_called_once()
        assert status == 200
        assert text == '{"success": true}'


def test_rate_limit_tracking_performance():
    """Test that rate limit tracking doesn't significantly impact performance."""
    client = CoinbaseClient(
        base_url="https://api.coinbase.com",
        rate_limit_per_minute=100,
        enable_throttle=True
    )
    
    # Mock transport for fast responses
    def mock_transport(method, url, headers, body, timeout):
        return (200, {}, '{"success": true}')
    
    client._transport = mock_transport
    
    # Time multiple requests
    start = time.time()
    for _ in range(50):
        client._request("GET", "/test")
    elapsed = time.time() - start
    
    # Should be very fast without hitting rate limits
    assert elapsed < 1.0  # 50 requests should take less than 1 second
    
    # Verify rate limit tracking is working
    assert len(client._request_times) == 50
