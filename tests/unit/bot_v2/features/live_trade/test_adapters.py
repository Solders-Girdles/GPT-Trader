"""Tests for live trade adapters"""

import pytest
from decimal import Decimal
from bot_v2.features.brokerages.core.interfaces import (
    OrderSide,
    OrderType,
    TimeInForce,
)
from bot_v2.features.live_trade.adapters import (
    to_core_tif,
    to_core_side,
    to_core_type,
    to_decimal,
)


class TestToCoreTimeInForce:
    """Test suite for to_core_tif function"""

    def test_string_day(self):
        """Test 'day' string conversion"""
        result = to_core_tif("day")
        assert result == TimeInForce.GTC

    def test_string_gtc(self):
        """Test 'gtc' string conversion"""
        result = to_core_tif("gtc")
        assert result == TimeInForce.GTC

    def test_string_ioc(self):
        """Test 'ioc' string conversion"""
        result = to_core_tif("ioc")
        assert result == TimeInForce.IOC

    def test_string_fok(self):
        """Test 'fok' string conversion"""
        result = to_core_tif("fok")
        assert result == TimeInForce.FOK

    def test_string_case_insensitive(self):
        """Test case-insensitive string conversion"""
        assert to_core_tif("GTC") == TimeInForce.GTC
        assert to_core_tif("Ioc") == TimeInForce.IOC
        assert to_core_tif("FOK") == TimeInForce.FOK

    def test_enum_passthrough(self):
        """Test that enum values pass through unchanged"""
        result = to_core_tif(TimeInForce.GTC)
        assert result == TimeInForce.GTC

        result = to_core_tif(TimeInForce.IOC)
        assert result == TimeInForce.IOC

    def test_unknown_string_defaults_to_gtc(self):
        """Test unknown string defaults to GTC"""
        result = to_core_tif("unknown")
        assert result == TimeInForce.GTC

    def test_empty_string_defaults_to_gtc(self):
        """Test empty string defaults to GTC"""
        result = to_core_tif("")
        assert result == TimeInForce.GTC


class TestToCoreOrderSide:
    """Test suite for to_core_side function"""

    def test_string_buy(self):
        """Test 'buy' string conversion"""
        result = to_core_side("buy")
        assert result == OrderSide.BUY

    def test_string_sell(self):
        """Test 'sell' string conversion"""
        result = to_core_side("sell")
        assert result == OrderSide.SELL

    def test_string_case_insensitive(self):
        """Test case-insensitive string conversion"""
        assert to_core_side("BUY") == OrderSide.BUY
        assert to_core_side("Buy") == OrderSide.BUY
        assert to_core_side("SELL") == OrderSide.SELL
        assert to_core_side("Sell") == OrderSide.SELL

    def test_enum_passthrough(self):
        """Test that enum values pass through unchanged"""
        result = to_core_side(OrderSide.BUY)
        assert result == OrderSide.BUY

        result = to_core_side(OrderSide.SELL)
        assert result == OrderSide.SELL

    def test_unknown_string_defaults_to_buy(self):
        """Test unknown string defaults to BUY"""
        result = to_core_side("unknown")
        assert result == OrderSide.BUY

    def test_empty_string_defaults_to_buy(self):
        """Test empty string defaults to BUY"""
        result = to_core_side("")
        assert result == OrderSide.BUY


class TestToCoreOrderType:
    """Test suite for to_core_type function"""

    def test_string_market(self):
        """Test 'market' string conversion"""
        result = to_core_type("market")
        assert result == OrderType.MARKET

    def test_string_limit(self):
        """Test 'limit' string conversion"""
        result = to_core_type("limit")
        assert result == OrderType.LIMIT

    def test_string_stop(self):
        """Test 'stop' string conversion"""
        result = to_core_type("stop")
        assert result == OrderType.STOP

    def test_string_stop_limit(self):
        """Test 'stop_limit' string conversion"""
        result = to_core_type("stop_limit")
        assert result == OrderType.STOP_LIMIT

    def test_string_case_insensitive(self):
        """Test case-insensitive string conversion"""
        assert to_core_type("MARKET") == OrderType.MARKET
        assert to_core_type("Market") == OrderType.MARKET
        assert to_core_type("LIMIT") == OrderType.LIMIT
        assert to_core_type("Limit") == OrderType.LIMIT

    def test_enum_passthrough(self):
        """Test that enum values pass through unchanged"""
        result = to_core_type(OrderType.MARKET)
        assert result == OrderType.MARKET

        result = to_core_type(OrderType.LIMIT)
        assert result == OrderType.LIMIT

    def test_unknown_string_defaults_to_market(self):
        """Test unknown string defaults to MARKET"""
        result = to_core_type("unknown")
        assert result == OrderType.MARKET

    def test_empty_string_defaults_to_market(self):
        """Test empty string defaults to MARKET"""
        result = to_core_type("")
        assert result == OrderType.MARKET


class TestToDecimal:
    """Test suite for to_decimal function"""

    def test_none_returns_none(self):
        """Test None returns None"""
        result = to_decimal(None)
        assert result is None

    def test_decimal_passthrough(self):
        """Test Decimal passes through unchanged"""
        value = Decimal("123.45")
        result = to_decimal(value)
        assert result == value
        assert result is value

    def test_integer_conversion(self):
        """Test integer to Decimal conversion"""
        result = to_decimal(100)
        assert result == Decimal("100")
        assert isinstance(result, Decimal)

    def test_float_conversion(self):
        """Test float to Decimal conversion"""
        result = to_decimal(123.45)
        assert result == Decimal("123.45")
        assert isinstance(result, Decimal)

    def test_string_conversion(self):
        """Test string to Decimal conversion"""
        result = to_decimal("99.99")
        assert result == Decimal("99.99")
        assert isinstance(result, Decimal)

    def test_zero_conversion(self):
        """Test zero conversion"""
        assert to_decimal(0) == Decimal("0")
        assert to_decimal(0.0) == Decimal("0")
        assert to_decimal("0") == Decimal("0")

    def test_negative_conversion(self):
        """Test negative number conversion"""
        assert to_decimal(-50) == Decimal("-50")
        assert to_decimal(-12.34) == Decimal("-12.34")
        assert to_decimal("-99.99") == Decimal("-99.99")

    def test_large_number_conversion(self):
        """Test large number conversion"""
        result = to_decimal(1000000)
        assert result == Decimal("1000000")

    def test_small_decimal_conversion(self):
        """Test small decimal conversion"""
        result = to_decimal("0.00000001")
        assert result == Decimal("0.00000001")

    def test_scientific_notation(self):
        """Test scientific notation conversion"""
        result = to_decimal("1.23e-5")
        assert result == Decimal("0.0000123")

    def test_precision_preservation(self):
        """Test that precision is preserved"""
        value = "123.456789012345678901234567890"
        result = to_decimal(value)
        assert str(result) == value
