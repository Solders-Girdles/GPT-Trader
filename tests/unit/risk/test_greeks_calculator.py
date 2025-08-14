"""
Tests for Greeks Calculator
Phase 3, Week 3: RISK-008
Test suite for options Greeks calculations
"""

from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import numpy as np
import pytest

from src.bot.risk.greeks_calculator import Greeks, GreeksCalculator, OptionType


class TestGreeksCalculator:
    """Test suite for Greeks Calculator"""

    @pytest.fixture
    def calculator(self):
        """Create GreeksCalculator instance"""
        return GreeksCalculator()

    @pytest.fixture
    def option_params(self):
        """Standard option parameters for testing"""
        return {
            "spot_price": 100.0,
            "strike_price": 105.0,
            "time_to_expiry": 0.25,  # 3 months
            "risk_free_rate": 0.05,
            "volatility": 0.20,
            "dividend_yield": 0.02,
        }

    def test_black_scholes_call_price(self, calculator, option_params):
        """Test Black-Scholes call option pricing"""
        price = calculator.black_scholes_price(option_type=OptionType.CALL, **option_params)

        # Price should be positive
        assert price > 0
        # Out-of-money call should be worth less than intrinsic value
        intrinsic = max(option_params["spot_price"] - option_params["strike_price"], 0)
        assert price > intrinsic  # Has time value
        # Reasonable bounds check
        assert price < option_params["spot_price"]

    def test_black_scholes_put_price(self, calculator, option_params):
        """Test Black-Scholes put option pricing"""
        price = calculator.black_scholes_price(option_type=OptionType.PUT, **option_params)

        # Price should be positive
        assert price > 0
        # In-the-money put should have value
        intrinsic = max(option_params["strike_price"] - option_params["spot_price"], 0)
        assert price > intrinsic
        # Reasonable bounds
        assert price < option_params["strike_price"]

    def test_put_call_parity(self, calculator, option_params):
        """Test put-call parity relationship"""
        call_price = calculator.black_scholes_price(option_type=OptionType.CALL, **option_params)
        put_price = calculator.black_scholes_price(option_type=OptionType.PUT, **option_params)

        # Put-Call Parity: C - P = S - K * exp(-r*T)
        S = option_params["spot_price"]
        K = option_params["strike_price"]
        r = option_params["risk_free_rate"]
        T = option_params["time_to_expiry"]
        q = option_params["dividend_yield"]

        left_side = call_price - put_price
        right_side = S * np.exp(-q * T) - K * np.exp(-r * T)

        # Should be approximately equal
        assert abs(left_side - right_side) < 0.01

    def test_delta_calculation(self, calculator, option_params):
        """Test Delta calculation"""
        call_delta = calculator.calculate_delta(option_type=OptionType.CALL, **option_params)
        put_delta = calculator.calculate_delta(option_type=OptionType.PUT, **option_params)

        # Call delta should be between 0 and 1
        assert 0 <= call_delta <= 1
        # Put delta should be between -1 and 0
        assert -1 <= put_delta <= 0
        # Put-call delta relationship
        assert (
            abs(
                (call_delta - put_delta)
                - np.exp(-option_params["dividend_yield"] * option_params["time_to_expiry"])
            )
            < 0.01
        )

    def test_gamma_calculation(self, calculator, option_params):
        """Test Gamma calculation"""
        call_gamma = calculator.calculate_gamma(**option_params)
        put_gamma = calculator.calculate_gamma(**option_params)

        # Gamma should be positive for both calls and puts
        assert call_gamma > 0
        assert put_gamma > 0
        # Gamma is the same for calls and puts
        assert abs(call_gamma - put_gamma) < 0.0001
        # Gamma should be highest ATM
        atm_params = option_params.copy()
        atm_params["strike_price"] = atm_params["spot_price"]
        atm_gamma = calculator.calculate_gamma(**atm_params)
        assert atm_gamma >= call_gamma

    def test_vega_calculation(self, calculator, option_params):
        """Test Vega calculation"""
        call_vega = calculator.calculate_vega(**option_params)
        put_vega = calculator.calculate_vega(**option_params)

        # Vega should be positive for both
        assert call_vega > 0
        assert put_vega > 0
        # Vega is the same for calls and puts
        assert abs(call_vega - put_vega) < 0.0001
        # Vega should decrease as time to expiry decreases
        short_term_params = option_params.copy()
        short_term_params["time_to_expiry"] = 0.01
        short_vega = calculator.calculate_vega(**short_term_params)
        assert short_vega < call_vega

    def test_theta_calculation(self, calculator, option_params):
        """Test Theta calculation"""
        call_theta = calculator.calculate_theta(option_type=OptionType.CALL, **option_params)
        put_theta = calculator.calculate_theta(option_type=OptionType.PUT, **option_params)

        # Theta should typically be negative (time decay)
        # Note: Can be positive for deep ITM puts due to interest
        assert call_theta < 0
        # Theta magnitude should be reasonable
        assert abs(call_theta) < option_params["spot_price"]
        assert abs(put_theta) < option_params["spot_price"]

    def test_rho_calculation(self, calculator, option_params):
        """Test Rho calculation"""
        call_rho = calculator.calculate_rho(option_type=OptionType.CALL, **option_params)
        put_rho = calculator.calculate_rho(option_type=OptionType.PUT, **option_params)

        # Call rho should be positive
        assert call_rho > 0
        # Put rho should be negative
        assert put_rho < 0
        # Magnitude check
        assert abs(call_rho) < option_params["strike_price"]
        assert abs(put_rho) < option_params["strike_price"]

    def test_all_greeks_calculation(self, calculator, option_params):
        """Test calculation of all Greeks together"""
        greeks = calculator.calculate_all_greeks(option_type=OptionType.CALL, **option_params)

        assert isinstance(greeks, Greeks)
        assert greeks.delta is not None
        assert greeks.gamma is not None
        assert greeks.vega is not None
        assert greeks.theta is not None
        assert greeks.rho is not None

        # Check relationships
        assert 0 <= greeks.delta <= 1
        assert greeks.gamma > 0
        assert greeks.vega > 0
        assert greeks.theta < 0
        assert greeks.rho > 0

    def test_second_order_greeks(self, calculator, option_params):
        """Test second-order Greeks (Vanna, Charm, etc.)"""
        # Vanna (dVega/dSpot or dDelta/dVol)
        vanna = calculator.calculate_vanna(**option_params)

        # Charm (dDelta/dTime)
        charm = calculator.calculate_charm(option_type=OptionType.CALL, **option_params)

        # Vanna can be positive or negative
        assert vanna is not None
        assert isinstance(vanna, float)

        # Charm represents delta decay
        assert charm is not None
        assert isinstance(charm, float)

    def test_portfolio_greeks_aggregation(self, calculator):
        """Test portfolio-level Greeks aggregation"""
        positions = [
            {
                "option_type": OptionType.CALL,
                "quantity": 10,
                "spot_price": 100,
                "strike_price": 105,
                "time_to_expiry": 0.25,
                "risk_free_rate": 0.05,
                "volatility": 0.20,
                "dividend_yield": 0.02,
            },
            {
                "option_type": OptionType.PUT,
                "quantity": -5,  # Short position
                "spot_price": 100,
                "strike_price": 95,
                "time_to_expiry": 0.25,
                "risk_free_rate": 0.05,
                "volatility": 0.20,
                "dividend_yield": 0.02,
            },
        ]

        portfolio_greeks = calculator.calculate_portfolio_greeks(positions)

        assert isinstance(portfolio_greeks, dict)
        assert "total_delta" in portfolio_greeks
        assert "total_gamma" in portfolio_greeks
        assert "total_vega" in portfolio_greeks
        assert "total_theta" in portfolio_greeks
        assert "total_rho" in portfolio_greeks

        # Check that aggregation considers position sizes
        first_delta = calculator.calculate_delta(
            option_type=positions[0]["option_type"],
            spot_price=positions[0]["spot_price"],
            strike_price=positions[0]["strike_price"],
            time_to_expiry=positions[0]["time_to_expiry"],
            risk_free_rate=positions[0]["risk_free_rate"],
            volatility=positions[0]["volatility"],
            dividend_yield=positions[0]["dividend_yield"],
        )

        # Portfolio delta should reflect position quantities
        assert abs(portfolio_greeks["total_delta"]) != abs(first_delta)

    def test_delta_hedging_calculation(self, calculator, option_params):
        """Test delta hedging requirements"""
        position_size = 100  # 100 call options

        delta = calculator.calculate_delta(option_type=OptionType.CALL, **option_params)

        hedge_shares = calculator.calculate_hedge_requirement(
            delta=delta, position_size=position_size
        )

        # Should need to short shares to hedge long calls
        assert hedge_shares == -delta * position_size

    def test_gamma_scalping_pnl(self, calculator, option_params):
        """Test gamma scalping P&L calculation"""
        gamma = calculator.calculate_gamma(**option_params)
        spot_move = 2.0  # $2 move in underlying

        gamma_pnl = calculator.calculate_gamma_pnl(
            gamma=gamma, spot_move=spot_move, position_size=100
        )

        # Gamma P&L formula: 0.5 * gamma * (spot_move)^2 * position_size
        expected_pnl = 0.5 * gamma * spot_move**2 * 100
        assert abs(gamma_pnl - expected_pnl) < 0.01

    def test_implied_volatility_calculation(self, calculator, option_params):
        """Test implied volatility calculation from option price"""
        # First calculate theoretical price
        market_price = calculator.black_scholes_price(option_type=OptionType.CALL, **option_params)

        # Now calculate implied vol from that price
        implied_vol = calculator.calculate_implied_volatility(
            option_type=OptionType.CALL,
            market_price=market_price,
            spot_price=option_params["spot_price"],
            strike_price=option_params["strike_price"],
            time_to_expiry=option_params["time_to_expiry"],
            risk_free_rate=option_params["risk_free_rate"],
            dividend_yield=option_params["dividend_yield"],
        )

        # Should recover the original volatility
        assert abs(implied_vol - option_params["volatility"]) < 0.001

    def test_greeks_at_expiry(self, calculator, option_params):
        """Test Greeks behavior at expiry"""
        expiry_params = option_params.copy()
        expiry_params["time_to_expiry"] = 0.001  # Very close to expiry

        greeks = calculator.calculate_all_greeks(option_type=OptionType.CALL, **expiry_params)

        # At expiry, ITM call delta -> 1, OTM -> 0
        if expiry_params["spot_price"] > expiry_params["strike_price"]:
            assert greeks.delta > 0.9
        else:
            assert greeks.delta < 0.1

        # Gamma should be very high near ATM at expiry
        if abs(expiry_params["spot_price"] - expiry_params["strike_price"]) < 1:
            assert greeks.gamma > 1

        # Theta should be very negative (rapid time decay)
        assert greeks.theta < -10

    def test_greeks_deep_itm_otm(self, calculator, option_params):
        """Test Greeks for deep ITM and OTM options"""
        # Deep ITM call
        itm_params = option_params.copy()
        itm_params["strike_price"] = 80  # Deep ITM
        itm_greeks = calculator.calculate_all_greeks(option_type=OptionType.CALL, **itm_params)

        # Deep ITM call should have delta close to 1
        assert itm_greeks.delta > 0.95
        # Low gamma (less sensitive to spot moves)
        assert itm_greeks.gamma < 0.01

        # Deep OTM call
        otm_params = option_params.copy()
        otm_params["strike_price"] = 130  # Deep OTM
        otm_greeks = calculator.calculate_all_greeks(option_type=OptionType.CALL, **otm_params)

        # Deep OTM call should have delta close to 0
        assert otm_greeks.delta < 0.05
        # Low gamma
        assert otm_greeks.gamma < 0.01

    def test_volatility_smile_adjustment(self, calculator, option_params):
        """Test volatility smile adjustments"""
        # Different volatilities for different strikes
        strikes = [90, 95, 100, 105, 110]
        vols = [0.25, 0.22, 0.20, 0.22, 0.25]  # Smile shape

        prices = []
        for strike, vol in zip(strikes, vols, strict=False):
            params = option_params.copy()
            params["strike_price"] = strike
            params["volatility"] = vol
            price = calculator.black_scholes_price(option_type=OptionType.CALL, **params)
            prices.append(price)

        # Check that smile affects prices appropriately
        # Wings should be relatively more expensive
        atm_idx = 2  # 100 strike
        assert prices[0] > 0  # 90 strike
        assert prices[4] > 0  # 110 strike

    def test_american_option_adjustment(self, calculator, option_params):
        """Test American option early exercise premium"""
        # For American options, especially puts, can be optimal to exercise early
        american_put_value = calculator.calculate_american_option_value(
            option_type=OptionType.PUT, **option_params
        )

        european_put_value = calculator.black_scholes_price(
            option_type=OptionType.PUT, **option_params
        )

        # American option should be worth at least as much as European
        assert american_put_value >= european_put_value

    def test_discrete_dividend_adjustment(self, calculator, option_params):
        """Test discrete dividend adjustments"""
        # Add discrete dividend
        dividend_params = option_params.copy()
        dividend_params["discrete_dividends"] = [{"amount": 2.0, "ex_date_days": 30}]

        price_with_div = calculator.black_scholes_price_with_dividends(
            option_type=OptionType.CALL, **dividend_params
        )

        price_without_div = calculator.black_scholes_price(
            option_type=OptionType.CALL, **option_params
        )

        # Call price should be lower with dividends
        assert price_with_div < price_without_div

    def test_risk_reversal_strategy(self, calculator, option_params):
        """Test risk reversal strategy Greeks"""
        # Buy call, sell put at same strike
        call_greeks = calculator.calculate_all_greeks(option_type=OptionType.CALL, **option_params)
        put_greeks = calculator.calculate_all_greeks(option_type=OptionType.PUT, **option_params)

        # Risk reversal Greeks
        rr_delta = call_greeks.delta - put_greeks.delta
        rr_gamma = call_greeks.gamma - put_greeks.gamma
        rr_vega = call_greeks.vega - put_greeks.vega

        # Risk reversal is directional (positive delta)
        assert rr_delta > 0
        # Gamma should be close to zero (calls and puts have same gamma)
        assert abs(rr_gamma) < 0.001
        # Vega should be close to zero
        assert abs(rr_vega) < 0.001

    def test_straddle_strategy_greeks(self, calculator, option_params):
        """Test straddle strategy Greeks"""
        # Buy call and put at same strike
        atm_params = option_params.copy()
        atm_params["strike_price"] = atm_params["spot_price"]

        call_greeks = calculator.calculate_all_greeks(option_type=OptionType.CALL, **atm_params)
        put_greeks = calculator.calculate_all_greeks(option_type=OptionType.PUT, **atm_params)

        # Straddle Greeks
        straddle_delta = call_greeks.delta + put_greeks.delta
        straddle_gamma = call_greeks.gamma + put_greeks.gamma
        straddle_vega = call_greeks.vega + put_greeks.vega

        # ATM straddle should be delta-neutral
        assert abs(straddle_delta) < 0.1
        # Double gamma (benefits from movement)
        assert straddle_gamma > call_greeks.gamma
        # Double vega (benefits from vol increase)
        assert straddle_vega > call_greeks.vega

    def test_error_handling(self, calculator):
        """Test error handling for invalid inputs"""
        # Negative spot price
        with pytest.raises(ValueError):
            calculator.black_scholes_price(
                option_type=OptionType.CALL,
                spot_price=-100,
                strike_price=100,
                time_to_expiry=0.25,
                risk_free_rate=0.05,
                volatility=0.20,
                dividend_yield=0.02,
            )

        # Negative volatility
        with pytest.raises(ValueError):
            calculator.calculate_all_greeks(
                option_type=OptionType.CALL,
                spot_price=100,
                strike_price=100,
                time_to_expiry=0.25,
                risk_free_rate=0.05,
                volatility=-0.20,
                dividend_yield=0.02,
            )

        # Invalid option type
        with pytest.raises(ValueError):
            calculator.black_scholes_price(
                option_type="INVALID",
                spot_price=100,
                strike_price=100,
                time_to_expiry=0.25,
                risk_free_rate=0.05,
                volatility=0.20,
                dividend_yield=0.02,
            )

    def test_numerical_stability(self, calculator):
        """Test numerical stability for extreme parameters"""
        # Very short time to expiry
        short_params = {
            "spot_price": 100.0,
            "strike_price": 100.0,
            "time_to_expiry": 0.0001,  # Very close to expiry
            "risk_free_rate": 0.05,
            "volatility": 0.20,
            "dividend_yield": 0.02,
        }

        greeks = calculator.calculate_all_greeks(option_type=OptionType.CALL, **short_params)

        # Should not produce NaN or Inf
        assert np.isfinite(greeks.delta)
        assert np.isfinite(greeks.gamma)

        # Very high volatility
        high_vol_params = short_params.copy()
        high_vol_params["volatility"] = 2.0  # 200% vol
        high_vol_params["time_to_expiry"] = 0.25

        high_vol_greeks = calculator.calculate_all_greeks(
            option_type=OptionType.CALL, **high_vol_params
        )

        assert np.isfinite(high_vol_greeks.delta)
        assert np.isfinite(high_vol_greeks.vega)


class TestGreeksCalculatorIntegration:
    """Integration tests for Greeks Calculator"""

    @pytest.fixture
    def mock_market_data(self):
        """Mock market data provider"""
        mock = Mock()
        mock.get_spot_price.return_value = 100.0
        mock.get_volatility.return_value = 0.20
        mock.get_risk_free_rate.return_value = 0.05
        return mock

    def test_integration_with_market_data(self, mock_market_data):
        """Test integration with market data provider"""
        calculator = GreeksCalculator(market_data=mock_market_data)

        greeks = calculator.calculate_greeks_for_position(
            symbol="AAPL",
            option_type=OptionType.CALL,
            strike_price=105,
            expiry_date=datetime.now() + timedelta(days=90),
        )

        assert mock_market_data.get_spot_price.called
        assert mock_market_data.get_volatility.called
        assert greeks is not None

    @patch("src.bot.risk.greeks_calculator.send_alert")
    def test_alert_on_extreme_greeks(self, mock_alert):
        """Test alert generation for extreme Greeks values"""
        calculator = GreeksCalculator()
        calculator.gamma_limit = 0.5

        # Create position with high gamma
        params = {
            "spot_price": 100.0,
            "strike_price": 100.0,  # ATM for high gamma
            "time_to_expiry": 0.01,  # Short expiry for high gamma
            "risk_free_rate": 0.05,
            "volatility": 0.30,
            "dividend_yield": 0.02,
        }

        greeks = calculator.calculate_and_monitor(option_type=OptionType.CALL, **params)

        # Should trigger alert for high gamma
        if greeks.gamma > calculator.gamma_limit:
            assert mock_alert.called


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
