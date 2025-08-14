"""
Greeks Calculator for Options and Portfolio Risk
Phase 3, Week 3: RISK-005
Calculate option Greeks and portfolio sensitivities
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from scipy import stats
from scipy.optimize import brentq
import warnings

logger = logging.getLogger(__name__)


class OptionType(Enum):
    """Option types"""
    CALL = "call"
    PUT = "put"


class GreekType(Enum):
    """Types of Greeks"""
    DELTA = "delta"      # Rate of change of option price w.r.t. underlying price
    GAMMA = "gamma"      # Rate of change of delta w.r.t. underlying price
    VEGA = "vega"        # Sensitivity to volatility
    THETA = "theta"      # Time decay
    RHO = "rho"          # Sensitivity to interest rate


@dataclass
class OptionContract:
    """Option contract details"""
    symbol: str
    underlying_price: float
    strike_price: float
    time_to_expiry: float  # In years
    volatility: float      # Implied volatility
    risk_free_rate: float
    option_type: OptionType
    quantity: int = 1
    
    # Optional market price for IV calculation
    market_price: Optional[float] = None


@dataclass
class Greeks:
    """Container for option Greeks"""
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    
    # Additional metrics
    lambda_leverage: Optional[float] = None  # Lambda (elasticity)
    vanna: Optional[float] = None  # dDelta/dVol
    charm: Optional[float] = None  # dDelta/dTime
    vomma: Optional[float] = None  # dVega/dVol
    speed: Optional[float] = None  # dGamma/dSpot
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'delta': self.delta,
            'gamma': self.gamma,
            'vega': self.vega,
            'theta': self.theta,
            'rho': self.rho,
            'lambda': self.lambda_leverage,
            'vanna': self.vanna,
            'charm': self.charm
        }


class BlackScholesCalculator:
    """
    Black-Scholes model for European options.
    
    Used for calculating theoretical prices and Greeks.
    """
    
    @staticmethod
    def calculate_d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> Tuple[float, float]:
        """
        Calculate d1 and d2 for Black-Scholes formula.
        
        Args:
            S: Current price of underlying
            K: Strike price
            T: Time to expiry (years)
            r: Risk-free rate
            sigma: Volatility
            
        Returns:
            Tuple of (d1, d2)
        """
        if T <= 0:
            return 0, 0
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        return d1, d2
    
    @staticmethod
    def calculate_option_price(option: OptionContract) -> float:
        """
        Calculate theoretical option price using Black-Scholes.
        
        Args:
            option: Option contract details
            
        Returns:
            Theoretical option price
        """
        S = option.underlying_price
        K = option.strike_price
        T = option.time_to_expiry
        r = option.risk_free_rate
        sigma = option.volatility
        
        if T <= 0:
            # Option expired
            if option.option_type == OptionType.CALL:
                return max(S - K, 0)
            else:
                return max(K - S, 0)
        
        d1, d2 = BlackScholesCalculator.calculate_d1_d2(S, K, T, r, sigma)
        
        if option.option_type == OptionType.CALL:
            price = S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
        else:  # PUT
            price = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)
        
        return price
    
    @staticmethod
    def calculate_implied_volatility(option: OptionContract, 
                                    market_price: float,
                                    max_iterations: int = 100) -> float:
        """
        Calculate implied volatility from market price.
        
        Args:
            option: Option contract details
            market_price: Current market price of option
            max_iterations: Maximum iterations for numerical method
            
        Returns:
            Implied volatility
        """
        def objective(sigma):
            test_option = OptionContract(
                symbol=option.symbol,
                underlying_price=option.underlying_price,
                strike_price=option.strike_price,
                time_to_expiry=option.time_to_expiry,
                volatility=sigma,
                risk_free_rate=option.risk_free_rate,
                option_type=option.option_type
            )
            return BlackScholesCalculator.calculate_option_price(test_option) - market_price
        
        try:
            # Use Brent's method to find IV
            iv = brentq(objective, 0.001, 5.0, maxiter=max_iterations)
            return iv
        except:
            # Fall back to simple approximation
            return 0.2  # Default 20% volatility


class GreeksCalculator:
    """
    Calculate option Greeks using Black-Scholes model.
    
    Provides first-order and second-order Greeks.
    """
    
    def __init__(self):
        """Initialize Greeks calculator"""
        self.bs_calculator = BlackScholesCalculator()
    
    def calculate_greeks(self, option: OptionContract) -> Greeks:
        """
        Calculate all Greeks for an option.
        
        Args:
            option: Option contract details
            
        Returns:
            Greeks object with all sensitivities
        """
        S = option.underlying_price
        K = option.strike_price
        T = option.time_to_expiry
        r = option.risk_free_rate
        sigma = option.volatility
        
        # Handle expired options
        if T <= 0:
            return Greeks(
                delta=0,
                gamma=0,
                vega=0,
                theta=0,
                rho=0
            )
        
        # Calculate d1 and d2
        d1, d2 = self.bs_calculator.calculate_d1_d2(S, K, T, r, sigma)
        
        # Calculate Greeks
        delta = self._calculate_delta(S, K, T, r, sigma, d1, option.option_type)
        gamma = self._calculate_gamma(S, K, T, r, sigma, d1)
        vega = self._calculate_vega(S, K, T, r, sigma, d1)
        theta = self._calculate_theta(S, K, T, r, sigma, d1, d2, option.option_type)
        rho = self._calculate_rho(S, K, T, r, sigma, d2, option.option_type)
        
        # Calculate option price for lambda
        option_price = self.bs_calculator.calculate_option_price(option)
        lambda_leverage = self._calculate_lambda(delta, S, option_price)
        
        # Second-order Greeks
        vanna = self._calculate_vanna(S, K, T, r, sigma, d1, d2)
        charm = self._calculate_charm(S, K, T, r, sigma, d1, d2, option.option_type)
        
        return Greeks(
            delta=delta,
            gamma=gamma,
            vega=vega,
            theta=theta,
            rho=rho,
            lambda_leverage=lambda_leverage,
            vanna=vanna,
            charm=charm
        )
    
    def _calculate_delta(self, S: float, K: float, T: float, r: float, 
                        sigma: float, d1: float, option_type: OptionType) -> float:
        """Calculate Delta"""
        if option_type == OptionType.CALL:
            return stats.norm.cdf(d1)
        else:  # PUT
            return stats.norm.cdf(d1) - 1
    
    def _calculate_gamma(self, S: float, K: float, T: float, r: float, 
                        sigma: float, d1: float) -> float:
        """Calculate Gamma (same for calls and puts)"""
        return stats.norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    def _calculate_vega(self, S: float, K: float, T: float, r: float, 
                       sigma: float, d1: float) -> float:
        """Calculate Vega (same for calls and puts)"""
        return S * stats.norm.pdf(d1) * np.sqrt(T) / 100  # Divided by 100 for 1% change
    
    def _calculate_theta(self, S: float, K: float, T: float, r: float, 
                        sigma: float, d1: float, d2: float, option_type: OptionType) -> float:
        """Calculate Theta"""
        term1 = -(S * stats.norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
        
        if option_type == OptionType.CALL:
            term2 = -r * K * np.exp(-r * T) * stats.norm.cdf(d2)
            theta = (term1 + term2) / 365  # Convert to daily theta
        else:  # PUT
            term2 = r * K * np.exp(-r * T) * stats.norm.cdf(-d2)
            theta = (term1 + term2) / 365  # Convert to daily theta
        
        return theta
    
    def _calculate_rho(self, S: float, K: float, T: float, r: float, 
                      sigma: float, d2: float, option_type: OptionType) -> float:
        """Calculate Rho"""
        if option_type == OptionType.CALL:
            return K * T * np.exp(-r * T) * stats.norm.cdf(d2) / 100  # For 1% change
        else:  # PUT
            return -K * T * np.exp(-r * T) * stats.norm.cdf(-d2) / 100  # For 1% change
    
    def _calculate_lambda(self, delta: float, S: float, option_price: float) -> float:
        """Calculate Lambda (elasticity)"""
        if option_price == 0:
            return 0
        return delta * S / option_price
    
    def _calculate_vanna(self, S: float, K: float, T: float, r: float, 
                        sigma: float, d1: float, d2: float) -> float:
        """Calculate Vanna (dDelta/dVol)"""
        return -stats.norm.pdf(d1) * d2 / sigma
    
    def _calculate_charm(self, S: float, K: float, T: float, r: float, 
                        sigma: float, d1: float, d2: float, option_type: OptionType) -> float:
        """Calculate Charm (dDelta/dTime)"""
        pdf_d1 = stats.norm.pdf(d1)
        term1 = pdf_d1 * (2 * r * T - d2 * sigma * np.sqrt(T)) / (2 * T * sigma * np.sqrt(T))
        
        if option_type == OptionType.CALL:
            return -term1
        else:  # PUT
            return -term1
    
    def calculate_portfolio_greeks(self, positions: List[OptionContract]) -> Dict[str, float]:
        """
        Calculate aggregated Greeks for a portfolio of options.
        
        Args:
            positions: List of option positions
            
        Returns:
            Dictionary of portfolio Greeks
        """
        portfolio_greeks = {
            'delta': 0,
            'gamma': 0,
            'vega': 0,
            'theta': 0,
            'rho': 0
        }
        
        for position in positions:
            greeks = self.calculate_greeks(position)
            
            # Weight by quantity
            portfolio_greeks['delta'] += greeks.delta * position.quantity
            portfolio_greeks['gamma'] += greeks.gamma * position.quantity
            portfolio_greeks['vega'] += greeks.vega * position.quantity
            portfolio_greeks['theta'] += greeks.theta * position.quantity
            portfolio_greeks['rho'] += greeks.rho * position.quantity
        
        return portfolio_greeks
    
    def calculate_delta_hedge_quantity(self, option: OptionContract) -> float:
        """
        Calculate quantity of underlying needed to delta hedge.
        
        Args:
            option: Option contract to hedge
            
        Returns:
            Quantity of underlying shares needed (negative for short)
        """
        greeks = self.calculate_greeks(option)
        return -greeks.delta * option.quantity
    
    def calculate_scenario_analysis(self, 
                                   option: OptionContract,
                                   price_range: Tuple[float, float] = (0.9, 1.1),
                                   vol_range: Tuple[float, float] = (0.5, 1.5),
                                   n_scenarios: int = 10) -> 'pd.DataFrame':
        """
        Calculate option value under different scenarios.
        
        Args:
            option: Option contract
            price_range: Range for underlying price (as multiplier)
            vol_range: Range for volatility (as multiplier)
            n_scenarios: Number of scenarios to calculate
            
        Returns:
            DataFrame with scenario analysis
        """
        import pandas as pd
        
        results = []
        
        # Generate scenarios
        price_multipliers = np.linspace(price_range[0], price_range[1], n_scenarios)
        vol_multipliers = np.linspace(vol_range[0], vol_range[1], n_scenarios)
        
        base_price = self.bs_calculator.calculate_option_price(option)
        
        for price_mult in price_multipliers:
            for vol_mult in vol_multipliers:
                # Create scenario option
                scenario_option = OptionContract(
                    symbol=option.symbol,
                    underlying_price=option.underlying_price * price_mult,
                    strike_price=option.strike_price,
                    time_to_expiry=option.time_to_expiry,
                    volatility=option.volatility * vol_mult,
                    risk_free_rate=option.risk_free_rate,
                    option_type=option.option_type
                )
                
                # Calculate price and Greeks
                scenario_price = self.bs_calculator.calculate_option_price(scenario_option)
                scenario_greeks = self.calculate_greeks(scenario_option)
                
                results.append({
                    'price_change': (price_mult - 1) * 100,
                    'vol_change': (vol_mult - 1) * 100,
                    'option_price': scenario_price,
                    'price_change_pct': (scenario_price / base_price - 1) * 100,
                    'delta': scenario_greeks.delta,
                    'gamma': scenario_greeks.gamma,
                    'vega': scenario_greeks.vega
                })
        
        return pd.DataFrame(results)


def demonstrate_greeks():
    """Demonstrate Greeks calculations"""
    print("Greeks Calculator Demonstration")
    print("=" * 60)
    
    # Create sample option
    option = OptionContract(
        symbol="AAPL_CALL_150",
        underlying_price=145.0,
        strike_price=150.0,
        time_to_expiry=30/365,  # 30 days
        volatility=0.25,  # 25% implied volatility
        risk_free_rate=0.05,  # 5% risk-free rate
        option_type=OptionType.CALL,
        quantity=10
    )
    
    # Initialize calculator
    calculator = GreeksCalculator()
    
    # Calculate option price
    price = calculator.bs_calculator.calculate_option_price(option)
    print(f"\nOption Details:")
    print(f"  Type: {option.option_type.value.upper()}")
    print(f"  Underlying: ${option.underlying_price:.2f}")
    print(f"  Strike: ${option.strike_price:.2f}")
    print(f"  Days to expiry: {option.time_to_expiry * 365:.0f}")
    print(f"  Implied Vol: {option.volatility:.1%}")
    print(f"  Theoretical Price: ${price:.2f}")
    
    # Calculate Greeks
    greeks = calculator.calculate_greeks(option)
    
    print(f"\nGreeks:")
    print(f"  Delta: {greeks.delta:.4f}")
    print(f"    → 1% move in underlying = ${greeks.delta * option.underlying_price * 0.01:.2f} change")
    print(f"  Gamma: {greeks.gamma:.4f}")
    print(f"    → Delta changes by {greeks.gamma:.4f} per $1 move")
    print(f"  Vega: {greeks.vega:.4f}")
    print(f"    → 1% IV change = ${greeks.vega:.2f} change")
    print(f"  Theta: {greeks.theta:.4f}")
    print(f"    → Daily time decay = ${greeks.theta:.2f}")
    print(f"  Rho: {greeks.rho:.4f}")
    print(f"    → 1% rate change = ${greeks.rho:.2f} change")
    
    # Delta hedge calculation
    hedge_qty = calculator.calculate_delta_hedge_quantity(option)
    print(f"\nDelta Hedge:")
    print(f"  Shares needed: {hedge_qty:.0f} shares")
    print(f"  Hedge value: ${abs(hedge_qty) * option.underlying_price:,.2f}")
    
    # Create a put option for comparison
    put_option = OptionContract(
        symbol="AAPL_PUT_140",
        underlying_price=145.0,
        strike_price=140.0,
        time_to_expiry=30/365,
        volatility=0.25,
        risk_free_rate=0.05,
        option_type=OptionType.PUT,
        quantity=10
    )
    
    put_price = calculator.bs_calculator.calculate_option_price(put_option)
    put_greeks = calculator.calculate_greeks(put_option)
    
    print(f"\nPut Option Comparison:")
    print(f"  Strike: ${put_option.strike_price:.2f}")
    print(f"  Price: ${put_price:.2f}")
    print(f"  Delta: {put_greeks.delta:.4f}")
    print(f"  Gamma: {put_greeks.gamma:.4f}")
    
    # Portfolio Greeks
    portfolio = [option, put_option]
    portfolio_greeks = calculator.calculate_portfolio_greeks(portfolio)
    
    print(f"\nPortfolio Greeks (Call + Put):")
    print(f"  Net Delta: {portfolio_greeks['delta']:.2f}")
    print(f"  Net Gamma: {portfolio_greeks['gamma']:.2f}")
    print(f"  Net Vega: {portfolio_greeks['vega']:.2f}")
    print(f"  Net Theta: ${portfolio_greeks['theta']:.2f}/day")
    
    print("\n✅ Greeks Calculator operational!")


if __name__ == "__main__":
    demonstrate_greeks()