"""
Realistic Performance Targets for Trading Models
Phase 2.5 - Day 8

Sets and validates realistic performance expectations for ML trading models.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime types"""
    BULL = "bull"           # Strong uptrend
    BEAR = "bear"           # Strong downtrend
    SIDEWAYS = "sideways"   # Range-bound
    VOLATILE = "volatile"   # High volatility
    STABLE = "stable"       # Low volatility


class ModelType(Enum):
    """Trading model types"""
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    ARBITRAGE = "arbitrage"
    MIXED = "mixed"


@dataclass
class PerformanceTarget:
    """Realistic performance targets for trading models"""
    
    # Accuracy targets (realistic for financial markets)
    min_accuracy: float = 0.52      # Minimum viable accuracy
    target_accuracy: float = 0.58   # Target accuracy
    max_accuracy: float = 0.65      # Maximum realistic accuracy
    
    # Precision/Recall trade-off
    min_precision: float = 0.55     # Minimum precision for trades
    target_precision: float = 0.60  # Target precision
    min_recall: float = 0.40        # Can miss opportunities
    
    # Risk-adjusted returns
    min_sharpe: float = 0.5         # Minimum Sharpe ratio
    target_sharpe: float = 1.0      # Target Sharpe (good performance)
    excellent_sharpe: float = 1.5   # Excellent performance
    
    # Win rate and profit factor
    min_win_rate: float = 0.45      # Can be <50% with good R:R
    target_win_rate: float = 0.52   # Slight edge
    min_profit_factor: float = 1.2  # Minimum profit factor
    target_profit_factor: float = 1.5  # Target profit factor
    
    # Drawdown limits
    max_drawdown: float = 0.20      # Maximum 20% drawdown
    typical_drawdown: float = 0.12  # Typical drawdown
    
    # Return expectations (annual)
    min_annual_return: float = 0.08     # 8% minimum
    target_annual_return: float = 0.15  # 15% target
    max_annual_return: float = 0.30     # 30% maximum realistic
    
    # Risk metrics
    max_var_95: float = 0.02        # 2% Value at Risk (95%)
    max_cvar_95: float = 0.03       # 3% Conditional VaR
    
    # Trading frequency
    min_trades_per_day: int = 1     # Minimum activity
    max_trades_per_day: int = 20    # Maximum for day trading
    
    # Model stability
    min_stability_score: float = 0.7    # Minimum stability
    performance_decay_rate: float = 0.02  # 2% monthly decay expected


@dataclass
class MarketAdjustedTargets:
    """Targets adjusted for market conditions"""
    regime: MarketRegime
    base_targets: PerformanceTarget
    adjustments: Dict[str, float]
    
    def get_adjusted_target(self, metric: str) -> float:
        """Get adjusted target for specific metric"""
        base_value = getattr(self.base_targets, metric, None)
        if base_value is None:
            return None
        
        adjustment = self.adjustments.get(metric, 1.0)
        return base_value * adjustment


@dataclass
class PerformanceValidation:
    """Validation results for model performance"""
    is_realistic: bool
    meets_minimum: bool
    warnings: List[str]
    recommendations: List[str]
    score: float  # 0-100 overall performance score
    
    # Detailed metrics
    accuracy_assessment: str
    risk_assessment: str
    profitability_assessment: str
    stability_assessment: str


class RealisticTargetSetter:
    """
    Sets and validates realistic performance targets for trading models.
    
    Features:
    - Market regime awareness
    - Model type specific targets
    - Historical baseline comparison
    - Risk-adjusted expectations
    """
    
    def __init__(self):
        """Initialize target setter"""
        self.base_targets = PerformanceTarget()
        self.market_adjustments = self._initialize_market_adjustments()
        self.model_adjustments = self._initialize_model_adjustments()
        
        logger.info("RealisticTargetSetter initialized")
    
    def _initialize_market_adjustments(self) -> Dict[MarketRegime, Dict[str, float]]:
        """Initialize market regime adjustments"""
        return {
            MarketRegime.BULL: {
                'target_accuracy': 1.1,      # Easier in trending markets
                'target_sharpe': 1.2,
                'target_annual_return': 1.3,
                'max_drawdown': 0.8          # Lower drawdowns in bull markets
            },
            MarketRegime.BEAR: {
                'target_accuracy': 0.95,     # Harder in bear markets
                'target_sharpe': 0.9,
                'target_annual_return': 0.7,
                'max_drawdown': 1.3          # Higher drawdowns expected
            },
            MarketRegime.SIDEWAYS: {
                'target_accuracy': 0.9,      # Harder without trends
                'target_sharpe': 0.8,
                'target_annual_return': 0.6,
                'max_drawdown': 1.0
            },
            MarketRegime.VOLATILE: {
                'target_accuracy': 0.85,     # Very difficult
                'target_sharpe': 0.7,
                'target_annual_return': 0.8,  # Opportunities exist
                'max_drawdown': 1.5          # Higher risk
            },
            MarketRegime.STABLE: {
                'target_accuracy': 1.05,     # Slightly easier
                'target_sharpe': 1.1,
                'target_annual_return': 0.9,  # Lower returns
                'max_drawdown': 0.7          # Lower risk
            }
        }
    
    def _initialize_model_adjustments(self) -> Dict[ModelType, Dict[str, float]]:
        """Initialize model type adjustments"""
        return {
            ModelType.TREND_FOLLOWING: {
                'target_accuracy': 0.9,      # Lower accuracy OK
                'min_win_rate': 0.8,         # Can have <40% win rate
                'target_profit_factor': 1.2  # Higher R:R compensates
            },
            ModelType.MEAN_REVERSION: {
                'target_accuracy': 1.1,      # Higher accuracy expected
                'min_win_rate': 1.2,         # Higher win rate
                'target_profit_factor': 0.9  # Lower R:R acceptable
            },
            ModelType.MOMENTUM: {
                'target_accuracy': 0.95,
                'target_sharpe': 1.1,
                'max_drawdown': 1.2          # Higher volatility
            },
            ModelType.ARBITRAGE: {
                'target_accuracy': 1.3,      # Very high accuracy needed
                'min_precision': 1.5,        # Must be very precise
                'target_sharpe': 1.5         # Should have excellent Sharpe
            },
            ModelType.MIXED: {
                # No adjustments for mixed strategies
            }
        }
    
    def get_targets(self,
                   market_regime: Optional[MarketRegime] = None,
                   model_type: Optional[ModelType] = None) -> PerformanceTarget:
        """
        Get performance targets adjusted for conditions.
        
        Args:
            market_regime: Current market regime
            model_type: Type of trading model
            
        Returns:
            Adjusted performance targets
        """
        targets = PerformanceTarget()
        
        # Apply market regime adjustments
        if market_regime and market_regime in self.market_adjustments:
            adjustments = self.market_adjustments[market_regime]
            for metric, multiplier in adjustments.items():
                if hasattr(targets, metric):
                    current_value = getattr(targets, metric)
                    setattr(targets, metric, current_value * multiplier)
        
        # Apply model type adjustments
        if model_type and model_type in self.model_adjustments:
            adjustments = self.model_adjustments[model_type]
            for metric, multiplier in adjustments.items():
                if hasattr(targets, metric):
                    current_value = getattr(targets, metric)
                    setattr(targets, metric, current_value * multiplier)
        
        return targets
    
    def validate_performance(self,
                            metrics: Dict[str, float],
                            market_regime: Optional[MarketRegime] = None,
                            model_type: Optional[ModelType] = None) -> PerformanceValidation:
        """
        Validate if performance metrics are realistic.
        
        Args:
            metrics: Dictionary of performance metrics
            market_regime: Current market regime
            model_type: Type of trading model
            
        Returns:
            Validation results with assessment
        """
        targets = self.get_targets(market_regime, model_type)
        
        warnings = []
        recommendations = []
        scores = []
        
        # Validate accuracy
        accuracy = metrics.get('accuracy', 0)
        accuracy_assessment = self._assess_accuracy(accuracy, targets, warnings, recommendations)
        scores.append(self._score_metric(accuracy, targets.min_accuracy, 
                                        targets.target_accuracy, targets.max_accuracy))
        
        # Validate risk metrics
        sharpe = metrics.get('sharpe_ratio', 0)
        drawdown = abs(metrics.get('max_drawdown', 0))
        risk_assessment = self._assess_risk(sharpe, drawdown, targets, warnings, recommendations)
        scores.append(self._score_metric(sharpe, targets.min_sharpe,
                                        targets.target_sharpe, targets.excellent_sharpe))
        
        # Validate profitability
        win_rate = metrics.get('win_rate', 0)
        profit_factor = metrics.get('profit_factor', 1)
        returns = metrics.get('annual_return', 0)
        profitability_assessment = self._assess_profitability(
            win_rate, profit_factor, returns, targets, warnings, recommendations
        )
        scores.append(self._score_metric(profit_factor, targets.min_profit_factor,
                                        targets.target_profit_factor, 2.0))
        
        # Validate stability
        stability = metrics.get('stability_score', 0)
        stability_assessment = self._assess_stability(stability, targets, warnings, recommendations)
        scores.append(self._score_metric(stability, targets.min_stability_score, 0.85, 1.0))
        
        # Overall assessment
        overall_score = np.mean(scores) * 100
        is_realistic = self._is_realistic(metrics, targets)
        meets_minimum = self._meets_minimum(metrics, targets)
        
        return PerformanceValidation(
            is_realistic=is_realistic,
            meets_minimum=meets_minimum,
            warnings=warnings,
            recommendations=recommendations,
            score=overall_score,
            accuracy_assessment=accuracy_assessment,
            risk_assessment=risk_assessment,
            profitability_assessment=profitability_assessment,
            stability_assessment=stability_assessment
        )
    
    def _assess_accuracy(self, accuracy: float, targets: PerformanceTarget,
                        warnings: List[str], recommendations: List[str]) -> str:
        """Assess accuracy metrics"""
        if accuracy < targets.min_accuracy:
            warnings.append(f"Accuracy {accuracy:.1%} below minimum {targets.min_accuracy:.1%}")
            recommendations.append("Consider improving feature engineering or data quality")
            return "Below Minimum"
        elif accuracy > targets.max_accuracy:
            warnings.append(f"Accuracy {accuracy:.1%} suspiciously high (>{targets.max_accuracy:.1%})")
            recommendations.append("Check for data leakage or overfitting")
            return "Suspiciously High"
        elif accuracy >= targets.target_accuracy:
            return "Excellent"
        else:
            return "Acceptable"
    
    def _assess_risk(self, sharpe: float, drawdown: float, targets: PerformanceTarget,
                    warnings: List[str], recommendations: List[str]) -> str:
        """Assess risk metrics"""
        assessments = []
        
        if sharpe < targets.min_sharpe:
            warnings.append(f"Sharpe ratio {sharpe:.2f} below minimum {targets.min_sharpe:.2f}")
            recommendations.append("Improve risk-adjusted returns through better position sizing")
            assessments.append("Poor Sharpe")
        elif sharpe >= targets.excellent_sharpe:
            assessments.append("Excellent Sharpe")
        elif sharpe >= targets.target_sharpe:
            assessments.append("Good Sharpe")
        else:
            assessments.append("Acceptable Sharpe")
        
        if drawdown > targets.max_drawdown:
            warnings.append(f"Max drawdown {drawdown:.1%} exceeds limit {targets.max_drawdown:.1%}")
            recommendations.append("Implement stricter risk management and position limits")
            assessments.append("Excessive Drawdown")
        elif drawdown <= targets.typical_drawdown:
            assessments.append("Good Drawdown Control")
        else:
            assessments.append("Acceptable Drawdown")
        
        return ", ".join(assessments)
    
    def _assess_profitability(self, win_rate: float, profit_factor: float, 
                             returns: float, targets: PerformanceTarget,
                             warnings: List[str], recommendations: List[str]) -> str:
        """Assess profitability metrics"""
        assessments = []
        
        if win_rate < targets.min_win_rate and profit_factor < targets.target_profit_factor:
            warnings.append(f"Low win rate {win_rate:.1%} without compensating profit factor")
            recommendations.append("Improve entry timing or increase reward/risk ratio")
            assessments.append("Poor Win Rate")
        elif win_rate >= targets.target_win_rate:
            assessments.append("Good Win Rate")
        else:
            assessments.append("Acceptable Win Rate")
        
        if profit_factor < targets.min_profit_factor:
            warnings.append(f"Profit factor {profit_factor:.2f} below minimum {targets.min_profit_factor:.2f}")
            recommendations.append("Review exit strategies and stop loss placement")
            assessments.append("Poor Profit Factor")
        elif profit_factor >= targets.target_profit_factor:
            assessments.append("Good Profit Factor")
        
        if returns < targets.min_annual_return:
            warnings.append(f"Annual return {returns:.1%} below minimum {targets.min_annual_return:.1%}")
            assessments.append("Low Returns")
        elif returns > targets.max_annual_return:
            warnings.append(f"Annual return {returns:.1%} may be unsustainable (>{targets.max_annual_return:.1%})")
            assessments.append("Possibly Unsustainable Returns")
        elif returns >= targets.target_annual_return:
            assessments.append("Good Returns")
        
        return ", ".join(assessments) if assessments else "Acceptable"
    
    def _assess_stability(self, stability: float, targets: PerformanceTarget,
                         warnings: List[str], recommendations: List[str]) -> str:
        """Assess model stability"""
        if stability < targets.min_stability_score:
            warnings.append(f"Stability score {stability:.2f} below minimum {targets.min_stability_score:.2f}")
            recommendations.append("Consider ensemble methods or more robust features")
            return "Unstable"
        elif stability >= 0.85:
            return "Very Stable"
        else:
            return "Acceptable Stability"
    
    def _score_metric(self, value: float, min_val: float, target_val: float, max_val: float) -> float:
        """Score a metric on 0-1 scale"""
        if value <= min_val:
            return 0
        elif value >= max_val:
            return 1
        elif value >= target_val:
            # Linear interpolation between target and max
            return 0.7 + 0.3 * (value - target_val) / (max_val - target_val)
        else:
            # Linear interpolation between min and target
            return 0.7 * (value - min_val) / (target_val - min_val)
    
    def _is_realistic(self, metrics: Dict[str, float], targets: PerformanceTarget) -> bool:
        """Check if metrics are realistic"""
        accuracy = metrics.get('accuracy', 0)
        sharpe = metrics.get('sharpe_ratio', 0)
        
        # Check for unrealistic combinations
        if accuracy > 0.70 and sharpe > 2.0:
            return False  # Too good to be true
        
        if accuracy > targets.max_accuracy * 1.1:
            return False  # Likely overfitting
        
        return True
    
    def _meets_minimum(self, metrics: Dict[str, float], targets: PerformanceTarget) -> bool:
        """Check if metrics meet minimum requirements"""
        checks = [
            metrics.get('accuracy', 0) >= targets.min_accuracy,
            metrics.get('sharpe_ratio', 0) >= targets.min_sharpe,
            metrics.get('profit_factor', 1) >= targets.min_profit_factor,
            abs(metrics.get('max_drawdown', 1)) <= targets.max_drawdown
        ]
        return all(checks)
    
    def detect_market_regime(self, price_data: pd.Series, 
                            volatility_window: int = 20) -> MarketRegime:
        """
        Detect current market regime from price data.
        
        Args:
            price_data: Historical price series
            volatility_window: Window for volatility calculation
            
        Returns:
            Detected market regime
        """
        returns = price_data.pct_change().dropna()
        
        # Calculate metrics
        total_return = (price_data.iloc[-1] / price_data.iloc[0]) - 1
        volatility = returns.rolling(volatility_window).std().iloc[-1]
        avg_volatility = returns.std()
        
        # Trend detection
        sma_short = price_data.rolling(20).mean().iloc[-1]
        sma_long = price_data.rolling(50).mean().iloc[-1]
        
        # Classify regime
        if volatility > avg_volatility * 1.5:
            return MarketRegime.VOLATILE
        elif volatility < avg_volatility * 0.5:
            return MarketRegime.STABLE
        elif total_return > 0.10 and sma_short > sma_long:
            return MarketRegime.BULL
        elif total_return < -0.10 and sma_short < sma_long:
            return MarketRegime.BEAR
        else:
            return MarketRegime.SIDEWAYS
    
    def generate_performance_report(self, 
                                   metrics: Dict[str, float],
                                   market_regime: Optional[MarketRegime] = None,
                                   model_type: Optional[ModelType] = None) -> str:
        """Generate performance assessment report"""
        validation = self.validate_performance(metrics, market_regime, model_type)
        targets = self.get_targets(market_regime, model_type)
        
        report = f"""
Performance Assessment Report
============================

Overall Score: {validation.score:.1f}/100
Status: {'✅ PASS' if validation.meets_minimum else '❌ FAIL'}
Realistic: {'Yes' if validation.is_realistic else 'No - Check for issues'}

Market Regime: {market_regime.value if market_regime else 'Not specified'}
Model Type: {model_type.value if model_type else 'Not specified'}

Accuracy Assessment: {validation.accuracy_assessment}
  Current: {metrics.get('accuracy', 0):.1%}
  Target: {targets.target_accuracy:.1%}
  Range: [{targets.min_accuracy:.1%}, {targets.max_accuracy:.1%}]

Risk Assessment: {validation.risk_assessment}
  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f} (Target: {targets.target_sharpe:.2f})
  Max Drawdown: {abs(metrics.get('max_drawdown', 0)):.1%} (Limit: {targets.max_drawdown:.1%})

Profitability Assessment: {validation.profitability_assessment}
  Win Rate: {metrics.get('win_rate', 0):.1%} (Target: {targets.target_win_rate:.1%})
  Profit Factor: {metrics.get('profit_factor', 1):.2f} (Target: {targets.target_profit_factor:.2f})
  Annual Return: {metrics.get('annual_return', 0):.1%} (Target: {targets.target_annual_return:.1%})

Stability Assessment: {validation.stability_assessment}
  Stability Score: {metrics.get('stability_score', 0):.2f} (Minimum: {targets.min_stability_score:.2f})

Warnings:
"""
        for warning in validation.warnings:
            report += f"  ⚠️  {warning}\n"
        
        report += "\nRecommendations:\n"
        for rec in validation.recommendations:
            report += f"  •  {rec}\n"
        
        return report


def create_target_setter() -> RealisticTargetSetter:
    """Create target setter instance"""
    return RealisticTargetSetter()


if __name__ == "__main__":
    # Example usage
    setter = create_target_setter()
    
    # Example metrics (realistic)
    metrics = {
        'accuracy': 0.58,
        'sharpe_ratio': 1.1,
        'max_drawdown': -0.15,
        'win_rate': 0.53,
        'profit_factor': 1.4,
        'annual_return': 0.12,
        'stability_score': 0.75
    }
    
    # Validate in different market conditions
    for regime in [MarketRegime.BULL, MarketRegime.BEAR, MarketRegime.VOLATILE]:
        print(f"\n{'='*60}")
        print(f"Market Regime: {regime.value}")
        print('='*60)
        
        # Get adjusted targets
        targets = setter.get_targets(regime, ModelType.TREND_FOLLOWING)
        print(f"\nAdjusted Targets:")
        print(f"  Accuracy: {targets.target_accuracy:.1%}")
        print(f"  Sharpe: {targets.target_sharpe:.2f}")
        print(f"  Annual Return: {targets.target_annual_return:.1%}")
        
        # Generate report
        report = setter.generate_performance_report(
            metrics, regime, ModelType.TREND_FOLLOWING
        )
        print(report)