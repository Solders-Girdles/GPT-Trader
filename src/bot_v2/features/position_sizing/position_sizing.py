"""
Position Sizing Main Orchestrator - Complete Isolation

Main entry point for intelligent position sizing combining Kelly Criterion,
confidence adjustments, and regime-based scaling. Complete isolation maintained.
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
import logging

from ...errors import ValidationError, RiskLimitExceeded, log_error
from ...validation import (
    PositiveNumberValidator, PercentageValidator, 
    RangeValidator, validate_inputs
)
from ...config import get_config

from .types import (
    PositionSizeRequest, PositionSizeResponse, PositionSizingResult,
    SizingMethod, RiskParameters, KellyParameters, ConfidenceAdjustment,
    RegimeMultipliers, TradeStatistics
)
from .kelly import (
    kelly_criterion, fractional_kelly, kelly_position_value,
    kelly_risk_metrics, validate_kelly_inputs
)
from .confidence import (
    confidence_adjusted_size, validate_confidence_inputs
)
from .regime import (
    regime_adjusted_size, validate_regime_inputs
)

logger = logging.getLogger(__name__)


def _validate_position_request(request: PositionSizeRequest, field_name: str) -> PositionSizeRequest:
    """Validate position sizing request with comprehensive checks"""
    if not isinstance(request, PositionSizeRequest):
        raise ValidationError(f"{field_name} must be PositionSizeRequest", field=field_name)
    
    # Validate basic fields
    if not request.symbol:
        raise ValidationError("Symbol is required", field="symbol")
    
    PositiveNumberValidator()(request.current_price, "current_price")
    PositiveNumberValidator()(request.portfolio_value, "portfolio_value")
    PositiveNumberValidator()(request.strategy_multiplier, "strategy_multiplier")
    
    # Validate Kelly inputs if provided
    if _has_kelly_data(request):
        try:
            validate_kelly_inputs(request.win_rate, request.avg_win, request.avg_loss)
        except Exception as e:
            raise ValidationError(f"Kelly validation failed: {e}", field="kelly_inputs")
    
    # Validate confidence if provided
    if request.confidence is not None:
        PercentageValidator(as_decimal=True)(request.confidence, "confidence")
    
    return request


def _validate_kelly_safety(win_rate: float, avg_win: float, avg_loss: float):
    """Additional Kelly safety validations"""
    # Check for division by zero
    if abs(avg_loss) < 1e-10:
        raise ValidationError("Average loss too close to zero - division by zero risk", field="avg_loss")
    
    # Check for reasonable win rate
    if win_rate < 0.01 or win_rate > 0.99:
        raise ValidationError(f"Win rate {win_rate} outside reasonable bounds [0.01, 0.99]", field="win_rate")
    
    # Check expected value is positive
    expected_value = win_rate * avg_win + (1 - win_rate) * avg_loss
    if expected_value <= 0:
        raise ValidationError(f"Strategy has negative expected value: {expected_value:.6f}", field="expected_value")


@validate_inputs(
    request=_validate_position_request
)
def calculate_position_size(request: PositionSizeRequest) -> PositionSizeResponse:
    """
    Calculate intelligent position size using all available information.
    
    This is the main entry point for position sizing that combines:
    - Kelly Criterion (if trade statistics available)
    - Confidence-based adjustments (if confidence score available)
    - Market regime adjustments (if regime data available)
    
    Args:
        request: Position sizing request with all available data
        
    Returns:
        Position sizing response with recommendation and details
    """
    try:
        # Get configuration with defaults
        config = get_config('position_sizing')
        
        # Apply configuration overrides to risk parameters if needed
        if hasattr(request.risk_params, '__dict__'):
            for key, value in config.items():
                if hasattr(request.risk_params, key) and getattr(request.risk_params, key) is None:
                    setattr(request.risk_params, key, value)
        
        logger.info(f"Calculating position size for {request.symbol} using {request.method.value}")
        
    except Exception as e:
        error = ValidationError(f"Configuration error: {e}", field="config")
        log_error(error)
        return _create_error_response(request, [str(error)])
    
    # Calculate base position size using requested method
    if request.method == SizingMethod.INTELLIGENT:
        return _calculate_intelligent_size(request)
    elif request.method == SizingMethod.KELLY:
        return _calculate_kelly_size(request)
    elif request.method == SizingMethod.FRACTIONAL_KELLY:
        return _calculate_fractional_kelly_size(request)
    elif request.method == SizingMethod.CONFIDENCE_ADJUSTED:
        return _calculate_confidence_size(request)
    elif request.method == SizingMethod.REGIME_ADJUSTED:
        return _calculate_regime_size(request)
    elif request.method == SizingMethod.FIXED:
        return _calculate_fixed_size(request)
    else:
        return _create_error_response(request, [f"Unknown sizing method: {request.method}"])


def _calculate_intelligent_size(request: PositionSizeRequest) -> PositionSizeResponse:
    """Calculate position size using all available intelligence."""
    notes = []
    warnings = []
    
    try:
        # Start with base Kelly sizing if statistics available
        if _has_kelly_data(request):
            # Additional validation for Kelly inputs
            _validate_kelly_safety(request.win_rate, request.avg_win, request.avg_loss)
            
            base_size = fractional_kelly(
                request.win_rate, 
                request.avg_win, 
                request.avg_loss,
                request.risk_params.kelly_fraction
            )
            
            # Safety check for extreme Kelly values
            if base_size > request.risk_params.max_position_size:
                warnings.append(f"Kelly fraction {base_size:.4f} exceeds max position size, capping at {request.risk_params.max_position_size:.4f}")
                base_size = request.risk_params.max_position_size
            
            notes.append(f"Base Kelly sizing: {base_size:.4f}")
        else:
            # Fallback to fixed sizing
            base_size = request.risk_params.max_position_size * 0.5
            notes.append(f"No trade statistics, using conservative fixed size: {base_size:.4f}")
            warnings.append("No historical trade data available for Kelly calculation")
            
    except Exception as e:
        error = ValidationError(f"Kelly calculation failed: {e}", field="kelly_inputs")
        log_error(error)
        warnings.append(str(error))
        base_size = request.risk_params.max_position_size * 0.1  # Very conservative fallback
    
    # Apply confidence adjustment if available
    confidence_adjustment = 1.0
    if request.confidence is not None:
        try:
            # Validate confidence score
            PercentageValidator(as_decimal=True)(request.confidence, "confidence")
            
            if request.confidence >= request.risk_params.confidence_threshold:
                adj_params = ConfidenceAdjustment(confidence=request.confidence)
                adjusted_size, conf_explanation = confidence_adjusted_size(base_size, request.confidence, adj_params)
                confidence_adjustment = adjusted_size / base_size if base_size > 0 else 1.0
                base_size = adjusted_size
                notes.append(f"Confidence adjustment: {conf_explanation}")
            else:
                confidence_adjustment = 0.0
                base_size = 0.0
                warnings.append(f"Confidence {request.confidence:.2f} below threshold {request.risk_params.confidence_threshold:.2f}")
                
        except ValidationError as e:
            log_error(e)
            warnings.append(f"Invalid confidence score: {e.message}")
            # Continue with no confidence adjustment
    
    # Apply regime adjustment if available
    regime_adjustment = 1.0
    if request.market_regime:
        try:
            multipliers = RegimeMultipliers()
            adjusted_size, regime_explanation = regime_adjusted_size(base_size, request.market_regime, multipliers)
            regime_adjustment = adjusted_size / base_size if base_size > 0 else 1.0
            base_size = adjusted_size
            notes.append(f"Regime adjustment: {regime_explanation}")
        except Exception as e:
            error = ValidationError(f"Regime adjustment failed: {e}", field="market_regime")
            log_error(error)
            warnings.append(f"Regime adjustment failed: {e}")
    
    # Apply strategy-specific multiplier
    base_size *= request.strategy_multiplier
    if request.strategy_multiplier != 1.0:
        notes.append(f"Strategy multiplier: {request.strategy_multiplier:.2f}x")
    
    # Final safety checks before position calculation
    if base_size < 0:
        raise ValidationError("Position size cannot be negative", field="base_size", value=base_size)
    
    if base_size > request.risk_params.max_position_size:
        raise RiskLimitExceeded(
            "Position size exceeds maximum allowed",
            limit_type="max_position_size",
            limit_value=request.risk_params.max_position_size,
            current_value=base_size
        )
    
    # Convert to actual position
    try:
        position_value, share_count = kelly_position_value(
            request.portfolio_value,
            base_size,
            request.current_price,
            request.risk_params
        )
    except Exception as e:
        error = ValidationError(f"Position value calculation failed: {e}", field="position_calculation")
        log_error(error)
        raise error
    
    # Calculate final metrics
    position_size_pct = position_value / request.portfolio_value
    risk_pct = _estimate_position_risk(request, position_size_pct)
    
    return PositionSizeResponse(
        symbol=request.symbol,
        recommended_shares=share_count,
        recommended_value=position_value,
        position_size_pct=position_size_pct,
        risk_pct=risk_pct,
        method_used=SizingMethod.INTELLIGENT,
        kelly_fraction=base_size if _has_kelly_data(request) else None,
        confidence_adjustment=confidence_adjustment if request.confidence else None,
        regime_adjustment=regime_adjustment if request.market_regime else None,
        max_loss_estimate=position_value * abs(request.avg_loss or 0.05),
        expected_return=position_value * (request.avg_win or 0.03) * (request.win_rate or 0.5),
        calculation_notes=notes,
        warnings=warnings
    )


def _calculate_kelly_size(request: PositionSizeRequest) -> PositionSizeResponse:
    """Calculate position size using full Kelly Criterion."""
    if not _has_kelly_data(request):
        return _create_error_response(request, ["Kelly sizing requires win_rate, avg_win, and avg_loss"])
    
    kelly_size = kelly_criterion(request.win_rate, request.avg_win, request.avg_loss)
    position_value, share_count = kelly_position_value(
        request.portfolio_value, kelly_size, request.current_price, request.risk_params
    )
    
    position_size_pct = position_value / request.portfolio_value
    risk_pct = _estimate_position_risk(request, position_size_pct)
    
    return PositionSizeResponse(
        symbol=request.symbol,
        recommended_shares=share_count,
        recommended_value=position_value,
        position_size_pct=position_size_pct,
        risk_pct=risk_pct,
        method_used=SizingMethod.KELLY,
        kelly_fraction=kelly_size,
        max_loss_estimate=position_value * abs(request.avg_loss),
        expected_return=position_value * request.avg_win * request.win_rate,
        calculation_notes=[f"Full Kelly Criterion: {kelly_size:.4f}"]
    )


def _calculate_fractional_kelly_size(request: PositionSizeRequest) -> PositionSizeResponse:
    """Calculate position size using fractional Kelly Criterion."""
    if not _has_kelly_data(request):
        return _create_error_response(request, ["Fractional Kelly sizing requires win_rate, avg_win, and avg_loss"])
    
    kelly_size = fractional_kelly(
        request.win_rate, 
        request.avg_win, 
        request.avg_loss,
        request.risk_params.kelly_fraction
    )
    
    position_value, share_count = kelly_position_value(
        request.portfolio_value, kelly_size, request.current_price, request.risk_params
    )
    
    position_size_pct = position_value / request.portfolio_value
    risk_pct = _estimate_position_risk(request, position_size_pct)
    
    return PositionSizeResponse(
        symbol=request.symbol,
        recommended_shares=share_count,
        recommended_value=position_value,
        position_size_pct=position_size_pct,
        risk_pct=risk_pct,
        method_used=SizingMethod.FRACTIONAL_KELLY,
        kelly_fraction=kelly_size,
        max_loss_estimate=position_value * abs(request.avg_loss),
        expected_return=position_value * request.avg_win * request.win_rate,
        calculation_notes=[f"Fractional Kelly ({request.risk_params.kelly_fraction:.2f}): {kelly_size:.4f}"]
    )


def _calculate_confidence_size(request: PositionSizeRequest) -> PositionSizeResponse:
    """Calculate position size using confidence-based adjustment."""
    if request.confidence is None:
        return _create_error_response(request, ["Confidence sizing requires confidence score"])
    
    base_size = request.risk_params.max_position_size * 0.5  # Conservative base
    adj_params = ConfidenceAdjustment(confidence=request.confidence)
    
    adjusted_size, explanation = confidence_adjusted_size(base_size, request.confidence, adj_params)
    position_value, share_count = kelly_position_value(
        request.portfolio_value, adjusted_size, request.current_price, request.risk_params
    )
    
    position_size_pct = position_value / request.portfolio_value
    risk_pct = _estimate_position_risk(request, position_size_pct)
    
    return PositionSizeResponse(
        symbol=request.symbol,
        recommended_shares=share_count,
        recommended_value=position_value,
        position_size_pct=position_size_pct,
        risk_pct=risk_pct,
        method_used=SizingMethod.CONFIDENCE_ADJUSTED,
        confidence_adjustment=adjusted_size / base_size,
        max_loss_estimate=position_value * 0.05,  # Conservative estimate
        expected_return=position_value * 0.03 * request.confidence,
        calculation_notes=[explanation]
    )


def _calculate_regime_size(request: PositionSizeRequest) -> PositionSizeResponse:
    """Calculate position size using regime-based adjustment."""
    if not request.market_regime:
        return _create_error_response(request, ["Regime sizing requires market_regime"])
    
    base_size = request.risk_params.max_position_size * 0.5  # Conservative base
    multipliers = RegimeMultipliers()
    
    adjusted_size, explanation = regime_adjusted_size(base_size, request.market_regime, multipliers)
    position_value, share_count = kelly_position_value(
        request.portfolio_value, adjusted_size, request.current_price, request.risk_params
    )
    
    position_size_pct = position_value / request.portfolio_value
    risk_pct = _estimate_position_risk(request, position_size_pct)
    
    return PositionSizeResponse(
        symbol=request.symbol,
        recommended_shares=share_count,
        recommended_value=position_value,
        position_size_pct=position_size_pct,
        risk_pct=risk_pct,
        method_used=SizingMethod.REGIME_ADJUSTED,
        regime_adjustment=adjusted_size / base_size,
        max_loss_estimate=position_value * 0.05,  # Conservative estimate
        expected_return=position_value * 0.03,  # Conservative estimate
        calculation_notes=[explanation]
    )


def _calculate_fixed_size(request: PositionSizeRequest) -> PositionSizeResponse:
    """Calculate fixed position size."""
    fixed_size = request.risk_params.max_position_size * 0.3  # Conservative fixed size
    position_value, share_count = kelly_position_value(
        request.portfolio_value, fixed_size, request.current_price, request.risk_params
    )
    
    position_size_pct = position_value / request.portfolio_value
    risk_pct = _estimate_position_risk(request, position_size_pct)
    
    return PositionSizeResponse(
        symbol=request.symbol,
        recommended_shares=share_count,
        recommended_value=position_value,
        position_size_pct=position_size_pct,
        risk_pct=risk_pct,
        method_used=SizingMethod.FIXED,
        max_loss_estimate=position_value * 0.05,
        expected_return=position_value * 0.03,
        calculation_notes=[f"Fixed sizing: {fixed_size:.4f}"]
    )


def calculate_portfolio_allocation(requests: List[PositionSizeRequest]) -> PositionSizingResult:
    """
    Calculate position sizes for multiple assets with portfolio-level constraints.
    
    Args:
        requests: List of position sizing requests for different assets
        
    Returns:
        Portfolio-level position sizing result
    """
    if not requests:
        return PositionSizingResult(
            primary=_create_error_response(
                PositionSizeRequest("", 0, 0, ""), ["No requests provided"]
            )
        )
    
    # Calculate individual position sizes
    individual_responses = []
    total_portfolio_value = requests[0].portfolio_value  # Assume same portfolio
    
    for request in requests:
        response = calculate_position_size(request)
        individual_responses.append(response)
    
    # Check portfolio-level constraints
    total_position_pct = sum(resp.position_size_pct for resp in individual_responses)
    total_risk_pct = sum(resp.risk_pct for resp in individual_responses)
    
    # Apply portfolio scaling if needed
    max_portfolio_allocation = 0.8  # Don't use more than 80% of portfolio
    if total_position_pct > max_portfolio_allocation:
        scale_factor = max_portfolio_allocation / total_position_pct
        
        # Scale down all positions proportionally
        scaled_responses = []
        for resp in individual_responses:
            scaled_value = resp.recommended_value * scale_factor
            scaled_shares = int(scaled_value / (resp.recommended_value / max(1, resp.recommended_shares)))
            
            scaled_resp = PositionSizeResponse(
                symbol=resp.symbol,
                recommended_shares=scaled_shares,
                recommended_value=scaled_value,
                position_size_pct=resp.position_size_pct * scale_factor,
                risk_pct=resp.risk_pct * scale_factor,
                method_used=resp.method_used,
                kelly_fraction=resp.kelly_fraction,
                confidence_adjustment=resp.confidence_adjustment,
                regime_adjustment=resp.regime_adjustment,
                max_loss_estimate=resp.max_loss_estimate * scale_factor,
                expected_return=resp.expected_return * scale_factor,
                calculation_notes=resp.calculation_notes + [f"Portfolio scaled by {scale_factor:.3f}"],
                warnings=resp.warnings
            )
            scaled_responses.append(scaled_resp)
        
        individual_responses = scaled_responses
    
    # Create portfolio impact metrics
    portfolio_impact = {
        'total_allocation_pct': sum(resp.position_size_pct for resp in individual_responses),
        'total_risk_pct': sum(resp.risk_pct for resp in individual_responses),
        'expected_portfolio_return': sum(resp.expected_return for resp in individual_responses),
        'max_portfolio_loss': sum(resp.max_loss_estimate for resp in individual_responses),
        'num_positions': len([resp for resp in individual_responses if resp.recommended_shares > 0])
    }
    
    # Primary response is the largest position or first if all similar
    primary = max(individual_responses, key=lambda x: x.recommended_value)
    
    return PositionSizingResult(
        primary=primary,
        alternatives=individual_responses,
        portfolio_impact=portfolio_impact
    )


def _has_kelly_data(request: PositionSizeRequest) -> bool:
    """Check if request has data needed for Kelly calculation."""
    return all([
        request.win_rate is not None,
        request.avg_win is not None,
        request.avg_loss is not None
    ])


def _estimate_position_risk(request: PositionSizeRequest, position_size_pct: float) -> float:
    """Estimate risk percentage for a position."""
    if request.avg_loss:
        return position_size_pct * abs(request.avg_loss)
    elif request.volatility:
        return position_size_pct * request.volatility * 1.5  # Conservative estimate
    else:
        return position_size_pct * 0.05  # Default 5% risk estimate


def _estimate_portfolio_risk(position_size_pct: float, avg_loss: Optional[float], 
                           volatility: Optional[float]) -> float:
    """Estimate portfolio risk with bounds checking"""
    try:
        if avg_loss:
            risk = position_size_pct * abs(avg_loss)
        elif volatility:
            risk = position_size_pct * volatility * 1.5  # Conservative estimate
        else:
            risk = position_size_pct * 0.05  # Default 5% risk estimate
        
        # Safety bounds
        return max(0.0, min(1.0, risk))
        
    except Exception as e:
        logger.warning(f"Risk estimation failed: {e}, using conservative default")
        return position_size_pct * 0.05


def _create_error_response(request: PositionSizeRequest, errors: List[str]) -> PositionSizeResponse:
    """Create error response for invalid requests."""
    return PositionSizeResponse(
        symbol=request.symbol or "UNKNOWN",
        recommended_shares=0,
        recommended_value=0.0,
        position_size_pct=0.0,
        risk_pct=0.0,
        method_used=request.method,
        warnings=errors
    )