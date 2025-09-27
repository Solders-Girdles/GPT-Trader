"""
Security Validator for Bot V2 Trading System

Implements input validation, rate limiting, and trading-specific security checks
to protect against injection attacks and ensure safe trading operations.
"""

import re
import time
import hashlib
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, List, Tuple
from dataclasses import dataclass
from collections import defaultdict, deque
from threading import Lock
import logging
from decimal import Decimal, InvalidOperation

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of validation check"""
    is_valid: bool
    errors: List[str]
    sanitized_value: Any = None


@dataclass
class RateLimitConfig:
    """Rate limit configuration"""
    requests: int
    period: int  # seconds
    burst: Optional[int] = None


class SecurityValidator:
    """
    Comprehensive security validation for trading operations.
    Prevents injection attacks and enforces trading limits.
    """
    
    # Regex patterns for validation
    PATTERNS = {
        'symbol': r'^[A-Z0-9]{1,10}(-[A-Z0-9]{2,10})?$',  # Trading symbols (e.g., AAPL, BTC-USD, BTC-PERP)
        'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
        'alphanumeric': r'^[a-zA-Z0-9]+$',
        'numeric': r'^-?\d+(\.\d+)?$',
        'sql_injection': r'(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|CREATE|ALTER|EXEC|EXECUTE|SCRIPT|JAVASCRIPT|EVAL)\b)',
        'xss_tags': r'<[^>]*>',
        'path_traversal': r'\.\.[\\/]'
    }
    
    # Trading limits
    TRADING_LIMITS = {
        'max_position_size': 0.05,  # 5% of portfolio
        'max_daily_loss': 0.02,      # 2% daily loss limit
        'max_leverage': 2.0,         # 2:1 leverage
        'max_concentration': 0.20,   # 20% in single symbol
        'max_orders_per_minute': 5,
        'min_order_value': 1.0,      # $1 minimum
        'max_order_value': 100000.0  # $100k maximum
    }
    
    # Rate limit configurations
    RATE_LIMITS = {
        'api_calls': RateLimitConfig(100, 60),  # 100/minute
        'order_submissions': RateLimitConfig(10, 60, burst=3),  # 10/minute with burst
        'login_attempts': RateLimitConfig(5, 3600),  # 5/hour
        'data_requests': RateLimitConfig(1000, 3600)  # 1000/hour
    }
    
    def __init__(self):
        self._lock = Lock()
        self._rate_limiters = defaultdict(lambda: defaultdict(deque))
        self._blocked_ips = set()
        self._suspicious_activity = defaultdict(int)
    
    def sanitize_string(self, input_str: str, max_length: int = 255) -> ValidationResult:
        """
        Sanitize string input and check for injection attempts.
        
        Args:
            input_str: Input string to sanitize
            max_length: Maximum allowed length
            
        Returns:
            ValidationResult with sanitized value
        """
        errors = []
        
        if not input_str:
            return ValidationResult(False, ["Input cannot be empty"])
        
        # Check length
        if len(input_str) > max_length:
            errors.append(f"Input exceeds maximum length of {max_length}")
            input_str = input_str[:max_length]
        
        # Check for SQL injection patterns
        if re.search(self.PATTERNS['sql_injection'], input_str, re.IGNORECASE):
            errors.append("Potential SQL injection detected")
            return ValidationResult(False, errors)
        
        # Check for XSS attempts
        if re.search(self.PATTERNS['xss_tags'], input_str):
            errors.append("HTML tags not allowed")
            # Strip HTML tags
            input_str = re.sub(self.PATTERNS['xss_tags'], '', input_str)
        
        # Check for path traversal
        if re.search(self.PATTERNS['path_traversal'], input_str):
            errors.append("Path traversal attempt detected")
            return ValidationResult(False, errors)
        
        # Escape special characters
        sanitized = input_str.replace("'", "''").replace('"', '""').strip()
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            sanitized_value=sanitized
        )
    
    def validate_symbol(self, symbol: str) -> ValidationResult:
        """Validate trading symbol"""
        errors = []
        
        if not symbol:
            return ValidationResult(False, ["Symbol cannot be empty"])
        
        # Check format
        if not re.match(self.PATTERNS['symbol'], symbol):
            errors.append("Invalid symbol format")
        
        # Check against blocklist (simplified)
        blocked_symbols = {'TEST', 'DEBUG', 'HACK'}
        if symbol in blocked_symbols:
            errors.append("Symbol is blocked")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            sanitized_value=symbol.upper() if not errors else None
        )
    
    def validate_numeric(self, value: Any, min_val: float = None, max_val: float = None) -> ValidationResult:
        """Validate numeric input"""
        errors = []
        
        try:
            # Convert to Decimal for precise financial calculations
            num_value = Decimal(str(value))
            
            if min_val is not None and num_value < Decimal(str(min_val)):
                errors.append(f"Value must be at least {min_val}")
            
            if max_val is not None and num_value > Decimal(str(max_val)):
                errors.append(f"Value must not exceed {max_val}")
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                sanitized_value=float(num_value) if not errors else None
            )
            
        except (InvalidOperation, ValueError):
            return ValidationResult(False, ["Invalid numeric value"])
    
    def validate_order_request(self, order: Dict[str, Any], account_value: float) -> ValidationResult:
        """
        Validate trading order request.
        
        Args:
            order: Order details
            account_value: Current account value
            
        Returns:
            ValidationResult
        """
        errors = []
        
        # Validate symbol
        symbol_result = self.validate_symbol(order.get('symbol', ''))
        if not symbol_result.is_valid:
            errors.extend(symbol_result.errors)
        
        # Validate quantity
        quantity = order.get('quantity', 0)
        qty_result = self.validate_numeric(quantity, min_val=0.001, max_val=1000000)
        if not qty_result.is_valid:
            errors.extend(qty_result.errors)
        
        # Validate price if limit order
        if order.get('order_type') == 'limit':
            price = order.get('price', 0)
            price_result = self.validate_numeric(price, min_val=0.01, max_val=1000000)
            if not price_result.is_valid:
                errors.extend(price_result.errors)
        
        # Check position size limits
        order_value = quantity * order.get('price', 100)  # Estimate if market order
        
        if order_value < self.TRADING_LIMITS['min_order_value']:
            errors.append(f"Order value below minimum: ${self.TRADING_LIMITS['min_order_value']}")
        
        if order_value > self.TRADING_LIMITS['max_order_value']:
            errors.append(f"Order value exceeds maximum: ${self.TRADING_LIMITS['max_order_value']}")
        
        # Check position concentration
        position_pct = order_value / account_value if account_value > 0 else 1.0
        if position_pct > self.TRADING_LIMITS['max_position_size']:
            errors.append(f"Position size exceeds {self.TRADING_LIMITS['max_position_size']*100}% limit")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            sanitized_value=order if not errors else None
        )
    
    def check_rate_limit(self, identifier: str, limit_type: str) -> Tuple[bool, Optional[str]]:
        """
        Check if request is within rate limits.
        
        Args:
            identifier: User ID or IP address
            limit_type: Type of rate limit to check
            
        Returns:
            Tuple of (allowed, error_message)
        """
        if identifier in self._blocked_ips:
            return False, "IP temporarily blocked due to suspicious activity"
        
        if limit_type not in self.RATE_LIMITS:
            return True, None
        
        config = self.RATE_LIMITS[limit_type]
        now = time.time()
        
        with self._lock:
            # Get request history
            history = self._rate_limiters[limit_type][identifier]
            
            # Remove old entries
            cutoff = now - config.period
            while history and history[0] < cutoff:
                history.popleft()
            
            # Check limit
            if len(history) >= config.requests:
                # Check for suspicious activity
                self._suspicious_activity[identifier] += 1
                if self._suspicious_activity[identifier] > 10:
                    self._blocked_ips.add(identifier)
                    logger.warning(f"Blocked {identifier} for excessive rate limit violations")
                
                return False, f"Rate limit exceeded: {config.requests} requests per {config.period} seconds"
            
            # Add current request
            history.append(now)
            
            return True, None
    
    def check_trading_hours(self, symbol: str, timestamp: Optional[datetime] = None) -> ValidationResult:
        """Check if trading is allowed at current time"""
        errors = []
        timestamp = timestamp or datetime.now()
        
        # Market hours (simplified - NYSE: 9:30 AM - 4:00 PM ET)
        hour = timestamp.hour
        minute = timestamp.minute
        weekday = timestamp.weekday()
        
        # Check weekend
        if weekday >= 5:  # Saturday = 5, Sunday = 6
            errors.append("Market closed on weekends")
        
        # Check market hours (simplified, not accounting for timezone)
        elif hour < 9 or (hour == 9 and minute < 30) or hour >= 16:
            errors.append("Outside market hours (9:30 AM - 4:00 PM ET)")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors
        )
    
    def detect_suspicious_activity(self, user_id: str, activity: Dict[str, Any]) -> bool:
        """
        Detect potentially suspicious trading activity.
        
        Args:
            user_id: User identifier
            activity: Activity details
            
        Returns:
            True if suspicious
        """
        suspicious_indicators = 0
        
        # Rapid-fire orders
        if activity.get('orders_per_minute', 0) > 10:
            suspicious_indicators += 1
            logger.warning(f"Rapid-fire orders detected for {user_id}")
        
        # Unusual order size
        avg_size = activity.get('average_order_size', 0)
        current_size = activity.get('current_order_size', 0)
        if avg_size > 0 and current_size > avg_size * 5:
            suspicious_indicators += 1
            logger.warning(f"Unusual order size detected for {user_id}")
        
        # Pattern detection (simplified)
        if activity.get('pattern_score', 0) > 0.8:
            suspicious_indicators += 1
            logger.warning(f"Suspicious pattern detected for {user_id}")
        
        return suspicious_indicators >= 2
    
    def validate_request(self, request: Dict[str, Any]) -> ValidationResult:
        """
        Comprehensive request validation.
        
        Args:
            request: Request data
            
        Returns:
            ValidationResult
        """
        errors = []
        
        # Check required fields
        required_fields = ['action', 'timestamp']
        for field in required_fields:
            if field not in request:
                errors.append(f"Missing required field: {field}")
        
        # Validate each field type
        if 'action' in request:
            action_result = self.sanitize_string(request['action'], max_length=50)
            if not action_result.is_valid:
                errors.extend(action_result.errors)
        
        # Check request size
        import sys
        request_size = sys.getsizeof(request)
        if request_size > 1048576:  # 1MB
            errors.append("Request size exceeds 1MB limit")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            sanitized_value=request if not errors else None
        )
    
    def clear_rate_limits(self, identifier: Optional[str] = None):
        """Clear rate limit history"""
        with self._lock:
            if identifier:
                for limit_type in self._rate_limiters:
                    if identifier in self._rate_limiters[limit_type]:
                        del self._rate_limiters[limit_type][identifier]
            else:
                self._rate_limiters.clear()


# Global instance
_validator = None


def get_validator() -> SecurityValidator:
    """Get the global validator instance"""
    global _validator
    if _validator is None:
        _validator = SecurityValidator()
    return _validator


# Convenience functions
def validate_order(order: Dict, account_value: float) -> ValidationResult:
    """Validate trading order"""
    return get_validator().validate_order_request(order, account_value)


def check_rate_limit(identifier: str, limit_type: str) -> Tuple[bool, Optional[str]]:
    """Check rate limit"""
    return get_validator().check_rate_limit(identifier, limit_type)


def sanitize_input(input_str: str, max_length: int = 255) -> ValidationResult:
    """Sanitize string input"""
    return get_validator().sanitize_string(input_str, max_length)
