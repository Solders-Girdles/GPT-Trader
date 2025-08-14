"""
Comprehensive Input Validation Module for GPT-Trader

Provides centralized validation for all user inputs across CLI, API, and configuration.
Implements defense-in-depth with multiple validation layers.
"""

import re
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator


class InputType(Enum):
    """Types of input requiring validation"""

    SYMBOL = "symbol"
    DATE = "date"
    NUMERIC = "numeric"
    PERCENTAGE = "percentage"
    PATH = "path"
    EMAIL = "email"
    API_KEY = "api_key"
    STRATEGY_NAME = "strategy_name"
    ORDER_TYPE = "order_type"
    TIME_FRAME = "time_frame"


class ValidationRules:
    """Centralized validation rules and patterns"""

    # Regex patterns
    SYMBOL_PATTERN = re.compile(r"^[A-Z]{1,5}(-[A-Z]{1,5})?$")  # Stock symbols
    EMAIL_PATTERN = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
    API_KEY_PATTERN = re.compile(r"^[A-Za-z0-9_\-]{32,128}$")  # Alphanumeric API keys
    STRATEGY_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_\-]{3,50}$")
    SAFE_PATH_PATTERN = re.compile(r"^[a-zA-Z0-9_\-/\\.]+$")  # No special chars in paths

    # Constraints
    MIN_PRICE = Decimal("0.01")
    MAX_PRICE = Decimal("1000000.00")
    MIN_QUANTITY = 1
    MAX_QUANTITY = 1000000
    MIN_PERCENTAGE = Decimal("0.0")
    MAX_PERCENTAGE = Decimal("100.0")

    # Date constraints
    MIN_DATE = datetime(2000, 1, 1)
    MAX_DATE = datetime.now() + timedelta(days=365)  # Max 1 year in future

    # Allowed values
    ALLOWED_ORDER_TYPES = ["market", "limit", "stop", "stop_limit"]
    ALLOWED_TIME_FRAMES = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w", "1M"]
    ALLOWED_STRATEGIES = ["demo_ma", "trend_breakout", "mean_reversion", "momentum"]


class InputValidator:
    """Main input validation class"""

    def __init__(self, strict_mode: bool = True):
        """
        Initialize validator

        Args:
            strict_mode: If True, reject any suspicious input. If False, sanitize when possible.
        """
        self.strict_mode = strict_mode
        self.rules = ValidationRules()

    def validate_symbol(self, symbol: str) -> str:
        """
        Validate stock/crypto symbol

        Args:
            symbol: Trading symbol to validate

        Returns:
            Validated and normalized symbol

        Raises:
            ValueError: If symbol is invalid
        """
        if not symbol:
            raise ValueError("Symbol cannot be empty")

        # Normalize to uppercase
        symbol = symbol.upper().strip()

        # Check pattern
        if not self.rules.SYMBOL_PATTERN.match(symbol):
            raise ValueError(
                f"Invalid symbol format: {symbol}. "
                "Must be 1-5 uppercase letters, optionally followed by hyphen and 1-5 letters."
            )

        # Additional checks for suspicious patterns
        if any(char in symbol for char in ["'", '"', ";", "--", "/*", "*/"]):
            raise ValueError(f"Symbol contains suspicious characters: {symbol}")

        return symbol

    def validate_date(self, date_str: str) -> datetime:
        """
        Validate and parse date string

        Args:
            date_str: Date string in YYYY-MM-DD format

        Returns:
            Parsed datetime object

        Raises:
            ValueError: If date is invalid or out of range
        """
        if not date_str:
            raise ValueError("Date cannot be empty")

        # Remove any extra whitespace
        date_str = date_str.strip()

        # Parse date
        try:
            date = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError as e:
            raise ValueError(f"Invalid date format: {date_str}. Use YYYY-MM-DD format.") from e

        # Check range
        if date < self.rules.MIN_DATE:
            raise ValueError(
                f"Date {date_str} is before minimum allowed date {self.rules.MIN_DATE.date()}"
            )

        if date > self.rules.MAX_DATE:
            raise ValueError(
                f"Date {date_str} is after maximum allowed date {self.rules.MAX_DATE.date()}"
            )

        return date

    def validate_price(self, price: str | float | Decimal) -> Decimal:
        """
        Validate price value

        Args:
            price: Price value to validate

        Returns:
            Validated price as Decimal

        Raises:
            ValueError: If price is invalid
        """
        try:
            price_decimal = Decimal(str(price))
        except (InvalidOperation, ValueError) as e:
            raise ValueError(f"Invalid price format: {price}") from e

        if price_decimal <= 0:
            raise ValueError(f"Price must be positive, got: {price}")

        if price_decimal < self.rules.MIN_PRICE:
            raise ValueError(f"Price {price} is below minimum {self.rules.MIN_PRICE}")

        if price_decimal > self.rules.MAX_PRICE:
            raise ValueError(f"Price {price} exceeds maximum {self.rules.MAX_PRICE}")

        return price_decimal

    def validate_quantity(self, quantity: str | int) -> int:
        """
        Validate quantity/volume

        Args:
            quantity: Quantity to validate

        Returns:
            Validated quantity as integer

        Raises:
            ValueError: If quantity is invalid
        """
        try:
            qty = int(quantity)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid quantity format: {quantity}") from e

        if qty < self.rules.MIN_QUANTITY:
            raise ValueError(f"Quantity {qty} is below minimum {self.rules.MIN_QUANTITY}")

        if qty > self.rules.MAX_QUANTITY:
            raise ValueError(f"Quantity {qty} exceeds maximum {self.rules.MAX_QUANTITY}")

        return qty

    def validate_percentage(self, percentage: str | float | Decimal) -> Decimal:
        """
        Validate percentage value (0-100)

        Args:
            percentage: Percentage to validate

        Returns:
            Validated percentage as Decimal

        Raises:
            ValueError: If percentage is invalid
        """
        try:
            pct = Decimal(str(percentage))
        except (InvalidOperation, ValueError) as e:
            raise ValueError(f"Invalid percentage format: {percentage}") from e

        if pct < self.rules.MIN_PERCENTAGE:
            raise ValueError(f"Percentage {pct} cannot be negative")

        if pct > self.rules.MAX_PERCENTAGE:
            raise ValueError(f"Percentage {pct} cannot exceed 100")

        return pct

    def validate_path(self, path_str: str, must_exist: bool = False) -> Path:
        """
        Validate file/directory path

        Args:
            path_str: Path string to validate
            must_exist: If True, verify path exists

        Returns:
            Validated Path object

        Raises:
            ValueError: If path is invalid or contains suspicious patterns
        """
        if not path_str:
            raise ValueError("Path cannot be empty")

        # Check for path traversal attempts
        if any(pattern in path_str for pattern in ["../", "..", "~/", "$", "`", "|", ";"]):
            raise ValueError(f"Path contains suspicious pattern: {path_str}")

        # Create Path object
        try:
            path = Path(path_str).resolve()
        except (ValueError, OSError) as e:
            raise ValueError(f"Invalid path: {path_str}") from e

        # Check if path is within allowed directories (configurable)
        # This prevents access to system files
        allowed_roots = [Path.cwd(), Path.home() / "trading_data"]
        if not any(path.is_relative_to(root) for root in allowed_roots if root.exists()):
            if self.strict_mode:
                raise ValueError(f"Path {path} is outside allowed directories")

        if must_exist and not path.exists():
            raise ValueError(f"Path does not exist: {path}")

        return path

    def validate_email(self, email: str) -> str:
        """
        Validate email address

        Args:
            email: Email address to validate

        Returns:
            Validated email address

        Raises:
            ValueError: If email is invalid
        """
        if not email:
            raise ValueError("Email cannot be empty")

        email = email.strip().lower()

        if not self.rules.EMAIL_PATTERN.match(email):
            raise ValueError(f"Invalid email format: {email}")

        # Check for suspicious patterns
        if any(char in email for char in ["'", '"', ";", "--", "/*"]):
            raise ValueError(f"Email contains suspicious characters: {email}")

        return email

    def validate_api_key(self, api_key: str) -> str:
        """
        Validate API key format

        Args:
            api_key: API key to validate

        Returns:
            Validated API key

        Raises:
            ValueError: If API key format is invalid
        """
        if not api_key:
            raise ValueError("API key cannot be empty")

        api_key = api_key.strip()

        if not self.rules.API_KEY_PATTERN.match(api_key):
            raise ValueError(
                "Invalid API key format. "
                "Must be 32-128 characters containing only letters, numbers, underscores, and hyphens."
            )

        return api_key

    def validate_strategy_name(self, name: str) -> str:
        """
        Validate strategy name

        Args:
            name: Strategy name to validate

        Returns:
            Validated strategy name

        Raises:
            ValueError: If name is invalid
        """
        if not name:
            raise ValueError("Strategy name cannot be empty")

        name = name.strip().lower()

        if not self.rules.STRATEGY_NAME_PATTERN.match(name):
            raise ValueError(
                f"Invalid strategy name: {name}. "
                "Must be 3-50 characters containing only letters, numbers, underscores, and hyphens."
            )

        return name

    def validate_order_type(self, order_type: str) -> str:
        """
        Validate order type

        Args:
            order_type: Order type to validate

        Returns:
            Validated order type

        Raises:
            ValueError: If order type is invalid
        """
        if not order_type:
            raise ValueError("Order type cannot be empty")

        order_type = order_type.strip().lower()

        if order_type not in self.rules.ALLOWED_ORDER_TYPES:
            raise ValueError(
                f"Invalid order type: {order_type}. "
                f"Must be one of: {', '.join(self.rules.ALLOWED_ORDER_TYPES)}"
            )

        return order_type

    def validate_time_frame(self, time_frame: str) -> str:
        """
        Validate trading time frame

        Args:
            time_frame: Time frame to validate

        Returns:
            Validated time frame

        Raises:
            ValueError: If time frame is invalid
        """
        if not time_frame:
            raise ValueError("Time frame cannot be empty")

        time_frame = time_frame.strip().lower()

        if time_frame not in self.rules.ALLOWED_TIME_FRAMES:
            raise ValueError(
                f"Invalid time frame: {time_frame}. "
                f"Must be one of: {', '.join(self.rules.ALLOWED_TIME_FRAMES)}"
            )

        return time_frame

    def sanitize_string(self, input_str: str, max_length: int = 1000) -> str:
        """
        Sanitize generic string input

        Args:
            input_str: String to sanitize
            max_length: Maximum allowed length

        Returns:
            Sanitized string

        Raises:
            ValueError: If string contains dangerous patterns
        """
        if not input_str:
            return ""

        # Truncate if too long
        if len(input_str) > max_length:
            input_str = input_str[:max_length]

        # Check for SQL injection patterns
        sql_patterns = [
            "select ",
            "insert ",
            "update ",
            "delete ",
            "drop ",
            "union ",
            "exec ",
            "execute ",
            "--",
            "/*",
            "*/",
            "xp_",
            "sp_",
        ]

        input_lower = input_str.lower()
        for pattern in sql_patterns:
            if pattern in input_lower:
                if self.strict_mode:
                    raise ValueError(f"Input contains suspicious SQL pattern: {pattern}")
                else:
                    # Remove the pattern
                    input_str = input_str.replace(pattern, "")

        # Check for script injection patterns
        script_patterns = ["<script", "</script", "javascript:", "onerror=", "onclick="]
        for pattern in script_patterns:
            if pattern in input_lower:
                if self.strict_mode:
                    raise ValueError(f"Input contains suspicious script pattern: {pattern}")
                else:
                    # Remove the pattern
                    input_str = input_str.replace(pattern, "")

        # Remove null bytes
        input_str = input_str.replace("\x00", "")

        return input_str.strip()


# Pydantic models for API input validation
class TradingOrderRequest(BaseModel):
    """Validated trading order request"""

    model_config = ConfigDict(str_strip_whitespace=True)

    symbol: str = Field(..., min_length=1, max_length=10)
    order_type: str = Field(..., pattern="^(market|limit|stop|stop_limit)$")
    quantity: int = Field(..., gt=0, le=1000000)
    price: Decimal | None = Field(None, gt=0, le=1000000)
    stop_price: Decimal | None = Field(None, gt=0, le=1000000)

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        validator = InputValidator()
        return validator.validate_symbol(v)

    @field_validator("price", "stop_price")
    @classmethod
    def validate_prices(cls, v: Decimal | None) -> Decimal | None:
        if v is not None:
            validator = InputValidator()
            return validator.validate_price(v)
        return v


class BacktestRequest(BaseModel):
    """Validated backtest request"""

    model_config = ConfigDict(str_strip_whitespace=True)

    strategy: str = Field(..., min_length=3, max_length=50)
    symbol: str = Field(..., min_length=1, max_length=10)
    start_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")
    end_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")
    initial_capital: Decimal = Field(..., gt=0, le=10000000)

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        validator = InputValidator()
        return validator.validate_strategy_name(v)

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        validator = InputValidator()
        return validator.validate_symbol(v)

    @field_validator("start_date", "end_date")
    @classmethod
    def validate_dates(cls, v: str) -> str:
        validator = InputValidator()
        date = validator.validate_date(v)
        return date.strftime("%Y-%m-%d")


# Singleton for application-wide access
_validator: InputValidator | None = None


def get_validator(strict_mode: bool = True) -> InputValidator:
    """Get or create the input validator instance"""
    global _validator
    if _validator is None or _validator.strict_mode != strict_mode:
        _validator = InputValidator(strict_mode=strict_mode)
    return _validator


# Convenience functions for common validations
def validate_trading_symbol(symbol: str) -> str:
    """Validate a trading symbol"""
    return get_validator().validate_symbol(symbol)


def validate_date_range(start_date: str, end_date: str) -> tuple[datetime, datetime]:
    """Validate a date range for backtesting"""
    validator = get_validator()
    start = validator.validate_date(start_date)
    end = validator.validate_date(end_date)

    if start >= end:
        raise ValueError(f"Start date {start_date} must be before end date {end_date}")

    return start, end


def validate_portfolio_allocation(allocations: dict[str, float]) -> dict[str, Decimal]:
    """Validate portfolio allocation percentages"""
    validator = get_validator()
    validated = {}
    total = Decimal("0")

    for symbol, percentage in allocations.items():
        sym = validator.validate_symbol(symbol)
        pct = validator.validate_percentage(percentage)
        validated[sym] = pct
        total += pct

    if abs(total - Decimal("100.0")) > Decimal("0.01"):
        raise ValueError(f"Portfolio allocations must sum to 100%, got {total}%")

    return validated
