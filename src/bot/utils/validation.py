"""
Consolidated validation utilities for GPT-Trader.

This module provides centralized validation functions that were previously
duplicated across multiple files in the codebase.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


class DateValidator:
    """Centralized date validation utilities."""

    @staticmethod
    def validate_date(date_str: str) -> datetime:
        """Validate and parse date string in YYYY-MM-DD format.

        Args:
            date_str: Date string in YYYY-MM-DD format

        Returns:
            Parsed datetime object

        Raises:
            ValueError: If date format is invalid
        """
        try:
            return datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError as e:
            raise ValueError(f"Invalid date format: {date_str}. Use YYYY-MM-DD format.") from e

    @staticmethod
    def validate_date_range(start: str, end: str) -> tuple[datetime, datetime]:
        """Validate date range.

        Args:
            start: Start date string
            end: End date string

        Returns:
            Tuple of (start_date, end_date)

        Raises:
            ValueError: If dates are invalid or range is incorrect
        """
        start_date = DateValidator.validate_date(start)
        end_date = DateValidator.validate_date(end)

        if start_date >= end_date:
            raise ValueError("Start date must be before end date")

        return start_date, end_date

    @staticmethod
    def parse_date_safe(date_str: str) -> datetime | None:
        """Safely parse date string, returning None on error.

        Args:
            date_str: Date string to parse

        Returns:
            Parsed datetime or None if invalid
        """
        try:
            return DateValidator.validate_date(date_str)
        except ValueError:
            return None


class SymbolValidator:
    """Centralized symbol validation utilities."""

    # Pattern for valid stock symbols (alphanumeric, may contain dots/hyphens)
    SYMBOL_PATTERN = re.compile(r"^[A-Z0-9.-]{1,10}$")

    @staticmethod
    def validate_symbol(symbol: str) -> str:
        """Validate stock symbol.

        Args:
            symbol: Stock symbol to validate

        Returns:
            Normalized symbol (uppercase, stripped)

        Raises:
            ValueError: If symbol is invalid
        """
        if not symbol or not symbol.strip():
            raise ValueError("Symbol cannot be empty")

        symbol = symbol.strip().upper()

        if not SymbolValidator.SYMBOL_PATTERN.match(symbol):
            raise ValueError(f"Invalid symbol format: {symbol}")

        return symbol

    @staticmethod
    def validate_symbols(symbols: list[str]) -> list[str]:
        """Validate list of symbols.

        Args:
            symbols: List of symbol strings

        Returns:
            List of validated symbols (deduplicated)

        Raises:
            ValueError: If no valid symbols found
        """
        if not symbols:
            raise ValueError("No symbols provided")

        validated = []
        invalid = []

        for symbol in symbols:
            try:
                validated.append(SymbolValidator.validate_symbol(symbol))
            except ValueError:
                invalid.append(symbol)

        if not validated:
            raise ValueError(f"No valid symbols found. Invalid: {invalid}")

        # Remove duplicates while preserving order
        seen = set()
        result = []
        for symbol in validated:
            if symbol not in seen:
                seen.add(symbol)
                result.append(symbol)

        return result

    @staticmethod
    def normalize_symbol(symbol: str) -> str:
        """Normalize symbol without validation (for internal use).

        Args:
            symbol: Symbol to normalize

        Returns:
            Normalized symbol
        """
        return symbol.strip().upper()


class FileValidator:
    """File and path validation utilities."""

    @staticmethod
    def validate_file_path(path: str) -> Path:
        """Validate file path exists and is accessible.

        Args:
            path: File path string

        Returns:
            Path object

        Raises:
            ValueError: If path is invalid or inaccessible
        """
        if not path or not path.strip():
            raise ValueError("Path cannot be empty")

        file_path = Path(path.strip())

        if not file_path.exists():
            raise ValueError(f"Path does not exist: {path}")

        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {path}")

        if not file_path.stat().st_size > 0:
            raise ValueError(f"File is empty: {path}")

        return file_path

    @staticmethod
    def validate_csv_file(path: str, required_columns: list[str] | None = None) -> Path:
        """Validate CSV file and optionally check for required columns.

        Args:
            path: Path to CSV file
            required_columns: Optional list of required column names

        Returns:
            Path object

        Raises:
            ValueError: If file is invalid or missing required columns
        """
        file_path = FileValidator.validate_file_path(path)

        if not file_path.suffix.lower() == ".csv":
            raise ValueError(f"File must be a CSV: {path}")

        if required_columns:
            try:
                df = pd.read_csv(file_path, nrows=0)  # Just read headers
                missing_columns = set(required_columns) - set(df.columns)
                if missing_columns:
                    raise ValueError(f"CSV missing required columns: {missing_columns}")
            except Exception as e:
                raise ValueError(f"Error reading CSV file: {e}") from e

        return file_path


class DataFrameValidator:
    """DataFrame validation utilities."""

    @staticmethod
    def validate_daily_bars(df: pd.DataFrame, symbol: str) -> None:
        """Validate daily price bars DataFrame.

        Args:
            df: DataFrame with OHLC data
            symbol: Symbol name for error messages

        Raises:
            ValueError: If DataFrame is invalid
        """
        if df.empty:
            raise ValueError(f"{symbol}: empty price frame")

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(f"{symbol}: index must be DatetimeIndex")

        if not df.index.is_monotonic_increasing:
            raise ValueError(f"{symbol}: index not sorted ascending")

        if df.index.has_duplicates:
            dup_ct = int(df.index.duplicated().sum())
            raise ValueError(f"{symbol}: duplicate dates detected ({dup_ct})")

        needed = [c for c in ("Open", "High", "Low", "Close") if c in df.columns]
        if not needed:
            raise ValueError(f"{symbol}: missing price columns")

        if df[needed].isna().any().any():
            n = int(df[needed].isna().sum().sum())
            raise ValueError(f"{symbol}: NaNs in OHLC ({n})")

    @staticmethod
    def validate_required_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
        """Validate DataFrame has required columns.

        Args:
            df: DataFrame to validate
            required_columns: List of required column names

        Raises:
            ValueError: If required columns are missing
        """
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"DataFrame missing required columns: {missing_columns}")

    @staticmethod
    def has_sufficient_data(df: pd.DataFrame, min_rows: int = 10) -> bool:
        """Check if DataFrame has sufficient data.

        Args:
            df: DataFrame to check
            min_rows: Minimum number of rows required

        Returns:
            True if sufficient data, False otherwise
        """
        return len(df) >= min_rows and not df.empty


class ParameterValidator:
    """Parameter and configuration validation utilities."""

    @staticmethod
    def validate_positive_number(value: Any, name: str) -> float:
        """Validate value is a positive number.

        Args:
            value: Value to validate
            name: Parameter name for error messages

        Returns:
            Validated float value

        Raises:
            ValueError: If value is not positive
        """
        try:
            num_value = float(value)
        except (ValueError, TypeError) as e:
            raise ValueError(f"{name} must be a number, got: {value}") from e

        if num_value <= 0:
            raise ValueError(f"{name} must be positive, got: {num_value}")

        return num_value

    @staticmethod
    def validate_positive_integer(value: Any, name: str) -> int:
        """Validate value is a positive integer.

        Args:
            value: Value to validate
            name: Parameter name for error messages

        Returns:
            Validated integer value

        Raises:
            ValueError: If value is not a positive integer
        """
        try:
            int_value = int(value)
        except (ValueError, TypeError) as e:
            raise ValueError(f"{name} must be an integer, got: {value}") from e

        if int_value <= 0:
            raise ValueError(f"{name} must be positive, got: {int_value}")

        return int_value

    @staticmethod
    def validate_range(value: Any, name: str, min_val: float, max_val: float) -> float:
        """Validate value is within specified range.

        Args:
            value: Value to validate
            name: Parameter name for error messages
            min_val: Minimum allowed value
            max_val: Maximum allowed value

        Returns:
            Validated float value

        Raises:
            ValueError: If value is outside range
        """
        try:
            num_value = float(value)
        except (ValueError, TypeError) as e:
            raise ValueError(f"{name} must be a number, got: {value}") from e

        if not (min_val <= num_value <= max_val):
            raise ValueError(f"{name} must be between {min_val} and {max_val}, got: {num_value}")

        return num_value
