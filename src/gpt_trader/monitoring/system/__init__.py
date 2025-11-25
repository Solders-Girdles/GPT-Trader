from enum import Enum
from typing import Any

from gpt_trader.utilities.logging_patterns import get_logger as _get_pattern_logger


def get_logger(name: str = "system", settings: Any = None, **kwargs: Any) -> Any:
    return _get_pattern_logger(name, **kwargs)


class LogLevel(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    DEBUG = "DEBUG"
