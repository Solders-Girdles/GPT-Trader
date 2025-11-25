from gpt_trader.utilities.logging_patterns import get_logger as _get_pattern_logger
from typing import Any

def get_logger(name: str = "system", settings: Any = None, **kwargs) -> Any:
    return _get_pattern_logger(name, **kwargs)

from enum import Enum, auto

class LogLevel(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    DEBUG = "DEBUG"

