from enum import Enum, auto


class CircuitBreakerAction(Enum):
    NONE = auto()
    WARN = auto()
    REDUCE_ONLY = auto()
    KILL_SWITCH = auto()
