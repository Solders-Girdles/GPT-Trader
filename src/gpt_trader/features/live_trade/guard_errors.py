class GuardError(Exception):
    pass

class RiskLimitExceeded(GuardError):
    pass

class RiskGuardError(GuardError):
    pass

class RiskGuardActionError(GuardError):
    pass

class RiskGuardComputationError(GuardError):
    pass

class RiskGuardDataCorrupt(GuardError):
    pass

class RiskGuardDataUnavailable(GuardError):
    pass

class RiskGuardTelemetryError(GuardError):
    pass

def record_guard_failure(error):
    pass
