# Coinbase Integration Remediation Plan

## Executive Summary
This document contains a comprehensive remediation plan for the GPT-Trader Coinbase integration, addressing critical issues identified through external review and internal analysis.

## Critical Issues Identified

### 1. Sandbox/API Mode Mismatch (CRITICAL)
- **Problem**: Sandbox URL uses legacy Exchange API but code uses Advanced Trade v3 endpoints
- **Impact**: Complete failure in sandbox mode - 404 errors on all requests
- **Location**: `broker_factory.py` line 44, `client.py` hardcoded endpoints

### 2. WebSocket Transport Not Initialized (CRITICAL)
- **Problem**: No default transport set, assertion failure at runtime
- **Impact**: All real-time trading features blocked
- **Location**: `ws.py` line 63

### 3. Security: .env Files Exposed (CRITICAL)
- **Problem**: Actual `.env` and `.env.production` files exist in repository
- **Impact**: Potential credential exposure
- **Evidence**: Files visible in repo despite .gitignore

### 4. Documentation Severely Outdated
- **Problem**: README says "to be implemented" but full implementation exists
- **Impact**: Developer confusion, lack of trust

### 5. Type System Duplication
- **Problem**: Duplicate broker types in `brokerages/core/interfaces.py` and `live_trade/types.py`
- **Impact**: Inconsistent models, harder maintenance

### 6. Paper Trading Coupling
- **Problem**: Paper engine auto-initializes real broker connections
- **Impact**: Unexpected network dependencies in tests

## Remediation Plan

## 🔴 Immediate Fixes (Block Production) - Week 0-1

### 1. Fix Sandbox/API Mode Mismatch

**Owner**: Brokerage slice dev

**Implementation**:
```python
# coinbase/models.APIConfig: Add explicit mode
api_mode: Literal["advanced", "exchange"] = "advanced"

# broker_factory.create_brokerage: Derive mode
if COINBASE_SANDBOX=1 and no override:
    api_mode = "exchange"
    base_url = "https://api-public.sandbox.exchange.coinbase.com"
    log.warning("Sandbox mode: Only legacy Exchange endpoints available")

# coinbase/client.CoinbaseClient: Route by mode
def _get_endpoint_path(self, endpoint_name: str) -> str:
    ENDPOINT_MAP = {
        'advanced': {
            'products': '/api/v3/brokerage/market/products',
            'accounts': '/api/v3/brokerage/accounts',
        },
        'exchange': {
            'products': '/products',
            'accounts': '/accounts',
        }
    }
    
    if endpoint_name not in ENDPOINT_MAP[self.api_mode]:
        raise InvalidRequestError(
            f"Endpoint '{endpoint_name}' not available in {self.api_mode} mode"
        )
    
    return ENDPOINT_MAP[self.api_mode][endpoint_name]
```

**Tests**:
- Unit: Verify factory selects correct api_mode and base_url for sandbox
- Unit: Ensure unsupported endpoints raise clear errors in exchange mode
- Integration: Verify get_products() hits correct path per mode

**Acceptance Criteria**:
- CLI smoke with --sandbox lists products without 404
- Advanced production mode remains untouched

### 2. Fix WebSocket Transport

**Owner**: Brokerage slice dev

**Implementation**:
```python
# coinbase/ws.CoinbaseWebSocket.connect():
def connect(self):
    if self._transport is None:
        try:
            from .transports import RealTransport
            self._transport = RealTransport()
        except ImportError:
            raise ImportError(
                "websocket-client not installed. "
                "Install with: pip install websocket-client"
            )
    # Continue with connection...
```

**Tests**:
- Unit: With no transport and websocket-client present → connect() succeeds
- Unit: Simulate missing dependency → actionable error message

**Acceptance Criteria**:
- stream_trades() does not assert-fail when transport not injected
- Demo scripts can still inject mock transports

### 3. Remove .env Files and Rotate Secrets

**Owner**: Maintainer

**Steps**:
1. Rotate any real keys in Coinbase, etc.
2. Execute:
   ```bash
   git rm --cached .env .env.production
   git commit -m "security: Remove exposed env files"
   ```
3. If history cleanup needed: Plan git filter-repo/BFG with owner approval
4. Add pre-commit hook to block .env* additions
5. Add SECURITY.md note on secrets handling

**Acceptance Criteria**:
- No .env* files tracked in git
- Secret scans pass
- .env.template remains for reference

## 🟡 High Priority (Within 1 Week)

### 4. Consolidate Type System

**Owner**: Live trade + Brokerage devs

**Implementation**:
1. Refactor `features/live_trade` to import from `brokerages/core/interfaces`
2. Create temporary shim with deprecation warning:
   ```python
   # live_trade/types.py (temporary)
   """DEPRECATED: Import from features.brokerages.core.interfaces instead"""
   from ..brokerages.core.interfaces import *
   import warnings
   warnings.warn(
       "live_trade.types is deprecated. Use brokerages.core.interfaces",
       DeprecationWarning,
       stacklevel=2
   )
   ```
3. Update all imports
4. Remove shim after migration

**Tests**: 
- Run full test suite
- Verify no duplicate class definitions
- All imports resolve correctly

**Acceptance Criteria**:
- Single broker type system remains
- No mismatched Order/Position types

### 5. Update Documentation

**Owner**: Docs/maintainer

**Tasks**:
1. Rewrite `src/bot_v2/features/brokerages/coinbase/README.md`:
   - Current features implemented
   - Auth modes (HMAC vs CDP)
   - Example environment setup
   - Sandbox limitations
2. Update `docs/coinbase.md`:
   - API mode matrix
   - Endpoint compatibility table
   - CLI smoke test instructions
3. Update `.env.template` comments:
   - Clarify COINBASE_API_MODE
   - Add CB-VERSION notes

**Acceptance Criteria**:
- New contributor can set up HMAC or CDP in < 5 minutes
- Sandbox behavior clearly documented

### 6. Decouple Paper Trading

**Owner**: Orchestration dev

**Implementation**:
```python
# PaperExecutionEngine.__init__:
def __init__(
    self,
    quote_provider: Optional[Callable[[str], Optional[float]]] = None,
    broker: Optional[IBrokerage] = None
):
    # Remove auto _init_broker() call
    self.quote_provider = quote_provider
    self.broker = broker
    
def get_quote(self, symbol: str) -> Optional[float]:
    if self.quote_provider:
        return self.quote_provider(symbol)
    elif self.broker and self.broker.connected:
        return self.broker.get_quote(symbol)
    else:
        return None  # Or mock price
```

**Tests**:
- Update integration tests to inject quote provider
- Verify offline tests make no network calls

**Acceptance Criteria**:
- No network calls unless broker explicitly passed
- Paper trading works fully offline

## 🟢 Medium Priority (Within 1 Month)

### 7. Archive Cleanup

**Owner**: Maintainer

**Options**:
1. Move `archived/` to separate archival repo
2. Place under Git LFS
3. Add clear deprecation notices

**Steps**:
- Add top-level README: `src/bot_v2` is primary, `src/bot` is deprecated
- Consider `.github/CODEOWNERS` to restrict archived/ changes

**Acceptance Criteria**:
- New contributors land in bot_v2 without distraction
- Clear navigation path

### 8. API Mode Consistency (Finalize)

**Owner**: Brokerage slice dev

**Tasks**:
1. Implement exchange variants for all methods or raise clear errors
2. Centralize path construction via mode-aware helper
3. Add logging for suspicious combinations (passphrase with advanced mode)

**Tests**:
- Mode-based routing for all endpoints
- Negative tests for unsupported calls

**Acceptance Criteria**:
- No accidental cross-mode calls
- Self-explanatory error messages

## 📊 Validation Scripts

### Environment Validation
```python
# scripts/validate_environment.py
def validate_coinbase_config():
    """Pre-flight check for Coinbase configuration"""
    mode = os.getenv('COINBASE_API_MODE', 'advanced')
    sandbox = os.getenv('COINBASE_SANDBOX', '0') == '1'
    
    if sandbox and mode == 'advanced':
        print("⚠️  WARNING: Advanced Trade may not work in sandbox")
        print("   Set COINBASE_API_MODE=exchange for sandbox")
    
    if os.getenv('COINBASE_CDP_API_KEY') and os.getenv('COINBASE_API_KEY'):
        print("⚠️  WARNING: Both CDP and HMAC credentials found")
        print("   CDP will take precedence")
```

### Critical Fixes Validation
```python
# scripts/validate_critical_fixes.py
def validate_sandbox_mode():
    """Ensure sandbox works with correct API mode"""
    os.environ['COINBASE_SANDBOX'] = '1'
    broker = create_brokerage()
    assert 'exchange' in broker._client.base_url
    products = broker.get_products()  # Should not 404
    assert len(products) > 0

def validate_ws_transport():
    """Ensure WebSocket initializes without assertion"""
    ws = CoinbaseWebSocket("wss://test")
    ws.connect()  # Should not assert
    assert ws._transport is not None
```

## 📈 Success Metrics

1. **Sandbox Success Rate**: % of sandbox API calls that succeed (target: 100%)
2. **WS Connection Rate**: % of WS connections without assertion (target: 100%)
3. **Type Import Locations**: Imports from new location (target: 100%)
4. **Documentation Freshness**: Days since last update (target: < 30)
5. **Test Coverage**: Coverage of Coinbase integration (target: > 80%)

## 🚀 Strategic Recommendation (Optional)

### Extract bot_v2 to Clean Repository

**Benefits**:
- Escape 159K lines of technical debt
- Clear boundaries and ownership
- Faster CI/CD
- Better developer experience

**Steps**:
1. Create new repo with `src/bot_v2`, `tests`, relevant `docs`
2. Use `git subtree split` to preserve history
3. Set up fresh CI/CD pipeline
4. Add secrets scanning and pre-commit hooks
5. Mark old repo as deprecated with forwarding link

**Acceptance Criteria**:
- New repo is lean with clear boundaries
- CI pipeline < 5 minutes
- Contributors onboard in minutes

## Timeline Summary

| Week | Priority | Tasks | Owner |
|------|----------|-------|-------|
| 0-1 | Immediate | API mode fix, WS transport, .env removal | Brokerage dev, Maintainer |
| 1 | High | Type consolidation, Docs update, Paper decoupling | Multiple |
| 2-4 | Medium | Archive cleanup, API consistency | Maintainer, Brokerage dev |
| Optional | Strategic | Repository extraction | Maintainer |

## Testing Checklist (CI Gates)

- [ ] Unit: Coinbase client path routing
- [ ] Unit: Auth headers validation
- [ ] Unit: Rate limiting and retries
- [ ] Unit: WS connect fallback
- [ ] Unit: Adapter streams with default WS
- [ ] Unit: Order validation
- [ ] Unit: Error mapping
- [ ] Integration: Paper engine offline mode
- [ ] Integration: Online tests skip without credentials
- [ ] Lint: No .env* tracked
- [ ] Security: Secrets scan passes
- [ ] Coverage: > 80% for critical paths

## Notes

- External review provided by independent consultant
- Plan validated against current codebase state
- All code snippets are illustrative; adapt to actual implementation
- Prioritization based on production impact and risk

---

*Last Updated: 2025-08-29*
*Status: Draft - Ready for Review*
*Next Review: After Week 1 completion*