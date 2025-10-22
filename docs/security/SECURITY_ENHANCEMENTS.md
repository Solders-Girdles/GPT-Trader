# Security and Configuration Hygiene Enhancements

This document describes the security enhancements implemented for the GPT-Trader system, focusing on secrets management, IP allowlisting, and configuration governance.

## Overview

The security enhancements provide:

1. **CDP Secrets Provider** - Secure management of Coinbase Developer Platform (CDP) API keys with short-lived JWT generation
2. **IP Allowlisting** - Enforcement of IP restrictions for API key usage, critical for INTX and production trading
3. **Two-Person Rule** - Approval workflow for critical configuration changes (risk limits, leverage)
4. **Event Store Logging** - Comprehensive audit trail of all configuration changes

## 1. CDP Secrets Provider

### Features

- **Secure Storage**: Integration with HashiCorp Vault or 1Password via SecretsManager
- **Short-Lived JWTs**: Generates tokens with 120-second expiration per Coinbase specifications
- **IP Allowlisting**: Validates client IPs before generating tokens
- **Automatic Rotation**: Tracks credential age and enforces rotation policies
- **Thread-Safe**: Concurrent access protection with caching

### Usage

#### Storing CDP Credentials

```python
from bot_v2.security.cdp_secrets_provider import store_cdp_credentials

# Store credentials with IP allowlist
store_cdp_credentials(
    service_name="coinbase_production",
    api_key_name="organizations/my-org/apiKeys/my-key",
    private_key_pem="-----BEGIN EC PRIVATE KEY-----\n...\n-----END EC PRIVATE KEY-----",
    rotation_policy_days=90,
    allowed_ips=["192.168.1.1", "10.0.0.0/24"],
)
```

#### Generating Short-Lived JWT

```python
from bot_v2.security.cdp_secrets_provider import generate_cdp_jwt

# Generate JWT for API request
token = generate_cdp_jwt(
    service_name="coinbase_production",
    method="GET",
    path="/api/v3/brokerage/accounts",
    client_ip="192.168.1.1",  # Optional - validated against allowlist
    base_url="https://api.coinbase.com",
)

if token:
    print(f"Token expires in: {token.seconds_until_expiry} seconds")
    # Use token.token in Authorization header
else:
    print("Token generation failed (IP rejected or credentials missing)")
```

#### Rotating Credentials

```python
from bot_v2.security.cdp_secrets_provider import get_cdp_secrets_provider

provider = get_cdp_secrets_provider()

# Rotate to new key
provider.rotate_credentials(
    service_name="coinbase_production",
    new_api_key_name="organizations/my-org/apiKeys/new-key",
    new_private_key_pem="-----BEGIN EC PRIVATE KEY-----\n...\n-----END EC PRIVATE KEY-----",
)
```

### Configuration

Use HashiCorp Vault for production:

```bash
# Environment variables
export VAULT_ADDR=https://vault.example.com:8200
export VAULT_TOKEN=your-vault-token
export BOT_V2_ENCRYPTION_KEY=your-encryption-key
```

### Security Specifications

- **JWT Algorithm**: ES256 (ECDSA with SHA-256)
- **Token Lifetime**: 120 seconds (2 minutes) per Coinbase spec
- **Key Rotation**: Configurable (default 90 days)
- **Storage**: Encrypted at rest (Fernet encryption) with Vault support

## 2. IP Allowlisting Enforcement

### Features

- **Per-Service Rules**: Configure IP allowlists per service (e.g., coinbase_production, coinbase_intx)
- **CIDR Support**: Allow entire IP ranges using CIDR notation (e.g., 10.0.0.0/24)
- **Audit Logging**: All validation attempts logged for security review
- **Runtime Configuration**: Enable/disable enforcement and rules dynamically

### Usage

#### Configuring IP Allowlists via Environment

```bash
# Enable IP allowlist enforcement (default: enabled)
export IP_ALLOWLIST_ENABLED=1

# Configure allowlist for INTX
export IP_ALLOWLIST_COINBASE_INTX=192.168.1.1,192.168.1.2,10.0.0.0/24

# Configure allowlist for production
export IP_ALLOWLIST_COINBASE_PRODUCTION=203.0.113.0/24
```

#### Programmatic Configuration

```python
from bot_v2.security.ip_allowlist_enforcer import add_ip_allowlist_rule, validate_ip

# Add rule
add_ip_allowlist_rule(
    service_name="coinbase_intx",
    allowed_ips=["192.168.1.1", "192.168.1.2", "10.0.0.0/24"],
    description="INTX production servers",
)

# Validate IP
result = validate_ip(client_ip="192.168.1.1", service_name="coinbase_intx")

if result.is_allowed:
    print(f"IP allowed (matched: {result.matched_rule})")
else:
    print(f"IP REJECTED: {result.reason}")
```

#### Managing Rules

```python
from bot_v2.security.ip_allowlist_enforcer import get_ip_allowlist_enforcer

enforcer = get_ip_allowlist_enforcer()

# List all rules
rules = enforcer.list_rules()
for rule in rules:
    print(f"{rule.service_name}: {rule.allowed_ips}")

# Disable rule temporarily
enforcer.disable_rule("coinbase_intx")

# Re-enable rule
enforcer.enable_rule("coinbase_intx")

# Remove rule
enforcer.remove_rule("coinbase_intx")
```

#### Audit Log

```python
# Get recent validation attempts
log = enforcer.get_validation_log(limit=100)

for entry in log:
    print(f"{entry['timestamp']}: {entry['client_ip']} -> {entry['service_name']}")
    print(f"  Allowed: {entry['is_allowed']}, Reason: {entry['reason']}")
```

### Critical for INTX

Coinbase International Exchange (INTX) **requires** IP allowlisting for API keys:

1. Create API keys at [international.coinbase.com](https://international.coinbase.com)
2. Configure IP allowlist in Coinbase dashboard
3. Mirror configuration in bot using `IP_ALLOWLIST_COINBASE_INTX`
4. Validate requests before generating JWTs

## 3. Two-Person Rule for Configuration Changes

### Features

- **Separation of Duties**: Requester and approver must be different users
- **Critical Fields**: Automatic detection of changes requiring approval
- **Time-Limited**: Approval requests expire after 24 hours (configurable)
- **Audit Trail**: All requests, approvals, and rejections logged to event store
- **Batch Changes**: Support multiple changes in single approval request

### Usage

#### Creating Approval Request

```python
from bot_v2.monitoring.two_person_rule import (
    create_approval_request,
    ConfigChange,
    ChangeType,
)
from bot_v2.security.auth_handler import User, Role

# Define requester
requester = User(
    id="trader-001",
    username="trader1",
    email="trader1@example.com",
    role=Role.TRADER,
)

# Define changes
changes = [
    ConfigChange(
        change_type=ChangeType.LEVERAGE,
        field_name="max_leverage",
        old_value=3,
        new_value=5,
        description="Increase max leverage to 5x for BTC-PERP",
    ),
    ConfigChange(
        change_type=ChangeType.RISK_LIMIT,
        field_name="daily_loss_limit",
        old_value=100,
        new_value=200,
        description="Increase daily loss limit",
    ),
]

# Create approval request
request = create_approval_request(
    requester=requester,
    changes=changes,
    metadata={
        "reason": "Production configuration update for Q1 2025",
        "ticket": "RISK-1234",
    },
)

print(f"Approval request created: {request.request_id}")
print(f"Expires at: {request.expires_at}")
```

#### Approving Request

```python
from bot_v2.monitoring.two_person_rule import approve_request

# Define approver (must be different from requester)
approver = User(
    id="admin-001",
    username="admin1",
    email="admin1@example.com",
    role=Role.ADMIN,
)

# Approve request
success, error = approve_request(
    request_id=request.request_id,
    approver=approver,
)

if success:
    print("Request approved!")
else:
    print(f"Approval failed: {error}")
```

#### Rejecting Request

```python
from bot_v2.monitoring.two_person_rule import get_two_person_rule

rule = get_two_person_rule()

success, error = rule.reject_request(
    request_id=request.request_id,
    reviewer=approver,
    reason="Risk parameters too aggressive for current market conditions",
)

if success:
    print("Request rejected")
```

#### Applying Approved Changes

```python
# After approval, mark as applied when changes are committed
success, error = rule.mark_applied(request.request_id)

if success:
    print("Changes applied and logged")
```

#### Checking Pending Approvals

```python
# Get all pending approval requests
pending = rule.get_pending_requests()

for req in pending:
    print(f"Request {req.request_id}:")
    print(f"  Requester: {req.requester_name}")
    print(f"  Changes: {len(req.changes)}")
    print(f"  Expires: {req.expires_at}")
```

### Critical Fields Requiring Approval

The following configuration fields automatically require two-person approval:

- `max_leverage` - Maximum leverage allowed
- `max_position_size` - Maximum position size
- `daily_loss_limit` - Daily loss limit
- `liquidation_buffer` - Liquidation safety buffer
- `circuit_breaker_threshold` - Circuit breaker trigger
- `kill_switch` - Emergency shutdown toggle
- `reduce_only_mode` - Reduce-only mode toggle
- `symbols` - Trading symbol universe
- `profile` - Configuration profile changes
- `per_symbol_leverage` - Per-symbol leverage limits

### Integration with ConfigurationGuardian

```python
from bot_v2.monitoring.configuration_guardian import ConfigurationGuardian
from bot_v2.monitoring.two_person_rule import get_two_person_rule

# ConfigurationGuardian now logs all deltas to event store
guardian = ConfigurationGuardian(baseline_snapshot)

# When resetting baseline (after approval)
guardian.reset_baseline(new_snapshot, user_id="admin-001")
# This automatically logs the configuration delta

# Check which fields need approval
rule = get_two_person_rule()
config_changes = {
    "max_leverage": 5,
    "daily_loss_limit": 200,
    "some_other_field": "value",
}

required_approval = rule.requires_approval(config_changes)
print(f"Fields requiring approval: {required_approval}")
# Output: ['max_leverage', 'daily_loss_limit']
```

## 4. Event Store Logging

All configuration changes are automatically logged to the event store for audit purposes.

### Configuration Delta Events

```python
from bot_v2.monitoring.two_person_rule import log_config_delta

# Log configuration changes
changes = {
    "max_leverage": (3, 5),  # (old_value, new_value)
    "daily_loss_limit": (100, 200),
}

log_config_delta(
    change_type="risk_limit_update",
    changes=changes,
    user_id="admin-001",
    metadata={
        "approval_request_id": "APR-20250122-abcd1234",
        "ticket": "RISK-1234",
    },
)
```

### Querying Event Store

```python
from bot_v2.persistence.event_store import EventStore

event_store = EventStore()

# Get recent config deltas
events = event_store.tail(
    bot_id="config_guardian",
    limit=50,
    types=["metric"],
)

for event in events:
    if event.get("event_type") == "config_delta":
        print(f"Change type: {event['change_type']}")
        print(f"Changes: {event['changes']}")
        print(f"User: {event.get('user_id')}")
        print(f"Time: {event['time']}")
```

## Production Setup Checklist

### 1. Secrets Management

- [ ] Configure HashiCorp Vault endpoint and token
- [ ] Set `BOT_V2_ENCRYPTION_KEY` for local encryption
- [ ] Store CDP credentials with IP allowlists
- [ ] Set rotation policy (recommended: 90 days)
- [ ] Test JWT generation for all services

### 2. IP Allowlisting

- [ ] Set `IP_ALLOWLIST_ENABLED=1`
- [ ] Configure `IP_ALLOWLIST_COINBASE_INTX` for INTX trading
- [ ] Configure `IP_ALLOWLIST_COINBASE_PRODUCTION` for production API
- [ ] Verify IP allowlists in Coinbase dashboard match bot configuration
- [ ] Test validation with production IPs

### 3. Two-Person Rule

- [ ] Configure approval timeout (default: 24 hours)
- [ ] Define admin users with approval permissions
- [ ] Test approval workflow with sample changes
- [ ] Set up monitoring for pending approvals
- [ ] Review event store for audit trail

### 4. Monitoring

- [ ] Set up alerts for rejected IP validation attempts
- [ ] Monitor pending approval requests
- [ ] Review event store logs regularly
- [ ] Set up alerts for expired approval requests
- [ ] Monitor credential rotation status

## Security Best Practices

1. **Secrets Rotation**: Rotate CDP credentials every 90 days or when compromised
2. **IP Allowlisting**: Always use IP allowlists for production and INTX
3. **Two-Person Rule**: Never bypass approval for critical changes
4. **Audit Logs**: Regularly review event store for suspicious activity
5. **Least Privilege**: Grant minimum necessary permissions to users
6. **Vault Access**: Restrict Vault token access to production systems only

## Troubleshooting

### JWT Generation Fails

```
Token generation failed (IP rejected or credentials missing)
```

**Solutions:**
1. Check IP allowlist configuration
2. Verify client IP is in allowed list
3. Confirm CDP credentials are stored correctly
4. Check Vault connectivity

### IP Validation Rejected

```
IP 10.0.0.5 REJECTED for coinbase_intx - not in allowlist
```

**Solutions:**
1. Add IP to allowlist: `enforcer.add_rule("coinbase_intx", ["10.0.0.5"])`
2. Update CIDR range to include IP
3. Verify IP matches production server
4. Check Coinbase dashboard allowlist matches bot config

### Approval Request Expired

```
Request has expired
```

**Solutions:**
1. Create new approval request
2. Increase approval timeout if needed
3. Set up alerts for pending approvals
4. Review approval workflow efficiency

## API Reference

### CDP Secrets Provider

- `store_cdp_credentials(service_name, api_key_name, private_key_pem, *, rotation_policy_days, allowed_ips)`
- `get_cdp_credentials(service_name) -> CDPCredentials | None`
- `generate_cdp_jwt(service_name, method, path, *, client_ip, base_url) -> CDPJWTToken | None`
- `rotate_credentials(service_name, new_api_key_name, new_private_key_pem) -> bool`

### IP Allowlist Enforcer

- `add_ip_allowlist_rule(service_name, allowed_ips, *, description) -> bool`
- `validate_ip(client_ip, service_name) -> IPValidationResult`
- `get_ip_allowlist_enforcer() -> IPAllowlistEnforcer`

### Two-Person Rule

- `create_approval_request(requester, changes, *, metadata) -> ApprovalRequest`
- `approve_request(request_id, approver) -> tuple[bool, str | None]`
- `reject_request(request_id, reviewer, reason) -> tuple[bool, str | None]`
- `mark_applied(request_id) -> tuple[bool, str | None]`
- `log_config_delta(change_type, changes, *, user_id, metadata)`

## References

- [Coinbase Advanced Trade API Authentication](https://docs.cdp.coinbase.com/advanced-trade/docs/rest-api-auth)
- [HashiCorp Vault Documentation](https://www.vaultproject.io/docs)
- [NIST Special Publication 800-53: Security Controls](https://csrc.nist.gov/publications/detail/sp/800-53/rev-5/final)
