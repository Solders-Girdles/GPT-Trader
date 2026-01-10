# Coinbase Advanced Trade API - Authentication Guide

---
status: current
created: 2025-10-19
last-verified: 2025-10-19
verification-schedule: quarterly
scope: Advanced Trade API v3 authentication methods
documentation-venue: docs.cdp.coinbase.com/advanced-trade/docs/rest-api-auth
---

> Status: GPT-Trader uses JWT-only authentication (CDPJWTAuth for INTX perps, SimpleAuth for spot). OAuth2 is not implemented.

## Overview

Coinbase Advanced Trade API authentication in GPT-Trader uses JWT-only methods:

| Method | Use Case | Rate Limit | Supported |
|--------|----------|------------|-----------|
| CDP (JWT) | Production perpetuals (INTX) | 30 req/sec private | Yes |
| SimpleAuth (JWT) | Spot trading | 30 req/sec private | Yes |

Note: If you need legacy Exchange authentication or OAuth2 for external integrations, use Coinbase's official docs. GPT-Trader does not implement those flows.

---

## 1. CDP API Keys with JWT Authentication

**Best for**: Production perpetual futures (INTX accounts only)

### Key Generation

1. Go to https://www.coinbase.com/developer-platform/
2. Create a new "Cloud API Keys" credential
3. Select permissions: "All" or specific (e.g., "Manage Orders", "View Balances")
4. Copy the API Key and Private Key

**Key Format:**
```
API Key:     organizations/{org_id}/apiKeys/{key_id}
Private Key: -----BEGIN EC PRIVATE KEY-----
             MHcCAQEEIIGlVtHkCJL...
             -----END EC PRIVATE KEY-----
```

### JWT Creation

JWT must be generated for each request and signed with your private key.

```python
from gpt_trader.features.brokerages.coinbase.auth import CDPJWTAuth

auth = CDPJWTAuth(
    api_key="organizations/{org_id}/apiKeys/{key_id}",
    private_key=private_key_pem,
)

token = auth.generate_jwt("GET", "/api/v3/brokerage/accounts")
```

### Making Requests with JWT

```python
import requests

method = "GET"
path = "/api/v3/brokerage/accounts"
headers = auth.get_headers(method, path)

response = requests.get(
    f"https://api.coinbase.com{path}",
    headers=headers,
    timeout=10,
)

response.raise_for_status()
print(response.json())
```

---

## 2. SimpleAuth (JWT) - Spot Trading

**Best for**: Default spot trading

```python
from gpt_trader.features.brokerages.coinbase.auth import SimpleAuth

auth = SimpleAuth(
    key_name="organizations/{org_id}/apiKeys/{key_id}",
    private_key=private_key_pem,
)

headers = auth.get_headers("GET", "/api/v3/brokerage/accounts")
```

---

## 3. Rate Limit Headers

All authenticated requests return rate limit information in headers:

```python
def check_rate_limits(response):
    """Extract and log rate limit info"""

    limit = response.headers.get("CB-RATELIMIT-LIMIT")
    remaining = response.headers.get("CB-RATELIMIT-REMAINING")
    reset_time = response.headers.get("CB-RATELIMIT-RESET")

    print(f"Rate limit: {remaining}/{limit}")
    print(f"Resets at: {reset_time}")

    if int(remaining) < int(limit) * 0.2:
        logger.warning(f"Approaching rate limit: {remaining} requests left")

    return {
        "limit": limit,
        "remaining": remaining,
        "reset": reset_time,
    }
```

---

## 4. Error Responses & Troubleshooting

### Common Authentication Errors

| Status | Error | Cause | Solution |
|--------|-------|-------|----------|
| 401 | AUTHENTICATION_ERROR | Invalid/missing credentials | Check API key and private key |
| 401 | Invalid Signature | Incorrect JWT signing | Verify private key format and algorithm |
| 403 | FORBIDDEN | Insufficient permissions | Check API key scopes/permissions |
| 403 | AUTHENTICATION_ERROR | IP not whitelisted | Add IP to API key IP whitelist |

### JWT Troubleshooting

```python
def debug_jwt(jwt_token):
    """Decode and inspect JWT (for debugging only)"""
    import json
    import base64

    parts = jwt_token.split('.')
    payload = parts[1]
    padding = 4 - len(payload) % 4
    payload += '=' * padding

    decoded = base64.urlsafe_b64decode(payload)
    print(json.dumps(json.loads(decoded), indent=2))
```

---

## 5. Best Practices

DO:
- Rotate API keys regularly
- Store secrets in environment variables (never in code)
- Use HTTPS/WSS only (never HTTP/WS)
- Validate SSL certificates in production
- Implement exponential backoff on rate limit errors
- Monitor rate limit headers to stay under limits

DON'T:
- Commit API keys to version control
- Share API secrets via email or Slack
- Log full API keys or secrets (truncate: "****...key123")
- Use the same key for multiple environments
- Ignore rate limit warnings (429 responses)
- Disable SSL verification

---

## 6. Not Supported in GPT-Trader

- Legacy Exchange API key authentication is not implemented.
- OAuth2 (user-delegated access) is not implemented.

Refer to the official Coinbase docs if you need those flows outside GPT-Trader.

---

## Maintenance & Versioning

- Last Updated: 2025-10-19
- Verification Schedule: Quarterly
- Changes: Check https://docs.cdp.coinbase.com/coinbase-app/introduction/changelog

---

## See Also

- [coinbase_websocket_reference.md](coinbase_websocket_reference.md) - WebSocket integration
- [coinbase_complete.md](coinbase_complete.md) - Complete integration guide
