# Coinbase Advanced Trade API - Authentication Guide

---
status: current
created: 2025-10-19
last-verified: 2025-10-19
verification-schedule: quarterly
scope: Advanced Trade API v3 authentication methods
documentation-venue: docs.cdp.coinbase.com/advanced-trade/docs/rest-api-auth
---

> **Status**: This guide covers three authentication methods: CDP/JWT (production perps), HMAC (spot + sandbox), and OAuth2 (delegated access). Verify current status and any breaking changes at https://docs.cdp.coinbase.com/coinbase-app/introduction/changelog

## Overview

Coinbase Advanced Trade API supports three authentication methods depending on your use case:

| Method | Use Case | Rate Limit | Recommended |
|--------|----------|-----------|-------------|
| **CDP (JWT)** | Production perpetuals (INTX) | 30 req/sec private | ✅ Perps |
| **HMAC** | Spot trading + sandbox | 30 req/sec private | ✅ Spot |
| **OAuth2** | User-delegated access | 30 req/sec private | Multiuser apps |

---

## 1. CDP API Keys with JWT Authentication

**Best for**: Production perpetual futures (INTX accounts only)

### Key Generation

1. Go to https://www.coinbase.com/developer-platform/
2. Create a new "Cloud API Keys" credential
3. Select permissions: "All" or specific (e.g., "Manage Orders", "View Balances")
4. Copy the **API Key** and **Private Key**

**Key Format:**
```
API Key:     organizations/{org_id}/apiKeys/{key_id}
Private Key: -----BEGIN EC PRIVATE KEY-----
             MHcCAQEEIIGlVtHkCJL...
             -----END EC PRIVATE KEY-----
```

### JWT Creation

JWT must be generated for **each request** and signed with your private key.

```python
import jwt
import time
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend

class CDPAuth:
    def __init__(self, api_key: str, private_key_pem: str):
        self.api_key = api_key
        self.private_key_pem = private_key_pem

    def generate_jwt(self, service: str = "cdp_service") -> str:
        """Generate JWT for each request"""
        now = int(time.time())

        private_key = serialization.load_pem_private_key(
            self.private_key_pem.encode(),
            password=None,
            backend=default_backend()
        )

        claims = {
            "sub": self.api_key,
            "iss": "coinbase-cloud",
            "nbf": now,
            "exp": now + 120,  # JWT valid for 2 minutes
            "iat": now,
            "service": service
        }

        token = jwt.encode(
            claims,
            private_key,
            algorithm="ES256",
            headers={"kid": self.api_key}
        )
        return token
```

### Making Requests with JWT

```python
import requests
import json

auth = CDPAuth(api_key, private_key_pem)

def make_request(method: str, path: str, body: dict = None) -> dict:
    """Make authenticated request with JWT"""
    jwt_token = auth.generate_jwt()

    url = f"https://api.coinbase.com{path}"

    headers = {
        "Authorization": f"Bearer {jwt_token}",
        "Content-Type": "application/json"
    }

    if method == "GET":
        response = requests.get(url, headers=headers)
    elif method == "POST":
        response = requests.post(url, headers=headers, json=body)
    elif method == "DELETE":
        response = requests.delete(url, headers=headers)

    return response.json()

# Example: Create order
order = make_request(
    "POST",
    "/api/v3/brokerage/orders",
    {
        "product_id": "BTC-PERP",
        "side": "BUY",
        "order_configuration": {
            "limit_limit_gtc": {
                "base_size": "0.01",
                "limit_price": "45000"
            }
        }
    }
)
```

### Error Handling

```python
def make_request_with_retry(method, path, body=None, max_retries=3):
    """Retry on rate limit and server errors"""
    import time

    for attempt in range(max_retries):
        try:
            response = requests.request(
                method,
                f"https://api.coinbase.com{path}",
                headers={
                    "Authorization": f"Bearer {auth.generate_jwt()}",
                    "Content-Type": "application/json"
                },
                json=body,
                timeout=10
            )

            # Check rate limit headers
            remaining = response.headers.get("CB-RATELIMIT-REMAINING")
            reset_time = response.headers.get("CB-RATELIMIT-RESET")

            if response.status_code == 429:
                retry_after = int(response.headers.get("retry-after", 5))
                logger.warning(f"Rate limited, retry after {retry_after}s")
                time.sleep(retry_after)
                continue

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)  # Exponential backoff

    return None
```

---

## 2. HMAC Authentication

**Best for**: Spot trading and sandbox testing

### API Key Generation

1. Go to https://www.coinbase.com/settings/api (for personal accounts)
2. Or Coinbase Pro/Advanced Trade dashboard for institutional accounts
3. Create API key with required permissions
4. Copy: **API Key**, **API Secret**, and **Passphrase**

### Request Signing

HMAC-SHA256 signature required for all private endpoints:

```python
import hmac
import hashlib
import base64
from datetime import datetime

class HMACAuth:
    def __init__(self, api_key: str, api_secret: str, passphrase: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase

    def create_signature(self, timestamp: str, method: str, path: str, body: str = "") -> str:
        """Create HMAC-SHA256 signature

        ⚠️ path must be FULL path: /api/v3/brokerage/accounts (not just /accounts)
        Example message to sign: "1697750400GET/api/v3/brokerage/accounts"
        """
        message = f"{timestamp}{method}{path}{body}"

        # Decode base64 secret
        secret_bytes = base64.b64decode(self.api_secret)

        # Create HMAC
        signature = hmac.new(
            secret_bytes,
            message.encode(),
            hashlib.sha256
        )

        # Return base64-encoded signature
        return base64.b64encode(signature.digest()).decode()

    def get_headers(self, method: str, path: str, body: str = "") -> dict:
        """Generate authentication headers"""
        timestamp = str(datetime.utcnow().timestamp())
        signature = self.create_signature(timestamp, method, path, body)

        return {
            "CB-ACCESS-KEY": self.api_key,
            "CB-ACCESS-SIGN": signature,
            "CB-ACCESS-TIMESTAMP": timestamp,
            "CB-ACCESS-PASSPHRASE": self.passphrase,
            "Content-Type": "application/json"
        }
```

### Making HMAC Requests

```python
import requests
import json

auth = HMACAuth(api_key, api_secret, passphrase)

def make_hmac_request(method: str, path: str, body: dict = None) -> dict:
    """Make HMAC-authenticated request

    ⚠️ CRITICAL: path must be the FULL request path including /api/v3/brokerage
    (e.g., "/api/v3/brokerage/accounts"), NOT just the endpoint (e.g., "/accounts").
    The signature is computed over the full path; truncated path = 401 Unauthorized.
    """
    url = "https://api.coinbase.com" + path  # Full URL

    body_str = json.dumps(body) if body else ""
    # Pass the FULL path to get_headers for signature computation
    headers = auth.get_headers(method, path, body_str)

    if method == "GET":
        response = requests.get(url, headers=headers)
    elif method == "POST":
        response = requests.post(url, headers=headers, data=body_str)
    elif method == "DELETE":
        response = requests.delete(url, headers=headers)

    return response.json()

# Example: List accounts (note: FULL path including /api/v3/brokerage)
accounts = make_hmac_request("GET", "/api/v3/brokerage/accounts")
print(f"Accounts: {json.dumps(accounts, indent=2)}")
```

### Sandbox Testing with HMAC

```python
# Use CDP sandbox base URL (Accounts and Orders endpoints only)
sandbox_url = "https://api-public.sandbox.exchange.coinbase.com/api/v3/brokerage"

def make_sandbox_request(method: str, path: str, body: dict = None):
    """Sandbox uses different base URL and limited endpoints"""
    # Sandbox only supports: /accounts, /accounts/{account_id}, /orders, /orders/batch, /orders/historical/fills
    # Other endpoints return 404. Responses are static and pre-defined, not live.

    url = sandbox_url + path
    body_str = json.dumps(body) if body else ""
    headers = auth.get_headers(method, path, body_str)

    # ... same request logic
```

---

## 3. OAuth2 Authentication

**Best for**: Delegated user access and multi-user applications

### OAuth2 Flow Overview

1. **User Authorization**: Redirect user to Coinbase login
2. **Authorization Code**: Receive code from user's browser
3. **Token Exchange**: Exchange code for access + refresh tokens
4. **API Requests**: Use access token to call API
5. **Token Refresh**: Use refresh token when access expires

### Setup (as Application Developer)

```python
from authlib.integrations.requests_client import OAuth2Session

CLIENT_ID = "your_client_id"
CLIENT_SECRET = "your_client_secret"
REDIRECT_URI = "http://localhost:8000/callback"

class OAuth2Coinbase:
    def __init__(self):
        self.session = OAuth2Session(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            redirect_uri=REDIRECT_URI
        )

    def get_authorization_url(self):
        """Step 1: Get URL to send user to Coinbase"""
        url, state = self.session.create_authorization_url(
            "https://www.coinbase.com/oauth/authorize",
            scope=["wallet:accounts:read", "wallet:transactions:read"]
        )
        return url, state

    def exchange_code_for_token(self, code: str, state: str):
        """Step 2: Exchange authorization code for tokens"""
        token = self.session.fetch_token(
            "https://www.coinbase.com/oauth/token",
            code=code,
            state=state
        )
        return token

    def make_request(self, access_token: str, method: str, url: str):
        """Step 3: Use access token to call API"""
        headers = {
            "Authorization": f"Bearer {access_token}"
        }
        response = requests.request(method, url, headers=headers)
        return response.json()
```

### Token Management

```python
class TokenManager:
    def __init__(self, refresh_token: str):
        self.refresh_token = refresh_token
        self.access_token = None
        self.token_expires_at = None

    def refresh_access_token(self):
        """Refresh expired access token"""
        response = requests.post(
            "https://www.coinbase.com/oauth/token",
            data={
                "grant_type": "refresh_token",
                "refresh_token": self.refresh_token,
                "client_id": CLIENT_ID,
                "client_secret": CLIENT_SECRET
            }
        )

        token_data = response.json()
        self.access_token = token_data["access_token"]

        # Refresh tokens expire after 1.5 years
        # Access tokens typically expire after 2 hours
        self.token_expires_at = time.time() + token_data.get("expires_in", 7200)

        return self.access_token

    def get_valid_token(self):
        """Get access token, refresh if needed"""
        if self.token_expires_at and time.time() > self.token_expires_at:
            self.refresh_access_token()
        return self.access_token
```

**Important OAuth2 Notes**:
- Refresh tokens expire after **1.5 years**
- Access tokens typically expire after **2 hours**
- OAuth connections enforce **portfolio account-level trade access**
- Revocation requires client credentials to be passed in request body

---

## 4. Rate Limit Headers

All authenticated requests return rate limit information in headers:

```python
def check_rate_limits(response):
    """Extract and log rate limit info"""

    limit = response.headers.get("CB-RATELIMIT-LIMIT")
    remaining = response.headers.get("CB-RATELIMIT-REMAINING")
    reset_time = response.headers.get("CB-RATELIMIT-RESET")

    print(f"Rate limit: {remaining}/{limit}")
    print(f"Resets at: {reset_time}")

    # Alert when approaching limit
    if int(remaining) < int(limit) * 0.2:
        logger.warning(f"Approaching rate limit: {remaining} requests left")

    return {
        "limit": limit,
        "remaining": remaining,
        "reset": reset_time
    }
```

---

## 5. Error Responses & Troubleshooting

### Common Authentication Errors

| Status | Error | Cause | Solution |
|--------|-------|-------|----------|
| 401 | `AUTHENTICATION_ERROR` | Invalid/missing credentials | Check API key, secret, and signature |
| 401 | `Invalid Signature` | Incorrect JWT signing | Verify private key format and algorithm |
| 403 | `FORBIDDEN` | Insufficient permissions | Check API key scopes/permissions |
| 403 | `AUTHENTICATION_ERROR` | IP not whitelisted | Add IP to API key IP whitelist |

### JWT Troubleshooting

```python
def debug_jwt(jwt_token):
    """Decode and inspect JWT (for debugging only)"""
    import json
    import base64

    # JWT format: header.payload.signature
    parts = jwt_token.split('.')

    # Decode payload (add padding if needed)
    payload = parts[1]
    padding = 4 - len(payload) % 4
    payload += '=' * padding

    decoded = base64.urlsafe_b64decode(payload)
    print(json.dumps(json.loads(decoded), indent=2))
```

### HMAC Troubleshooting

**Python (Safe - Handles Binary Correctly):**
```python
# Common issues:
# 1. Timestamp format - must be Unix timestamp (seconds), not milliseconds
timestamp = str(int(time.time()))  # ✅ Correct
timestamp = str(int(time.time() * 1000))  # ❌ Wrong - milliseconds

# 2. Message format - must be exact (no spaces!)
message = f"{timestamp}{method}{path}{body}"
# ❌ Wrong: f"{timestamp} {method} {path} {body}" (spaces!)

# 3. Base64 secret - must be decoded before HMAC
secret_bytes = base64.b64decode(api_secret)  # ✅ Correct (Python handles binary)
secret_bytes = api_secret.encode()  # ❌ Wrong - not decoded

# 4. Empty body - must be empty string, not null
body_str = ""  # ✅ Correct
body_str = None  # ❌ Wrong
```

**Shell/cURL (Critical NULL-Byte Safety Issue):**
```bash
# ⚠️ CRITICAL: Shell variables TRUNCATE on NULL bytes!

# ❌ BROKEN - Silently corrupts signatures:
#    SECRET_DEC=$(echo ... | base64 -d)
#    openssl dgst -sha256 -hmac "$SECRET_DEC"
#    ^ 32-byte secret becomes 30 bytes after NULL byte
#    ^ Signatures are WRONG but command succeeds (silent failure!)
#    ^ API returns 401 Unauthorized with no clear reason

# ✅ CORRECT - Use hex encoding to avoid shell variables holding binary:
#    KEY_HEX=$(printf %s "$SECRET_B64" | base64 -d | xxd -p -c256 | tr -d '\n')
#    openssl dgst -sha256 -mac HMAC -macopt hexkey:$KEY_HEX
#    ^ Hex is safe in variables (only 0-9a-f characters)
#    ^ Handles any byte values including NULLs
#    ^ Guaranteed correct signatures

# Alternative: Write decoded secret to temp file
#    TEMP_KEY=$(mktemp)
#    printf %s "$SECRET_B64" | base64 -d > "$TEMP_KEY"
#    openssl dgst -sha256 -hmac "$(cat "$TEMP_KEY")"
#    rm "$TEMP_KEY"
```

**Other Common Issues:**
```bash
# 1. ❌ Using process substitution (passes literal /dev/fd/... string):
#    openssl dgst -sha256 -hmac <(echo ... | base64 -d)

# 2. ❌ Wrong message format (must be exactly: timestamp+method+path+body):
#    message = f"{timestamp} {method} {path} {body}"  # Spaces!

# 3. ❌ Milliseconds in timestamp instead of seconds
#    timestamp=$(date +%s%N)  # Wrong - adds nanoseconds
```

---

## 6. Authentication Method Comparison

| Feature | CDP/JWT | HMAC | OAuth2 |
|---------|---------|------|--------|
| **Setup Complexity** | Medium | Easy | Complex |
| **Token Expiry** | 2 min per request | N/A | 2 hours + 1.5yr refresh |
| **Perps Support** | ✅ (INTX only) | ❌ | ❌ (spot only) |
| **Spot Support** | ✅ | ✅ | ✅ |
| **Sandbox Support** | ❌ | ✅ | ❌ |
| **Multi-user** | Single user | Single user | ✅ Multiple users |
| **Rate Limit** | 30 req/sec | 30 req/sec | 30 req/sec |

---

## 7. Migration Guide

### From HMAC to CDP/JWT (Spot → Perps)

```python
# Old HMAC auth (spot)
old_auth = HMACAuth(api_key, api_secret, passphrase)

# New CDP auth (perps)
new_auth = CDPAuth(api_key, private_key_pem)  # New key format!

# Detection logic
if api_key.startswith("organizations/"):
    auth = CDPAuth(api_key, private_key_pem)  # CDP keys always start with this
else:
    auth = HMACAuth(api_key, api_secret, passphrase)  # HMAC keys are hex strings
```

### From HMAC to OAuth2 (Single-user → Multi-user)

```python
# Get initial tokens from user authorization
tokens = oauth2.exchange_code_for_token(authorization_code)

# Store and refresh tokens for future use
token_manager = TokenManager(tokens["refresh_token"])

# Use in requests
access_token = token_manager.get_valid_token()
```

---

## Best Practices

✅ **DO**:
- Rotate API keys regularly
- Store secrets in environment variables (never in code)
- Use HTTPS/WSS only (never HTTP/WS)
- Validate SSL certificates in production
- Implement exponential backoff on rate limit errors
- Monitor rate limit headers to stay under limits

❌ **DON'T**:
- Commit API keys to version control
- Share API secrets via email or Slack
- Log full API keys or secrets (truncate: `****...key123`)
- Use the same key for multiple environments
- Ignore rate limit warnings (429 responses)
- Disable SSL verification

---

## Maintenance & Versioning

- **Last Updated**: 2025-10-19
- **Verification Schedule**: Quarterly
- **Changes**: Check [Changelog](https://docs.cdp.coinbase.com/coinbase-app/introduction/changelog)

---

## See Also

- [coinbase_api_endpoints.md](coinbase_api_endpoints.md) - REST API endpoints
- [coinbase_websocket_reference.md](coinbase_websocket_reference.md) - WebSocket authentication
- [coinbase_quick_reference.md](coinbase_quick_reference.md) - Quick reference
- [coinbase_complete.md](coinbase_complete.md) - Complete integration guide
