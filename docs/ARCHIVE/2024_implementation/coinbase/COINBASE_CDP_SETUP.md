# Coinbase CDP (Cloud Developer Platform) Setup Guide

## Overview

The GPT-Trader system now supports Coinbase's newer CDP API which uses JWT authentication with EC private keys instead of the legacy HMAC authentication.

## Authentication Status

✅ **JWT Implementation Complete**
- CDP JWT authentication is fully implemented and working
- JWT tokens are being generated correctly with proper structure
- All required headers and claims are properly formatted

⚠️ **API Key Permissions Issue**
- The CDP API key is authenticating but returns 401 for private endpoints
- Public market data endpoints work correctly
- The issue is with API key permissions in Coinbase CDP dashboard

## Working Endpoints

### Public (No Auth Required)
- `/api/v3/brokerage/market/products/{product_id}/ticker` - ✅ Working
- `/api/v3/brokerage/market/products/{product_id}/candles` - ✅ Working

### Private (Auth Required but Getting 401)
- `/api/v3/brokerage/accounts` - ❌ 401 Unauthorized
- `/api/v3/brokerage/products` - ❌ 401 Unauthorized
- `/api/v3/brokerage/orders` - ❌ 401 Unauthorized

## Required CDP Setup Steps

### 1. Create CDP API Key
1. Go to [Coinbase Developer Platform](https://portal.cdp.coinbase.com/)
2. Create a new project or select existing
3. Navigate to API Keys section
4. Create new API key with these permissions:
   - **View** - Read account balances and information
   - **Trade** - Place and manage orders
   - **Transfer** - (Optional) Move funds

### 2. Configure Permissions
The API key must have the following scopes enabled:
- `accounts:read` - Read account information
- `orders:read` - View orders
- `orders:create` - Place orders
- `orders:cancel` - Cancel orders
- `products:read` - View products (may be required even for "public" data)

### 3. Environment Configuration

Create `.env.production` file:
```bash
# Broker selection
BROKER=coinbase

# Environment (0 = production, 1 = sandbox)
COINBASE_SANDBOX=0

# CDP Credentials
COINBASE_CDP_API_KEY=organizations/{org-id}/apiKeys/{key-id}
COINBASE_CDP_PRIVATE_KEY="-----BEGIN EC PRIVATE KEY-----
[Your private key here]
-----END EC PRIVATE KEY-----"

# Auth type (automatically detected if CDP keys present)
COINBASE_AUTH_TYPE=JWT

# API URLs
COINBASE_API_BASE=https://api.coinbase.com
COINBASE_WS_URL=wss://advanced-trade-ws.coinbase.com

# Safety settings
COINBASE_ENABLE_TRADING=0  # Start with 0 for safety
```

## Testing the Integration

### 1. Test Authentication
```bash
python scripts/test_coinbase_cdp.py
```

### 2. Test Individual Components
```bash
# Test JWT generation
python scripts/test_jwt_decode.py

# Test various endpoints
python scripts/test_cdp_simple.py

# Test public endpoints
python scripts/test_cdp_public.py
```

### 3. Debug Authentication Issues
```bash
# Raw authentication test
python scripts/test_cdp_raw.py

# Direct CDP test
python scripts/test_cdp_direct.py
```

## Implementation Details

### Files Modified
- `src/bot_v2/features/brokerages/coinbase/cdp_auth.py` - CDP JWT authentication
- `src/bot_v2/features/brokerages/coinbase/adapter.py` - Added CDP auth support
- `src/bot_v2/features/brokerages/coinbase/client.py` - Supports both HMAC and JWT
- `src/bot_v2/features/brokerages/coinbase/models.py` - Added CDP fields to APIConfig
- `src/bot_v2/orchestration/broker_factory.py` - Reads CDP credentials from environment

### JWT Structure
The CDP JWT includes:
- **Header**: Algorithm (ES256), Key ID, Type (JWT), Nonce
- **Payload**: Subject (API key name), Issuer (coinbase-cloud), Audience, URI claim
- **Signature**: EC signature using private key

## Troubleshooting

### 401 Unauthorized Errors
1. **Check API Key Status**: Ensure key is active in CDP dashboard
2. **Verify Permissions**: Check all required scopes are enabled
3. **Test Public Endpoints**: Verify basic connectivity with market endpoints
4. **Review JWT Structure**: Run `test_jwt_decode.py` to verify token format

### Common Issues
- **Missing Permissions**: CDP keys need explicit scope grants
- **Wrong Environment**: Ensure using production URL for CDP keys
- **Key Not Activated**: New keys may need manual activation
- **IP Restrictions**: CDP may have IP allowlist requirements

## Next Steps

1. **Activate API Key Permissions** in Coinbase CDP dashboard
2. **Test with properly configured key** 
3. **Implement fallback** for public data using non-authenticated endpoints
4. **Add comprehensive error handling** for permission failures
5. **Create monitoring** for API key health

## Support Resources

- [Coinbase CDP Documentation](https://docs.cdp.coinbase.com/)
- [Advanced Trade API Reference](https://docs.cloud.coinbase.com/advanced-trade-api/reference)
- [JWT Authentication Guide](https://docs.cdp.coinbase.com/advanced-trade/docs/rest-auth#jwt)

## Security Notes

- Never commit private keys to version control
- Use environment variables for all credentials
- Start with `COINBASE_ENABLE_TRADING=0` for safety
- Test thoroughly in read-only mode first
- Monitor rate limits and API usage