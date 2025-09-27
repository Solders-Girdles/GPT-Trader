# Coinbase CDP API Key Permission Issue

## Current Status

### ✅ What's Working:
1. **JWT Authentication**: Correctly implemented and tokens are valid
2. **API Key Recognition**: Coinbase recognizes the API key 
3. **Public Endpoints**: Server time endpoint works (`/api/v3/brokerage/time`)
4. **Market Data**: Public market endpoints work without auth

### ❌ What's Not Working:
All authenticated endpoints return `401 Unauthorized`:
- `/api/v3/brokerage/accounts` - Account information
- `/api/v3/brokerage/portfolios` - Portfolio listing
- `/api/v3/brokerage/user` - User information
- `/api/v3/brokerage/best_bid_ask` - Market data (requires auth)
- `/api/v3/brokerage/products` - Product listing (requires auth)

## The Issue

Your CDP API key has **all permissions enabled** in the dashboard:
- ✅ View (read only)
- ✅ Trade (execute trades)
- ✅ Transfer (initiate transfers)
- ✅ Exports (export private key)
- ✅ Manage (modify policies)

However, the key **cannot access any trading account data**.

## Root Cause

This is a known issue with Coinbase CDP keys. Even with all permissions enabled, the key needs to be **linked to a specific portfolio/account**. 

### Possible Causes:

1. **Portfolio Linking Required**
   - CDP keys must be explicitly linked to a trading portfolio
   - This is a separate step from granting permissions
   - Check if there's a "Link Portfolio" or "Assign Account" option in CDP

2. **Wrong API Key Type**
   - CDP has multiple key types (Platform, Commerce, Advanced Trade)
   - You need specifically an "Advanced Trade API" key
   - Platform/Commerce keys won't work with Advanced Trade endpoints

3. **Account Verification**
   - Your Coinbase account might need additional verification
   - CDP access might require business/institutional account

4. **Regional Restrictions**
   - Some CDP features are region-locked
   - Advanced Trade API might not be available in your region

## Solution Steps

### Option 1: Link Portfolio in CDP
1. Go to [CDP Dashboard](https://portal.cdp.coinbase.com/)
2. Navigate to your API key settings
3. Look for "Portfolio Management" or "Account Linking"
4. Link the key to your default trading portfolio
5. Save and test again

### Option 2: Create New Advanced Trade Key
1. In CDP, specifically select "Advanced Trade API" when creating key
2. Ensure it's not a "Platform API" or "Commerce API" key
3. During creation, select which portfolio to link
4. Enable all required permissions
5. Test immediately after creation

### Option 3: Use Legacy API (Temporary)
While resolving CDP issues, you can use:
1. Legacy Exchange API with your sandbox credentials (working)
2. Or create new Advanced Trade credentials (non-CDP)
3. These use HMAC authentication instead of JWT

### Option 4: Contact Support
If all permissions are enabled but still getting 401:
1. This is likely a CDP platform issue
2. Contact Coinbase support
3. Reference: "CDP API key with all permissions cannot access Advanced Trade endpoints"
4. Provide your Organization ID: `5184a9ea-2cec-4a66-b00e-7cf6daaf048e`

## Technical Details

### Working Request Example:
```
GET /api/v3/brokerage/time
Response: 200 OK
{"iso":"2025-08-24T10:43:19Z","epoch":1756032199}
```

### Failing Request Example:
```
GET /api/v3/brokerage/accounts
Response: 401 Unauthorized
Body: "Unauthorized"
```

### JWT Token Structure (Correct):
```json
{
  "header": {
    "alg": "ES256",
    "kid": "organizations/.../apiKeys/...",
    "typ": "JWT",
    "nonce": "..."
  },
  "payload": {
    "sub": "organizations/.../apiKeys/...",
    "iss": "coinbase-cloud",
    "aud": ["retail_rest_api_proxy"],
    "uri": "GET /api/v3/brokerage/accounts",
    "nbf": ...,
    "exp": ...
  }
}
```

## Workaround

While this is being resolved, the system can operate in:

### Market Data Only Mode
- Use public endpoints for prices
- Paper trade with simulated execution
- Test strategies without real orders

### Sandbox Mode
- Switch back to sandbox credentials
- Use legacy Exchange API
- Full functionality for testing

## Next Steps

1. **Check CDP Dashboard** for portfolio linking options
2. **Try creating a new API key** specifically for Advanced Trade
3. **Contact Coinbase support** if the issue persists
4. **Use sandbox mode** for development in the meantime

## References

- [CDP Documentation](https://docs.cdp.coinbase.com/)
- [Advanced Trade API](https://docs.cloud.coinbase.com/advanced-trade-api/docs)
- [JWT Authentication](https://docs.cdp.coinbase.com/advanced-trade/docs/rest-auth#jwt)
- Organization ID: `5184a9ea-2cec-4a66-b00e-7cf6daaf048e`
- Key ID: `7e24f68f-9e72-4d19-9418-86ee7d65bcb4`