# Coinbase API Setup Guide

## Step 1: Choose Your Environment

### Option A: Sandbox (Recommended for Testing)
- **URL**: https://public.sandbox.exchange.coinbase.com/
- **Purpose**: Safe testing without real money
- **Note**: Limited trading pairs and features

### Option B: Production
- **URL**: https://www.coinbase.com/
- **Purpose**: Real trading with actual funds
- **Warning**: Be careful - real money at risk!

## Step 2: Create API Keys

### For Sandbox:
1. Go to https://public.sandbox.exchange.coinbase.com/
2. Sign in with your Coinbase account
3. Navigate to **Settings** → **API**
4. Click **New API Key**
5. Configure permissions:
   - ✅ View (required)
   - ✅ Trade (for placing orders)
   - ❌ Transfer (not needed, safer to leave off)
6. **IMPORTANT**: Save these immediately (shown only once):
   - API Key
   - API Secret (base64 encoded string)
   - Passphrase (if shown)

### For Production:
1. Go to https://www.coinbase.com/settings/api
2. Click **New API Key**
3. Choose **Advanced Trade API** (not the legacy Exchange API)
4. Set permissions carefully:
   - ✅ View
   - ✅ Trade (if you want to place orders)
   - ❌ Transfer (keep disabled for safety)
5. Save credentials securely

## Step 3: Configure Environment

Create or update your `.env` file:

```bash
# Broker Selection
BROKER=coinbase

# Environment (1 for sandbox, 0 for production)
COINBASE_SANDBOX=1

# Your API Credentials
COINBASE_API_KEY=your-api-key-here
COINBASE_API_SECRET=your-base64-secret-here
COINBASE_API_PASSPHRASE=  # Often blank for Advanced Trade API

# Optional: Override API URLs
# COINBASE_API_BASE=https://api-public.sandbox.exchange.coinbase.com  # Sandbox
# COINBASE_API_BASE=https://api.exchange.coinbase.com  # Production

# Trading Settings (for testing)
COINBASE_ENABLE_DERIVATIVES=0
COINBASE_RUN_ORDER_TESTS=0
COINBASE_ORDER_SYMBOL=BTC-USD
COINBASE_TEST_LIMIT_PRICE=10
COINBASE_TEST_QTY=0.001
```

## Step 4: Common Issues and Solutions

### Issue 1: "Invalid API Key"
**Solution**: 
- Make sure you're using the correct environment (sandbox vs production)
- Check that COINBASE_SANDBOX matches your key source
- Verify no extra spaces in your API key

### Issue 2: "Invalid Signature" 
**Solution**:
- The API secret should be the base64 string exactly as shown by Coinbase
- Don't decode or modify the secret
- Make sure there are no line breaks in the secret

### Issue 3: "Permission Denied"
**Solution**:
- Ensure your API key has the required permissions (View at minimum)
- For sandbox, make sure you're using sandbox URLs

### Issue 4: "Not Found" errors
**Solution**:
- Sandbox has limited products (usually only BTC-USD, ETH-USD)
- Check available products first with the test script

## Step 5: Test Your Configuration

Run the verification script:
```bash
python scripts/test_coinbase_connection.py
```

This will test:
1. API key format
2. Connection to correct environment
3. Authentication
4. Basic API operations

## Security Best Practices

1. **Never commit credentials**: 
   - Add `.env` to `.gitignore`
   - Use `.env.local` for local overrides

2. **Use minimal permissions**:
   - Start with View only
   - Add Trade only when needed
   - Never enable Transfer unless absolutely necessary

3. **Use sandbox first**:
   - Test everything in sandbox
   - Only move to production when confident

4. **Rotate keys regularly**:
   - Delete unused keys
   - Create new keys periodically

5. **Monitor API usage**:
   - Check Coinbase dashboard for unusual activity
   - Set up alerts if available

## Advanced Trade API vs Exchange API

**Use Advanced Trade API (Recommended)**:
- Path: `/api/v3/brokerage/*`
- Modern, actively maintained
- Better documentation
- No passphrase required usually

**Legacy Exchange API (Deprecated)**:
- Path: `/products`, `/orders`
- Being phased out
- Requires passphrase
- Limited support

## Troubleshooting Commands

```bash
# Check if credentials are loaded
python -c "import os; print('API Key set:', bool(os.getenv('COINBASE_API_KEY')))"

# Test basic connection
python scripts/test_coinbase_connection.py

# Run smoke test
python scripts/test_coinbase_basic.py

# Enable debug logging
export COINBASE_DEBUG=1
python scripts/test_coinbase_connection.py
```

## Need Help?

1. Check Coinbase API Status: https://status.coinbase.com/
2. API Documentation: https://docs.cloud.coinbase.com/advanced-trade-api/
3. Common errors: Look at `src/bot_v2/features/brokerages/coinbase/errors.py`