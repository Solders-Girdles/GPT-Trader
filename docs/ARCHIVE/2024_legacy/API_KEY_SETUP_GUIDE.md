---
status: deprecated
archived: 2024-12-31
reason: Pre-perpetuals documentation from Alpaca/equities era
---

# ⚠️ DEPRECATED DOCUMENT

This document is from the legacy Alpaca/Equities version of GPT-Trader and is no longer current.
The project has migrated to Coinbase Perpetual Futures.

For current documentation, see: [docs/README.md](/docs/README.md)

---


# API Key Setup Guide - Perps Trading System

## 🎯 Overview

This guide follows the project manager's guidelines for setting up API keys with proper naming conventions, least-privilege permissions, and environment separation.

## 📋 Key Naming Convention

**Format**: `perps-{env}-{bot}-{perm}-{yyyymm}`

**Examples**:
- `perps-prod-bot01-trade-202510`
- `perps-demo-botA-trade-202510`
- `perps-prod-monitor-read-202510`

## 🔑 Required Keys Per Environment

### Demo Environment (Sandbox)
- **Trade Key**: `perps-demo-bot01-trade-202510`
- **Monitor Key**: `perps-demo-monitor-read-202510`

### Production Environment
- **Trade Key**: `perps-prod-bot01-trade-202510`
- **Monitor Key**: `perps-prod-monitor-read-202510`

## 🌐 Key Creation URLs

### Demo (Sandbox)
- **URL**: https://public.sandbox.exchange.coinbase.com/
- **Purpose**: Testing with tiny notionals
- **API Type**: Advanced Trade (JWT/CDP) or Exchange (HMAC)

### Production
- **URL**: https://www.coinbase.com/settings/api
- **Purpose**: Real trading with full perps (CFM) support
- **API Type**: Advanced Trade (JWT/CDP)

## 🔧 Required Permissions

### For Both Environments
- ✅ **Accounts/Portfolios**: read
- ✅ **Orders**: read/write (place, cancel, modify)
- ✅ **Products/Market Data**: read
- ✅ **Fills/Positions**: read
- ✅ **Derivatives (CFM/perpetuals)**: read/write
- ❌ **Transfers/Withdrawals**: none (explicitly disable)

## 📝 Step-by-Step Creation Process

### Step 1: Create Trade Key
1. Go to the appropriate URL for your environment
2. Sign in to your Coinbase account
3. Navigate to **Settings** → **API**
4. Click **"New API Key"**
5. Choose **"Advanced Trade API"** (not legacy Exchange)
6. Set permissions as listed above
7. Use the naming convention: `perps-{env}-bot01-trade-{yyyymm}`
8. Save credentials immediately (shown only once)

### Step 2: Create Monitor Key
1. Repeat the process above
2. Use naming convention: `perps-{env}-monitor-read-{yyyymm}`
3. Set **read-only permissions** (no trade/write access)
4. Save credentials immediately

## 🔒 Security Checklist

### Key Creation
- ✅ IP allowlist enabled (if available)
- ✅ Transfer/withdrawal permissions disabled
- ✅ Keys rotated every 90 days
- ✅ Credentials stored securely

### Environment Separation
- ✅ No reuse across environments
- ✅ No mixing demo/prod credentials
- ✅ Separate configuration files

### Access Control
- ✅ Least-privilege permissions
- ✅ Separate trade and monitor keys
- ✅ No transfer capabilities

## 🚀 Automated Setup

Run the automated setup script:
```bash
python setup_complete_api_keys.py
```

This script will:
1. Guide you through environment selection
2. Provide step-by-step key creation instructions
3. Collect all credentials securely
4. Create environment-specific configuration files
5. Validate the setup

## 📁 Configuration Files

The setup creates these files:
- `.env.demo` - Demo environment configuration
- `.env.prod` - Production environment configuration
- `.env` - Main configuration (points to active environment)

## 🧪 Testing & Validation

### Test Demo Environment
```bash
cp .env.demo .env
python scripts/test_coinbase_connection.py
```

### Test Production Environment
```bash
cp .env.prod .env
python scripts/test_coinbase_connection.py
```

### Comprehensive Validation
```bash
python scripts/validate_critical_fixes_v2.py
```

## ⚠️ Important Reminders

### Security
- **Never commit** `.env` files to git
- **Rotate keys** every 90 days
- **Use IP allowlisting** in production
- **Monitor key usage** and permissions

### Environment Management
- **Start with demo** for testing
- **Use tiny notionals** in demo
- **Enable safety features** in demo
- **Conservative sizing** in production

### Key Rotation
- **Document rotation** in CHANGELOG/RUNBOOK
- **Create new keys** before revoking old ones
- **Test new keys** before switching
- **Update configuration** files

## 🔍 Troubleshooting

### Common Issues
1. **401 Unauthorized**: Check API key permissions
2. **403 Forbidden**: Verify IP allowlist settings
3. **Rate limiting**: Check request frequency
4. **Sandbox issues**: Ensure correct sandbox URLs

### Validation Commands
```bash
# Test connection
python scripts/test_coinbase_connection.py

# Check configuration
python test_current_setup.py

# Validate fixes
python scripts/validate_critical_fixes_v2.py
```

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review the validation output
3. Verify key permissions in Coinbase dashboard
4. Test with the connection validation scripts
