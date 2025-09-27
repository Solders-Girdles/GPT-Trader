#!/bin/bash
# Fix environment configuration for Coinbase Perpetuals trading
# Run this script to resolve all preflight check failures

set -e  # Exit on error

echo "🔧 ENVIRONMENT CONFIGURATION FIX"
echo "================================="

# 1. Set required environment variables
echo "✅ Setting environment variables..."
export COINBASE_API_MODE=advanced
export COINBASE_AUTH_TYPE=JWT
export COINBASE_ENABLE_DERIVATIVES=1
export COINBASE_SANDBOX=1  # Start with sandbox

echo "   COINBASE_API_MODE=$COINBASE_API_MODE"
echo "   COINBASE_AUTH_TYPE=$COINBASE_AUTH_TYPE"
echo "   COINBASE_ENABLE_DERIVATIVES=$COINBASE_ENABLE_DERIVATIVES"
echo "   COINBASE_SANDBOX=$COINBASE_SANDBOX"

# 2. Handle private key securely
echo ""
echo "🔐 Securing private key..."

# Create secure directory if it doesn't exist
SECURE_DIR="$HOME/.coinbase/keys"
mkdir -p "$SECURE_DIR"
chmod 700 "$SECURE_DIR"

KEY_FILE="$SECURE_DIR/cdp_private_key.pem"

# Check if key is in environment
if [ -n "$COINBASE_CDP_PRIVATE_KEY" ]; then
    echo "   Moving private key from environment to file..."
    echo "$COINBASE_CDP_PRIVATE_KEY" > "$KEY_FILE"
    chmod 400 "$KEY_FILE"
    export COINBASE_CDP_PRIVATE_KEY_PATH="$KEY_FILE"
    unset COINBASE_CDP_PRIVATE_KEY
    echo "   ✅ Private key secured at: $KEY_FILE"
elif [ -n "$COINBASE_CDP_PRIVATE_KEY_PATH" ]; then
    echo "   Private key already using file path: $COINBASE_CDP_PRIVATE_KEY_PATH"
    # Ensure proper permissions
    if [ -f "$COINBASE_CDP_PRIVATE_KEY_PATH" ]; then
        chmod 400 "$COINBASE_CDP_PRIVATE_KEY_PATH"
        echo "   ✅ Permissions set to 400"
    fi
else
    echo "   ⚠️  WARNING: No private key found in environment"
    echo "   Please set COINBASE_CDP_PRIVATE_KEY or COINBASE_CDP_PRIVATE_KEY_PATH"
fi

# 3. Fix .env file permissions
echo ""
echo "📝 Fixing .env file permissions..."
if [ -f ".env" ]; then
    chmod 600 .env
    echo "   ✅ .env permissions set to 600"
else
    echo "   ℹ️  No .env file found (OK if using environment variables)"
fi

# 4. Install missing dependencies
echo ""
echo "📦 Installing dependencies..."
pip install python-dotenv websockets coinbase -q
echo "   ✅ Dependencies installed"

# 5. Test WebSocket connectivity
echo ""
echo "🌐 Testing network connectivity..."

# Test DNS resolution
echo "   Testing DNS resolution..."
if nslookup advanced-trade-ws.coinbase.com > /dev/null 2>&1; then
    echo "   ✅ DNS resolution OK"
else
    echo "   ❌ DNS resolution failed for advanced-trade-ws.coinbase.com"
fi

# Test TLS connection
echo "   Testing TLS connection..."
if echo | openssl s_client -connect advanced-trade-ws.coinbase.com:443 -servername advanced-trade-ws.coinbase.com 2>/dev/null | grep -q "SSL handshake"; then
    echo "   ✅ TLS connection OK"
else
    echo "   ⚠️  TLS connection test inconclusive"
fi

# Alternative WebSocket endpoints to test
WS_ENDPOINTS=(
    "wss://ws-direct.sandbox.coinbase.com"
    "wss://ws-direct.coinbase.com"
    "wss://advanced-trade-ws.coinbase.com"
)

echo "   Testing WebSocket endpoints..."
for endpoint in "${WS_ENDPOINTS[@]}"; do
    host=$(echo $endpoint | sed 's|wss://||' | sed 's|/.*||')
    if ping -c 1 -W 2 $host > /dev/null 2>&1; then
        echo "   ✅ $host is reachable"
    else
        echo "   ❌ $host is not reachable"
    fi
done

# 6. Check for proxy/firewall issues
echo ""
echo "🔥 Checking proxy/firewall configuration..."
if [ -n "$HTTP_PROXY" ] || [ -n "$HTTPS_PROXY" ]; then
    echo "   ⚠️  Proxy detected. Setting NO_PROXY for Coinbase..."
    export NO_PROXY="${NO_PROXY},.coinbase.com"
    echo "   NO_PROXY=$NO_PROXY"
fi

# 7. Create configuration summary
echo ""
echo "📋 Creating configuration summary..."
cat > environment_config.txt << EOF
Environment Configuration Summary
=================================
Date: $(date)

Environment Variables:
----------------------
COINBASE_API_MODE=$COINBASE_API_MODE
COINBASE_AUTH_TYPE=$COINBASE_AUTH_TYPE
COINBASE_ENABLE_DERIVATIVES=$COINBASE_ENABLE_DERIVATIVES
COINBASE_SANDBOX=$COINBASE_SANDBOX
COINBASE_CDP_API_KEY=$COINBASE_CDP_API_KEY
COINBASE_CDP_PRIVATE_KEY_PATH=$COINBASE_CDP_PRIVATE_KEY_PATH

Security:
---------
Private Key Location: $KEY_FILE
Private Key Permissions: $(stat -c %a "$KEY_FILE" 2>/dev/null || stat -f %A "$KEY_FILE" 2>/dev/null || echo "N/A")
.env Permissions: $(stat -c %a .env 2>/dev/null || stat -f %A .env 2>/dev/null || echo "N/A")

Network:
--------
DNS Resolution: $(nslookup advanced-trade-ws.coinbase.com > /dev/null 2>&1 && echo "OK" || echo "FAILED")
Proxy: ${HTTP_PROXY:-None}
NO_PROXY: ${NO_PROXY:-None}

Dependencies:
-------------
python-dotenv: $(pip show python-dotenv 2>/dev/null | grep Version || echo "Not installed")
websockets: $(pip show websockets 2>/dev/null | grep Version || echo "Not installed")
coinbase: $(pip show coinbase 2>/dev/null | grep Version || echo "Not installed")
EOF

echo "   ✅ Configuration saved to environment_config.txt"

# 8. Export variables for current session
echo ""
echo "💾 Saving environment for current session..."
cat > set_env.sh << 'EOF'
#!/bin/bash
# Source this file to set environment variables
export COINBASE_API_MODE=advanced
export COINBASE_AUTH_TYPE=JWT
export COINBASE_ENABLE_DERIVATIVES=1
export COINBASE_SANDBOX=1
EOF

if [ -n "$COINBASE_CDP_API_KEY" ]; then
    echo "export COINBASE_CDP_API_KEY='$COINBASE_CDP_API_KEY'" >> set_env.sh
fi

if [ -n "$COINBASE_CDP_PRIVATE_KEY_PATH" ]; then
    echo "export COINBASE_CDP_PRIVATE_KEY_PATH='$COINBASE_CDP_PRIVATE_KEY_PATH'" >> set_env.sh
fi

if [ -n "$NO_PROXY" ]; then
    echo "export NO_PROXY='$NO_PROXY'" >> set_env.sh
fi

chmod +x set_env.sh
echo "   ✅ Environment script saved to set_env.sh"
echo "   Run: source set_env.sh"

echo ""
echo "================================="
echo "✅ ENVIRONMENT CONFIGURATION COMPLETE"
echo ""
echo "Next steps:"
echo "1. Source the environment: source set_env.sh"
echo "2. Run preflight check: python scripts/preflight_check.py"
echo "3. If all passes, proceed with demo trading"
echo ""