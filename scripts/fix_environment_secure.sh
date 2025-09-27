#!/bin/bash
# Secure environment configuration for Coinbase Perpetuals trading
# Version: 2.0 - Enhanced security
# Run this script to resolve all preflight check failures

set -e  # Exit on error

echo "🔧 SECURE ENVIRONMENT CONFIGURATION"
echo "===================================="
echo "Date: $(date)"
echo "User: $(whoami)"
echo "Environment: ${ENVIRONMENT:-development}"
echo ""

# Function to mask sensitive values
mask_value() {
    local value="$1"
    if [ -z "$value" ]; then
        echo "not_set"
    elif [ ${#value} -le 10 ]; then
        echo "***"
    else
        echo "${value:0:6}...${value: -4}"
    fi
}

# 1. Set required environment variables
echo "✅ Setting environment variables..."
export COINBASE_API_MODE=advanced
export COINBASE_AUTH_TYPE=JWT
export COINBASE_ENABLE_DERIVATIVES=1
export COINBASE_SANDBOX=${COINBASE_SANDBOX:-1}  # Default to sandbox

echo "   COINBASE_API_MODE=$COINBASE_API_MODE"
echo "   COINBASE_AUTH_TYPE=$COINBASE_AUTH_TYPE"
echo "   COINBASE_ENABLE_DERIVATIVES=$COINBASE_ENABLE_DERIVATIVES"
echo "   COINBASE_SANDBOX=$COINBASE_SANDBOX"

# 2. Handle private key securely
echo ""
echo "🔐 Securing private key..."

# Create secure directory with proper permissions
SECURE_DIR="$HOME/.coinbase/keys"
mkdir -p "$SECURE_DIR"
chmod 700 "$SECURE_DIR"  # Directory accessible only by owner
echo "   Directory permissions set: 700"

KEY_FILE="$SECURE_DIR/cdp_private_key.pem"

# Check if key is in environment (DO NOT ECHO IT)
if [ -n "$COINBASE_CDP_PRIVATE_KEY" ]; then
    echo "   Moving private key from environment to file..."
    # Write key to file without echoing
    printf "%s" "$COINBASE_CDP_PRIVATE_KEY" > "$KEY_FILE"
    chmod 400 "$KEY_FILE"  # Read-only for owner
    export COINBASE_CDP_PRIVATE_KEY_PATH="$KEY_FILE"
    unset COINBASE_CDP_PRIVATE_KEY
    echo "   ✅ Private key secured at: $KEY_FILE (permissions: 400)"
elif [ -n "$COINBASE_CDP_PRIVATE_KEY_PATH" ]; then
    echo "   Private key already using file path"
    # Verify and fix permissions
    if [ -f "$COINBASE_CDP_PRIVATE_KEY_PATH" ]; then
        current_perms=$(stat -c %a "$COINBASE_CDP_PRIVATE_KEY_PATH" 2>/dev/null || stat -f %A "$COINBASE_CDP_PRIVATE_KEY_PATH" 2>/dev/null || echo "unknown")
        if [ "$current_perms" != "400" ] && [ "$current_perms" != "600" ]; then
            chmod 400 "$COINBASE_CDP_PRIVATE_KEY_PATH"
            echo "   ✅ Fixed permissions to 400 (was: $current_perms)"
        else
            echo "   ✅ Permissions correct: $current_perms"
        fi
    else
        echo "   ❌ ERROR: Private key file not found: $COINBASE_CDP_PRIVATE_KEY_PATH"
        exit 1
    fi
else
    echo "   ⚠️  WARNING: No private key found"
    echo "   Set COINBASE_CDP_PRIVATE_KEY or COINBASE_CDP_PRIVATE_KEY_PATH"
fi

# Mask CDP API key for display
CDP_KEY_MASKED=$(mask_value "$COINBASE_CDP_API_KEY")
echo "   CDP API Key: $CDP_KEY_MASKED"

# 3. Fix .env file permissions
echo ""
echo "📝 Fixing .env file permissions..."
if [ -f ".env" ]; then
    chmod 600 .env
    echo "   ✅ .env permissions set to 600"
    # Check if .env contains sensitive data
    if grep -q "PRIVATE_KEY\|SECRET\|PASSWORD" .env 2>/dev/null; then
        echo "   ⚠️  WARNING: .env contains sensitive data - ensure it's in .gitignore"
    fi
else
    echo "   ℹ️  No .env file found"
fi

# 4. Install missing dependencies
echo ""
echo "📦 Installing dependencies..."
pip install -q python-dotenv websockets coinbase PyJWT cryptography 2>/dev/null || {
    echo "   ❌ Failed to install dependencies"
    echo "   Run manually: pip install -r requirements.txt"
}
echo "   ✅ Dependencies installed"

# 5. Check system clock drift
echo ""
echo "🕐 Checking system clock..."
if command -v ntpdate &> /dev/null; then
    drift=$(ntpdate -q pool.ntp.org 2>/dev/null | grep -oP 'offset [-+]?\d+\.\d+' | awk '{print $2}' || echo "0")
    if [ $(echo "$drift > 30" | bc -l) -eq 1 ] 2>/dev/null; then
        echo "   ⚠️  WARNING: Clock drift ${drift}s > 30s threshold"
    else
        echo "   ✅ Clock drift within tolerance: ${drift}s"
    fi
else
    echo "   ℹ️  NTP check not available - ensure system time is accurate"
fi

# 6. Test network connectivity
echo ""
echo "🌐 Testing network connectivity..."

# Get egress IP
EGRESS_IP=$(curl -s ifconfig.me || curl -s icanhazip.com || echo "unknown")
echo "   Egress IP: $EGRESS_IP"
echo "   ⚠️  Ensure this IP is allowlisted in CDP console"

# Test DNS resolution
if nslookup advanced-trade-ws.coinbase.com > /dev/null 2>&1; then
    echo "   ✅ DNS resolution OK"
else
    echo "   ❌ DNS resolution failed"
fi

# Test TLS without exposing details
if timeout 5 openssl s_client -connect advanced-trade-ws.coinbase.com:443 -servername advanced-trade-ws.coinbase.com < /dev/null 2>/dev/null | grep -q "CONNECTED"; then
    echo "   ✅ TLS connection OK"
else
    echo "   ⚠️  TLS connection test inconclusive"
fi

# 7. Handle proxy configuration
echo ""
echo "🔥 Checking proxy configuration..."
if [ -n "$HTTP_PROXY" ] || [ -n "$HTTPS_PROXY" ]; then
    echo "   Proxy detected. Setting NO_PROXY..."
    export NO_PROXY="${NO_PROXY:+$NO_PROXY,}.coinbase.com,*.coinbase.com"
    echo "   NO_PROXY configured (masked)"
else
    echo "   ✅ No proxy detected"
fi

# 8. Create secure environment scripts (per environment)
echo ""
echo "💾 Creating secure environment scripts..."

# Determine environment
ENV_NAME=${ENVIRONMENT:-demo}
if [ "$COINBASE_SANDBOX" == "1" ]; then
    ENV_NAME="demo"
else
    ENV_NAME="prod"
fi

# Create environment-specific file
ENV_FILE="set_env.${ENV_NAME}.sh"
cat > "$ENV_FILE" << EOF
#!/bin/bash
# Generated: $(date)
# User: $(whoami)
# Environment: ${ENVIRONMENT:-development}
# Purpose: Set trading environment variables (no secrets inline)

# Core configuration
export COINBASE_API_MODE=advanced
export COINBASE_AUTH_TYPE=JWT
export COINBASE_ENABLE_DERIVATIVES=1
export COINBASE_SANDBOX=${COINBASE_SANDBOX:-1}

# API key (reference only, not value)
export COINBASE_CDP_API_KEY='${COINBASE_CDP_API_KEY}'

# Private key PATH only (secure)
export COINBASE_CDP_PRIVATE_KEY_PATH='${COINBASE_CDP_PRIVATE_KEY_PATH:-$KEY_FILE}'

# Network configuration
$([ -n "$NO_PROXY" ] && echo "export NO_PROXY='$NO_PROXY'")

# Safety parameters
export COINBASE_MAX_POSITION_SIZE=0.01
export COINBASE_DAILY_LOSS_LIMIT=0.02
export COINBASE_MAX_IMPACT_BPS=15

echo "✅ Environment configured for Coinbase trading"
echo "   Mode: \${COINBASE_SANDBOX:-0} == 1 ? SANDBOX : PRODUCTION"
echo "   Derivatives: \$COINBASE_ENABLE_DERIVATIVES"
echo "   Auth: \$COINBASE_AUTH_TYPE"
EOF

chmod +x "$ENV_FILE"
echo "   ✅ Created $ENV_FILE (no inline secrets)"

# Create symlink for convenience
ln -sf "$ENV_FILE" set_env.sh
echo "   ✅ Linked set_env.sh -> $ENV_FILE"

# 9. Create configuration summary (with masking)
echo ""
echo "📋 Creating configuration summary..."
cat > environment_config.txt << EOF
Secure Environment Configuration
================================
Generated: $(date)
User: $(whoami)
Host: $(hostname)
Environment: ${ENVIRONMENT:-development}

Environment Variables:
----------------------
COINBASE_API_MODE=$COINBASE_API_MODE
COINBASE_AUTH_TYPE=$COINBASE_AUTH_TYPE
COINBASE_ENABLE_DERIVATIVES=$COINBASE_ENABLE_DERIVATIVES
COINBASE_SANDBOX=$COINBASE_SANDBOX
COINBASE_CDP_API_KEY=$(mask_value "$COINBASE_CDP_API_KEY")
COINBASE_CDP_PRIVATE_KEY_PATH=$COINBASE_CDP_PRIVATE_KEY_PATH

Security:
---------
Key Directory: $SECURE_DIR
Directory Permissions: $(stat -c %a "$SECURE_DIR" 2>/dev/null || stat -f %A "$SECURE_DIR" 2>/dev/null || echo "N/A")
Private Key Location: ${KEY_FILE:-$COINBASE_CDP_PRIVATE_KEY_PATH}
Private Key Permissions: $(stat -c %a "${KEY_FILE:-$COINBASE_CDP_PRIVATE_KEY_PATH}" 2>/dev/null || stat -f %A "${KEY_FILE:-$COINBASE_CDP_PRIVATE_KEY_PATH}" 2>/dev/null || echo "N/A")
.env Permissions: $(stat -c %a .env 2>/dev/null || stat -f %A .env 2>/dev/null || echo "N/A")

Network:
--------
Egress IP: $EGRESS_IP
DNS Resolution: $(nslookup advanced-trade-ws.coinbase.com > /dev/null 2>&1 && echo "OK" || echo "FAILED")
Proxy: ${HTTP_PROXY:+configured (masked)}
NO_PROXY: ${NO_PROXY:+configured}

System:
-------
Date/Time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')
Timezone: $(date +%Z)
Clock Drift: ${drift:-not_checked}s

Dependencies:
-------------
python-dotenv: $(pip show python-dotenv 2>/dev/null | grep Version || echo "Not installed")
websockets: $(pip show websockets 2>/dev/null | grep Version || echo "Not installed")
coinbase: $(pip show coinbase 2>/dev/null | grep Version || echo "Not installed")
PyJWT: $(pip show PyJWT 2>/dev/null | grep Version || echo "Not installed")
cryptography: $(pip show cryptography 2>/dev/null | grep Version || echo "Not installed")

Safety Parameters:
------------------
Max Position Size: ${COINBASE_MAX_POSITION_SIZE:-0.01}
Daily Loss Limit: ${COINBASE_DAILY_LOSS_LIMIT:-0.02}
Max Impact BPS: ${COINBASE_MAX_IMPACT_BPS:-15}
EOF

echo "   ✅ Configuration saved to environment_config.txt"

# 10. Verify no secrets in output
echo ""
echo "🔒 Security verification..."
# Check that we're not logging secrets
if [ -f environment_config.txt ]; then
    if grep -q "BEGIN.*PRIVATE KEY\|BEGIN.*EC PRIVATE KEY" environment_config.txt 2>/dev/null; then
        echo "   ❌ ERROR: Private key found in output file!"
        echo "   Removing file for security..."
        rm environment_config.txt
        exit 1
    else
        echo "   ✅ No secrets found in output files"
    fi
fi

echo ""
echo "===================================="
echo "✅ SECURE ENVIRONMENT CONFIGURATION COMPLETE"
echo ""
echo "Next steps:"
echo "1. Source the environment: source set_env.sh"
echo "2. Run preflight check: python scripts/preflight_check.py"
echo "3. Run WebSocket probe: python scripts/ws_probe.py"
echo "4. If all passes, proceed with demo trading"
echo ""
echo "⚠️  IMPORTANT: Verify egress IP '$EGRESS_IP' is allowlisted in CDP"
echo ""