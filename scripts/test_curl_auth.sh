#!/bin/bash
# Test CDP authentication with curl directly

echo "======================================================================"
echo "CURL CDP AUTHENTICATION TEST"
echo "======================================================================"

# Load environment
source .env

# Generate JWT using Python
JWT=$(poetry run python -c "
import os
from dotenv import load_dotenv
load_dotenv()

from bot_v2.features.brokerages.coinbase.cdp_auth_v2 import CDPAuthV2

api_key = os.getenv('COINBASE_PROD_CDP_API_KEY') or os.getenv('COINBASE_CDP_API_KEY')
private_key = os.getenv('COINBASE_PROD_CDP_PRIVATE_KEY') or os.getenv('COINBASE_CDP_PRIVATE_KEY')

auth = CDPAuthV2(
    api_key_name=api_key,
    private_key_pem=private_key,
    base_host='api.coinbase.com'
)

token = auth.generate_jwt('GET', '/api/v3/brokerage/accounts')
print(token)
" 2>/dev/null)

if [ -z "$JWT" ]; then
    echo "❌ Failed to generate JWT"
    exit 1
fi

echo "✅ Generated JWT token"
echo "   Token (first 50 chars): ${JWT:0:50}..."
echo ""
echo "Making API request with curl..."
echo ""

# Make the request
response=$(curl -s -w "\n---STATUS:%{http_code}---" \
    -H "Authorization: Bearer $JWT" \
    -H "Content-Type: application/json" \
    -H "CB-VERSION: 2024-10-24" \
    "https://api.coinbase.com/api/v3/brokerage/accounts" 2>&1)

# Extract status code
status=$(echo "$response" | grep -o "STATUS:[0-9]*" | cut -d: -f2)
body=$(echo "$response" | sed 's/---STATUS:[0-9]*---//')

echo "Response Status: $status"
echo "Response Body:"
echo "$body" | head -20

if [ "$status" = "200" ]; then
    echo ""
    echo "✅ Authentication successful!"
else
    echo ""
    echo "❌ Authentication failed with status $status"
    echo ""
    echo "Debugging information:"
    echo "1. Your IP: $(curl -s https://api.ipify.org)"
    echo "2. API Key: ${COINBASE_PROD_CDP_API_KEY:0:50}..."
fi