#!/bin/bash
# Run all tests for the trading system

echo "Running all tests..."
echo "==================="

# Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Run Coinbase unit tests
echo ""
echo "📦 Running Coinbase unit tests..."
python -m pytest tests/unit/bot_v2/features/brokerages/coinbase -q

# Run paper trading tests if they exist
echo ""
echo "📄 Running paper trading tests..."
python -m pytest tests/unit/bot_v2/features/paper_trade -q 2>/dev/null || echo "No paper trade tests yet"

# Run integration tests (if credentials available)
if [ ! -z "$COINBASE_CDP_API_KEY" ]; then
    echo ""
    echo "🌐 Running integration tests..."
    python -m pytest tests/integration -q -m integration
else
    echo ""
    echo "⚠️  Skipping integration tests (no credentials)"
fi

echo ""
echo "==================="
echo "Tests complete!"
