#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "$ROOT_DIR"

poetry run ruff check \
  src/bot_v2/features/brokerages/coinbase/account_manager.py \
  src/bot_v2/features/brokerages/coinbase/client.py

poetry run mypy \
  src/bot_v2/features/brokerages/coinbase/account_manager.py \
  src/bot_v2/features/brokerages/coinbase/client.py

poetry run pytest tests/unit/bot_v2/features/brokerages/coinbase -q
