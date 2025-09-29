#!/usr/bin/env bash
set -euo pipefail

PROFILE=${BOT_PROFILE:-canary}

ARGS=("--profile" "$PROFILE")

if [[ "${BOT_DRY_RUN:-1}" != "0" ]]; then
  ARGS+=("--dry-run")
fi

if [[ "${BOT_DEV_FAST:-0}" == "1" ]]; then
  ARGS+=("--dev-fast")
fi

if [[ -n "${BOT_SYMBOLS:-}" ]]; then
  # Allow comma-separated symbols; strip spaces
  IFS=',' read -r -a SYMBOL_LIST <<< "${BOT_SYMBOLS// /}"
  if [[ ${#SYMBOL_LIST[@]} -gt 0 ]]; then
    ARGS+=("--symbols")
    for sym in "${SYMBOL_LIST[@]}"; do
      if [[ -n "$sym" ]]; then
        ARGS+=("$sym")
      fi
    done
  fi
fi

if [[ $# -gt 0 ]]; then
  ARGS+=("$@")
fi

exec python -m bot_v2.cli "${ARGS[@]}"
