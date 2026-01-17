.PHONY: dev-up dev-down lint typecheck test smoke preflight preflight-readiness dash cov clean clean-dry-run scaffold-slice \
	readiness-window legacy-bundle agent-setup agent-check agent-impact agent-impact-full agent-map agent-tests agent-risk \
	agent-naming agent-health agent-health-fast agent-health-full agent-chaos-smoke agent-chaos-week \
	agent-regenerate agent-docs-links

COMPOSE_DIR=deploy/gpt_trader/docker
COMPOSE_FILE=$(COMPOSE_DIR)/docker-compose.yaml
AGENT_HEALTH_FAST_QUALITY_CHECKS?=lint,format,types
AGENT_CHAOS_DAYS?=2
AGENT_CHAOS_SCENARIO?=volatile_market
AGENT_CHAOS_OUTPUT?=var/agents/health/chaos_smoke.json
AGENT_CHAOS_MAX_DRAWDOWN_PCT?=10
AGENT_CHAOS_MAX_FEES_PCT?=4.5
PREFLIGHT_PROFILE?=canary
READINESS_REPORT_DIR?=runtime_data/$(PREFLIGHT_PROFILE)/reports
READINESS_WINDOW_HOURS?=24

# Start local development stack (bot by default)
dev-up:
	docker compose --project-directory $(COMPOSE_DIR) -f $(COMPOSE_FILE) --env-file $(COMPOSE_DIR)/.env up -d

dev-down:
	docker compose --project-directory $(COMPOSE_DIR) -f $(COMPOSE_FILE) --env-file $(COMPOSE_DIR)/.env down -v

lint:
	uv run ruff check .
	uv run black --check .

typecheck:
	uv run mypy src

test:
	uv run pytest -q

smoke:
	uv run python scripts/production_preflight.py --profile dev --verbose

preflight:
	BROKER="$${BROKER:-coinbase}" COINBASE_SANDBOX="$${COINBASE_SANDBOX:-0}" COINBASE_API_MODE="$${COINBASE_API_MODE:-advanced}" \
	uv run python scripts/production_preflight.py --profile canary --verbose

preflight-readiness:
	BROKER="$${BROKER:-coinbase}" COINBASE_SANDBOX="$${COINBASE_SANDBOX:-0}" COINBASE_API_MODE="$${COINBASE_API_MODE:-advanced}" \
	GPT_TRADER_READINESS_REPORT="$(READINESS_REPORT_DIR)" uv run python scripts/production_preflight.py \
		--profile $(PREFLIGHT_PROFILE) --verbose

canary-liveness:
	@age=$$(sqlite3 runtime_data/canary/events.db "select coalesce(cast(round((julianday('now') - julianday(max(timestamp)))*86400) as integer), 999999) as last_event_age_seconds from events;"); \
	echo "last_event_age_seconds=$$age"; \
	if [ "$$age" -gt 300 ]; then echo "liveness_status=RED"; else echo "liveness_status=GREEN"; fi
	@sqlite3 runtime_data/canary/events.db "select event_type, max(timestamp) as last_ts from events group by event_type order by last_ts desc limit 5;"

canary-liveness-check:
	@age=$$(sqlite3 runtime_data/canary/events.db "select coalesce(cast(round((julianday('now') - julianday(max(timestamp)))*86400) as integer), 999999) as last_event_age_seconds from events;"); \
	echo "last_event_age_seconds=$$age"; \
	if [ "$$age" -gt 300 ]; then echo "liveness_status=RED"; exit 1; else echo "liveness_status=GREEN"; fi

canary-daily:
	@$(MAKE) canary-liveness-check
	uv run gpt-trader report daily --profile canary --report-format both
	DRY_RUN=1 $(MAKE) preflight-readiness PREFLIGHT_PROFILE=canary
	$(MAKE) readiness-window PREFLIGHT_PROFILE=canary READINESS_WINDOW_HOURS=24
	@echo "daily_report_path=runtime_data/canary/reports/daily_report_$$(date -u +%F).json"
	@latest_preflight=$$(ls -t preflight_report_*.json 2>/dev/null | head -1 || true); \
	if [ -n "$$latest_preflight" ]; then echo "preflight_report_path=$$latest_preflight"; fi
	@echo "Next: append/update docs/READINESS.md 3-day streak log entry."

readiness-window:
	uv run python scripts/readiness_window.py --profile $(PREFLIGHT_PROFILE) --hours $(READINESS_WINDOW_HOURS)

cov:
	uv run pytest -m "not slow and not performance" -q \
		--cov=src/gpt_trader/cli \
		--cov=src/gpt_trader/config \
		--cov=src/gpt_trader/features/brokerages/coinbase/client \
		--cov=src/gpt_trader/features/brokerages/coinbase/utilities \
		--cov-report=html:var/results/coverage/html \
		--cov-report=term --cov-fail-under=80

dash:
	uv run python scripts/monitoring/export_metrics.py \
		--profile canary \
		--runtime-root . \
		--port 9102

clean:
	uv run python scripts/maintenance/cleanup_workspace.py --apply

clean-dry-run:
	uv run python scripts/maintenance/cleanup_workspace.py

scaffold-slice:
	uv run python scripts/maintenance/feature_slice_scaffold.py --name $(name) $(flags)

legacy-bundle:
	@echo "Legacy bundling helper retired; see docs/archive/legacy_recovery.md for manual recovery steps."

agent-setup:
	uv sync --all-extras

agent-check:
	uv run agent-check --format text

agent-impact:
	uv run agent-impact --from-git --include-importers --source-files --exclude-integration --format text

agent-impact-full:
	uv run agent-impact --from-git --include-importers --format text

agent-map:
	uv run agent-map --format text

agent-tests:
	uv run agent-tests --stdout

agent-risk:
	uv run agent-risk --with-docs

agent-naming:
	uv run agent-naming

agent-health:
	$(MAKE) agent-health-full

agent-health-fast:
	BROKER=coinbase COINBASE_SANDBOX=1 COINBASE_API_MODE=advanced COINBASE_ENABLE_DERIVATIVES=0 \
	RISK_MAX_LEVERAGE=3 RISK_DAILY_LOSS_LIMIT=100 RISK_MAX_POSITION_PCT_PER_SYMBOL=0.10 \
	uv run agent-health --quality-checks $(AGENT_HEALTH_FAST_QUALITY_CHECKS) \
	--format json --output var/agents/health/health_report.json

agent-health-full:
	BROKER=coinbase COINBASE_SANDBOX=1 COINBASE_API_MODE=advanced COINBASE_ENABLE_DERIVATIVES=0 \
	RISK_MAX_LEVERAGE=3 RISK_DAILY_LOSS_LIMIT=100 RISK_MAX_POSITION_PCT_PER_SYMBOL=0.10 \
	uv run agent-health --format json --output var/agents/health/health_report.json \
	--text-output var/agents/health/health_report.txt --pytest-args -q tests/unit

agent-chaos-smoke:
	uv run python scripts/analysis/paper_trade_stress_test.py \
		--days $(AGENT_CHAOS_DAYS) \
		--chaos \
		--chaos-scenario $(AGENT_CHAOS_SCENARIO) \
		--max-drawdown-pct $(AGENT_CHAOS_MAX_DRAWDOWN_PCT) \
		--max-fees-pct $(AGENT_CHAOS_MAX_FEES_PCT) \
		--export $(AGENT_CHAOS_OUTPUT)

agent-chaos-week:
	AGENT_CHAOS_DAYS=7 AGENT_CHAOS_OUTPUT=var/agents/health/chaos_week.json \
	$(MAKE) agent-chaos-smoke

agent-regenerate:
	uv run agent-regenerate

agent-docs-links:
	uv run python scripts/maintenance/docs_link_audit.py
