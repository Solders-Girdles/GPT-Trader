.PHONY: dev-up dev-down lint fmt fmt-check lint-fix typecheck docs-audit tui-css-check test-guardrails ci-required test smoke preflight preflight-readiness dash cov clean clean-dry-run scaffold-slice \
	readiness-window legacy-bundle agent-setup agent-check agent-impact agent-impact-full agent-map agent-tests agent-risk \
	agent-naming agent-health agent-health-fast agent-health-full agent-chaos-smoke agent-chaos-week \
	agent-regenerate agent-verify agent-docs-links canary-liveness canary-liveness-check canary-daily canary-decision-traces \
	canary-decision-trace-probe canary-runtime-info canary-stop canary-start \
	canary-restart canary-status canary-watchdog canary-watchdog-once ops-controls-smoke \
	test-triage test-triage-check test-unit test-property test-contract test-real-api test-integration test-integration-fast \
	backtest backtest-quick backtest-walk-forward backtest-walk-forward-quick guard-parity \
	legacy-patterns \
	test-snapshots

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
BACKTEST_PROFILE?=canary
BACKTEST_SYMBOL?=BTC-USD
BACKTEST_GRANULARITY?=FIVE_MINUTE
BACKTEST_DAYS?=30
BACKTEST_STRATEGY?=
BACKTEST_WF_WINDOWS?=6
BACKTEST_WF_WINDOW_DAYS?=90
BACKTEST_WF_STEP_DAYS?=30
BACKTEST_EXTRA_ARGS?=
GUARD_PARITY_PROFILE?=canary
GUARD_PARITY_SYMBOL?=BTC-USD
GUARD_PARITY_OUTPUT_DIR?=runtime_data/$(GUARD_PARITY_PROFILE)/reports
GUARD_PARITY_RUN_ID?=

# Start local development stack (bot by default)
dev-up:
	docker compose --project-directory $(COMPOSE_DIR) -f $(COMPOSE_FILE) --env-file $(COMPOSE_DIR)/.env up -d

dev-down:
	docker compose --project-directory $(COMPOSE_DIR) -f $(COMPOSE_FILE) --env-file $(COMPOSE_DIR)/.env down -v

lint:
	uv run ruff check .
	uv run black --check .

lint-fix:
	uv run ruff check . --fix

fmt:
	uv run black .

fmt-check:
	uv run black --check .

typecheck:
	uv run mypy src

docs-audit:
	uv run python scripts/maintenance/docs_link_audit.py
	uv run python scripts/maintenance/docs_reachability_check.py

tui-css-check:
	uv run python scripts/ci/check_tui_css_up_to_date.py

test-guardrails:
	uv run python scripts/ci/check_test_hygiene.py
	uv run python scripts/ci/check_legacy_patterns.py
	uv run python scripts/ci/check_legacy_test_triage.py
	uv run python scripts/ci/check_dedupe_manifest.py --strict
	$(MAKE) test-triage-check

ci-required:
	$(MAKE) lint
	uv run ruff check scripts/ops scripts/backtest_runner.py scripts/perps_dashboard.py scripts/monitoring/export_metrics.py scripts/monitoring/canary_reduce_only_test.py scripts/monitoring/manage_logs.py scripts/production_preflight.py scripts/readiness_window.py scripts/test_api_connectivity.py scripts/test_paper_broker.py
	@if grep -rn -E "(from|import)\\s+gpt_trader\\.orchestration" src tests scripts --include="*.py"; then \
		echo "::error::gpt_trader.orchestration was removed in v3.0"; \
		echo "Use canonical paths: app.*, features.live_trade.*, features.brokerages.*"; \
		echo "See docs/DEPRECATIONS.md for migration guidance."; \
		exit 1; \
	fi
	@echo "No orchestration imports found - package was removed in v3.0."
	uv run python scripts/ci/check_deprecation_registry.py
	$(MAKE) docs-audit
	$(MAKE) typecheck
	uv run agent-regenerate --verify
	$(MAKE) tui-css-check
	$(MAKE) test-guardrails
	GPT_TRADER_STRICT_CONTAINER=1 PYTHONWARNINGS=default uv run pytest tests/unit -n auto -q --ignore-glob=tests/unit/gpt_trader/tui/test_snapshots_*.py

legacy-patterns:
	uv run python scripts/ci/check_legacy_patterns.py

test:
	uv run pytest -q

test-triage:
	uv run python scripts/maintenance/test_legacy_triage.py

test-triage-check:
	uv run python scripts/maintenance/test_legacy_triage.py --check

test-unit:
	uv run pytest -q tests/unit

test-property:
	uv run pytest -q tests/property

test-contract:
	uv run pytest -q tests/contract

test-real-api:
	uv run pytest -q -o addopts= -m real_api tests/real_api

test-integration:
	uv run pytest -q -o addopts= -m "integration and not real_api" tests/integration

test-integration-fast:
	uv run pytest -q -o addopts= -m "integration and not slow and not real_api" tests/integration

test-snapshots:
	uv run pytest -q -n 0 tests/unit/gpt_trader/tui/test_snapshots_*.py

backtest:
	uv run python scripts/backtest_runner.py \
		--profile $(BACKTEST_PROFILE) \
		--symbol $(BACKTEST_SYMBOL) \
		--granularity $(BACKTEST_GRANULARITY) \
		--days $(BACKTEST_DAYS) \
		$(if $(BACKTEST_STRATEGY),--strategy-type $(BACKTEST_STRATEGY),) \
		$(BACKTEST_EXTRA_ARGS)

backtest-quick:
	uv run python scripts/backtest_runner.py \
		--profile $(BACKTEST_PROFILE) \
		--symbol $(BACKTEST_SYMBOL) \
		--granularity $(BACKTEST_GRANULARITY) \
		--days $(BACKTEST_DAYS) \
		--quick \
		$(if $(BACKTEST_STRATEGY),--strategy-type $(BACKTEST_STRATEGY),) \
		$(BACKTEST_EXTRA_ARGS)

backtest-walk-forward:
	uv run python scripts/backtest_runner.py \
		--profile $(BACKTEST_PROFILE) \
		--symbol $(BACKTEST_SYMBOL) \
		--granularity $(BACKTEST_GRANULARITY) \
		--walk-forward \
		--wf-windows $(BACKTEST_WF_WINDOWS) \
		--wf-window-days $(BACKTEST_WF_WINDOW_DAYS) \
		--wf-step-days $(BACKTEST_WF_STEP_DAYS) \
		$(if $(BACKTEST_STRATEGY),--strategy-type $(BACKTEST_STRATEGY),) \
		$(BACKTEST_EXTRA_ARGS)

backtest-walk-forward-quick:
	uv run python scripts/backtest_runner.py \
		--profile $(BACKTEST_PROFILE) \
		--symbol $(BACKTEST_SYMBOL) \
		--granularity $(BACKTEST_GRANULARITY) \
		--walk-forward \
		--wf-windows $(BACKTEST_WF_WINDOWS) \
		--wf-window-days $(BACKTEST_WF_WINDOW_DAYS) \
		--wf-step-days $(BACKTEST_WF_STEP_DAYS) \
		--quick \
		$(if $(BACKTEST_STRATEGY),--strategy-type $(BACKTEST_STRATEGY),) \
		$(BACKTEST_EXTRA_ARGS)

guard-parity:
	uv run python scripts/analysis/guard_parity_regression.py \
		--profile $(GUARD_PARITY_PROFILE) \
		--symbol $(GUARD_PARITY_SYMBOL) \
		--output-dir $(GUARD_PARITY_OUTPUT_DIR) \
		$(if $(GUARD_PARITY_RUN_ID),--run-id $(GUARD_PARITY_RUN_ID),)

smoke:
	uv run python scripts/production_preflight.py --profile dev --verbose

preflight:
	BROKER="$${BROKER:-coinbase}" COINBASE_SANDBOX="$${COINBASE_SANDBOX:-0}" COINBASE_API_MODE="$${COINBASE_API_MODE:-advanced}" \
	RISK_MAX_LEVERAGE="$${RISK_MAX_LEVERAGE:-1}" RISK_DAILY_LOSS_LIMIT="$${RISK_DAILY_LOSS_LIMIT:-50}" RISK_MAX_POSITION_PCT_PER_SYMBOL="$${RISK_MAX_POSITION_PCT_PER_SYMBOL:-0.10}" \
	uv run python scripts/production_preflight.py --profile canary --verbose

preflight-readiness:
	BROKER="$${BROKER:-coinbase}" COINBASE_SANDBOX="$${COINBASE_SANDBOX:-0}" COINBASE_API_MODE="$${COINBASE_API_MODE:-advanced}" \
	RISK_MAX_LEVERAGE="$${RISK_MAX_LEVERAGE:-1}" RISK_DAILY_LOSS_LIMIT="$${RISK_DAILY_LOSS_LIMIT:-50}" RISK_MAX_POSITION_PCT_PER_SYMBOL="$${RISK_MAX_POSITION_PCT_PER_SYMBOL:-0.10}" \
	GPT_TRADER_READINESS_REPORT="$(READINESS_REPORT_DIR)" uv run python scripts/production_preflight.py \
		--profile $(PREFLIGHT_PROFILE) --verbose

canary-liveness:
	@-uv run python scripts/ops/liveness_check.py --profile canary --event-type heartbeat --event-type price_tick --max-age-seconds 300
	@sqlite3 runtime_data/canary/events.db "select event_type, max(timestamp) as last_ts from events group by event_type order by last_ts desc limit 5;"

canary-liveness-check:
	uv run python scripts/ops/liveness_check.py --profile canary --event-type heartbeat --event-type price_tick --max-age-seconds 300

canary-daily:
	@$(MAKE) canary-liveness-check
	uv run gpt-trader report daily --profile canary --report-format both
	DRY_RUN=1 $(MAKE) preflight-readiness PREFLIGHT_PROFILE=canary
	$(MAKE) readiness-window PREFLIGHT_PROFILE=canary READINESS_WINDOW_HOURS=24
	@echo "daily_report_path=runtime_data/canary/reports/daily_report_$$(date -u +%F).json"
	@latest_preflight=$$(ls -t preflight_report_*.json 2>/dev/null | head -1 || true); \
	if [ -n "$$latest_preflight" ]; then echo "preflight_report_path=$$latest_preflight"; fi
	@echo "Next: append/update docs/READINESS.md 3-day streak log entry."

canary-decision-traces:
	uv run python scripts/ops/tail_decision_traces.py --profile canary --limit 10

canary-decision-trace-probe:
	uv run python scripts/ops/decision_trace_probe.py --profile canary --symbol BTC-USD --side buy

canary-runtime-info:
	uv run python scripts/ops/runtime_fingerprint.py --profile canary

canary-status:
	uv run python scripts/ops/canary_process.py --profile canary status

canary-stop:
	uv run python scripts/ops/canary_process.py --profile canary stop

canary-start:
	uv run python scripts/ops/canary_process.py --profile canary start

canary-restart:
	uv run python scripts/ops/canary_process.py --profile canary restart

canary-watchdog:
	uv run python scripts/ops/canary_watchdog.py --profile canary --auto-restart

canary-watchdog-once:
	uv run python scripts/ops/canary_watchdog.py --profile canary --once

ops-controls-smoke:
	uv run python scripts/ops/controls_smoke.py

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
	@echo "Legacy bundling helper retired; use git history if you need the old recovery steps."

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
	BROKER=coinbase COINBASE_SANDBOX=1 COINBASE_API_MODE=advanced COINBASE_ENABLE_INTX_PERPS=0 \
	RISK_MAX_LEVERAGE=3 RISK_DAILY_LOSS_LIMIT=100 RISK_MAX_POSITION_PCT_PER_SYMBOL=0.10 \
	uv run agent-health --quality-checks $(AGENT_HEALTH_FAST_QUALITY_CHECKS) \
	--format json --output var/agents/health/health_report.json

agent-health-full:
	BROKER=coinbase COINBASE_SANDBOX=1 COINBASE_API_MODE=advanced COINBASE_ENABLE_INTX_PERPS=0 \
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

agent-verify:
	uv run agent-regenerate --verify

agent-docs-links:
	uv run python scripts/maintenance/docs_link_audit.py
	uv run python scripts/maintenance/docs_reachability_check.py
