.PHONY: dev-up dev-down lint typecheck test smoke preflight dash cov clean clean-dry-run scaffold-slice \
	legacy-bundle agent-check agent-impact agent-map agent-tests agent-risk agent-naming agent-health agent-regenerate \
	agent-docs-links

COMPOSE_DIR=deploy/gpt_trader/docker
COMPOSE_FILE=$(COMPOSE_DIR)/docker-compose.yaml

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
	uv run python scripts/production_preflight.py --profile canary --verbose

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
		--metrics-file var/data/coinbase_trader/prod/metrics.json \
		--port 9102

clean:
	uv run python scripts/maintenance/cleanup_workspace.py --apply

clean-dry-run:
	uv run python scripts/maintenance/cleanup_workspace.py

scaffold-slice:
	uv run python scripts/maintenance/feature_slice_scaffold.py --name $(name) $(flags)

legacy-bundle:
	@echo "Legacy bundling helper retired; see docs/archive/legacy_recovery.md for manual recovery steps."

agent-check:
	uv run agent-check --format text

agent-impact:
	uv run agent-impact --from-git --format text

agent-map:
	uv run agent-map --format text

agent-tests:
	uv run agent-tests --stdout

agent-risk:
	uv run agent-risk --with-docs

agent-naming:
	uv run agent-naming

agent-health:
	uv run agent-health

agent-regenerate:
	uv run agent-regenerate

agent-docs-links:
	uv run python scripts/maintenance/docs_link_audit.py
