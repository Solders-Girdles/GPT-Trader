.PHONY: dev-up dev-down lint typecheck test smoke preflight dash cov clean clean-dry-run legacy-bundle

COMPOSE_DIR=deploy/gpt_trader/docker
COMPOSE_FILE=$(COMPOSE_DIR)/docker-compose.yaml

# Start local development stack (bot by default)
dev-up:
	docker compose --project-directory $(COMPOSE_DIR) -f $(COMPOSE_FILE) --env-file $(COMPOSE_DIR)/.env up -d

dev-down:
	docker compose --project-directory $(COMPOSE_DIR) -f $(COMPOSE_FILE) --env-file $(COMPOSE_DIR)/.env down -v

lint:
	poetry run ruff check .
	poetry run black --check .

typecheck:
	poetry run mypy src

test:
	poetry run pytest -q

smoke:
	poetry run python scripts/production_preflight.py --profile dev --verbose

preflight:
	poetry run python scripts/production_preflight.py --profile canary --verbose

cov:
	poetry run pytest -m "not slow and not performance" -q \
		--cov=src/gpt_trader/cli \
		--cov=src/gpt_trader/config \
		--cov=src/gpt_trader/features/brokerages/coinbase/client \
		--cov=src/gpt_trader/features/brokerages/coinbase/utilities \
		--cov-report=html:var/results/coverage/html \
		--cov-report=term --cov-fail-under=80

dash:
	poetry run python scripts/monitoring/export_metrics.py \
		--metrics-file var/data/coinbase_trader/prod/metrics.json \
		--port 9102

clean:
	poetry run python scripts/maintenance/cleanup_workspace.py --apply

clean-dry-run:
	poetry run python scripts/maintenance/cleanup_workspace.py

legacy-bundle:
	@echo "Legacy bundling helper retired; see docs/archive/legacy_recovery.md for manual recovery steps."
