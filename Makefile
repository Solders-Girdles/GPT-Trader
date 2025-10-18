.PHONY: dev-up dev-down lint typecheck test smoke preflight dash cov clean clean-dry-run legacy-bundle

COMPOSE_DIR=deploy/bot_v2/docker
COMPOSE_FILE=$(COMPOSE_DIR)/docker-compose.yaml

# Start local development stack (databases, broker, bot)
dev-up:
	docker compose --project-directory $(COMPOSE_DIR) -f $(COMPOSE_FILE) --env-file $(COMPOSE_DIR)/.env --profile local-dev up -d

dev-down:
	docker compose --project-directory $(COMPOSE_DIR) -f $(COMPOSE_FILE) --env-file $(COMPOSE_DIR)/.env --profile local-dev down -v

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
		--cov=src/bot_v2/cli \
		--cov=src/bot_v2/config \
		--cov=src/bot_v2/features/brokerages/coinbase/client \
		--cov=src/bot_v2/features/brokerages/coinbase/utilities \
		--cov-report=term --cov-fail-under=80

dash:
	poetry run python scripts/monitoring/export_metrics.py \
		--metrics-file var/data/perps_bot/prod/metrics.json \
		--port 9102

clean:
	poetry run python scripts/maintenance/cleanup_workspace.py --apply

clean-dry-run:
	poetry run python scripts/maintenance/cleanup_workspace.py

legacy-bundle:
	poetry run python scripts/maintenance/create_legacy_bundle.py
