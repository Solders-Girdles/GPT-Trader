# Multi-stage Dockerfile for bot_v2 Trading System
# Uses Poetry for dependency management across staged build targets.

# Stage 1: Base runtime image shared by all stages
FROM python:3.12-slim AS base
ENV PYTHONUNBUFFERED=1     PYTHONDONTWRITEBYTECODE=1

# Common tooling required across stages (curl used by healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends     curl     && rm -rf /var/lib/apt/lists/*

# Create non-root user available to descendant stages
RUN groupadd -r trader && useradd -r -g trader trader
WORKDIR /app

# Stage 2: Dependency builder (installs Poetry-managed deps)
FROM base AS dependencies
ENV POETRY_VERSION=2.1.4     POETRY_HOME=/opt/poetry     PATH="/opt/poetry/bin:${PATH}"

# Build toolchain for native-backed Python dependencies
RUN apt-get update && apt-get install -y --no-install-recommends     build-essential     gcc     g++     make     wget     libssl-dev     libffi-dev     libpq-dev     && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry==${POETRY_VERSION}

# Install only runtime dependencies (no dev extras) into the global site-packages
COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false     && poetry install --no-interaction --no-ansi --without dev --no-root

# Stage 3: Development environment (includes dev dependencies and full source tree)
FROM dependencies AS development
COPY --chown=trader:trader . .
RUN poetry install --no-interaction --no-ansi --with dev --no-root
ENV PYTHONPATH=/app     ENV=development
USER trader
CMD ["python", "-m", "bot_v2", "--profile", "dev", "--dry-run"]

# Stage 4: Testing environment (runs unit tests with coverage)
FROM dependencies AS testing
COPY --chown=trader:trader . .
RUN poetry install --no-interaction --no-ansi --with dev --no-root
RUN poetry run pytest tests/unit/ \
    --deselect tests/unit/bot_v2/state/backup/services/test_transport.py::TestPathInitialization::test_network_path_creation_failure_handled \
    --deselect tests/unit/bot_v2/state/backup/services/test_transport.py::TestDeleteOperations::test_delete_returns_false_on_error \
    --cov=src/bot_v2 --cov-report=term-missing

# Stage 5: Production build (minimal runtime footprint)
FROM base AS production

# Runtime libraries only copied from builder stage
RUN apt-get update && apt-get install -y --no-install-recommends     libpq5     libgomp1     && rm -rf /var/lib/apt/lists/*

COPY --from=dependencies /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

# Application code and configuration
COPY --chown=trader:trader src/bot_v2/ ./src/bot_v2/
COPY --chown=trader:trader config/ ./config/

# Prepare runtime directories
RUN mkdir -p /app/logs /app/data /app/cache     && chown -R trader:trader /app

ENV PYTHONPATH=/app     PYTHONUNBUFFERED=1     ENV=production     LOG_LEVEL=INFO

USER trader

# Expose service ports
EXPOSE 8080 8443 9090

# Health check uses the metrics server /health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3     CMD curl -f http://localhost:9090/health || exit 1

ENTRYPOINT ["python", "-m", "bot_v2"]
CMD ["--mode", "production", "--config", "/app/config/production.yaml"]

# Stage 6: Security scanner (optional tooling layer)
FROM production AS security
USER root
RUN pip install --no-cache-dir safety bandit
RUN safety check --json     && bandit -r /app/src -f json -o /tmp/bandit-report.json
USER trader

# Labels to preserve provenance metadata
LABEL maintainer="Bot V2 Trading Team"
LABEL version="1.0.0"
LABEL description="Production-ready trading bot with ML intelligence"
LABEL org.opencontainers.image.source="https://github.com/yourusername/gpt-trader"
LABEL org.opencontainers.image.licenses="MIT"
