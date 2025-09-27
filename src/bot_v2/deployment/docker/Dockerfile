# Multi-stage Dockerfile for bot_v2 Trading System
# Optimized for security, size, and build speed

# Stage 1: Base runtime environment
FROM python:3.11-slim as base

# Security: Create non-root user
RUN groupadd -r trader && useradd -r -g trader trader

# Set working directory
WORKDIR /app

# Install system dependencies and security updates
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Stage 2: Dependencies builder
FROM base as dependencies

# Copy requirements file
COPY requirements.txt .
COPY requirements-prod.txt* .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt \
    && if [ -f requirements-prod.txt ]; then pip install --no-cache-dir -r requirements-prod.txt; fi

# Stage 3: Development environment
FROM dependencies as development

# Install development tools
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    black \
    flake8 \
    mypy \
    ipdb

# Copy source code
COPY --chown=trader:trader . .

# Set environment
ENV PYTHONPATH=/app
ENV ENV=development

# Switch to non-root user
USER trader

# Development command
CMD ["python", "-m", "bot_v2", "--mode", "development"]

# Stage 4: Testing environment
FROM dependencies as testing

# Copy source and tests
COPY --chown=trader:trader src/ ./src/
COPY --chown=trader:trader tests/ ./tests/
COPY --chown=trader:trader pytest.ini .

# Install test dependencies
RUN pip install --no-cache-dir pytest pytest-cov pytest-asyncio

# Run tests
RUN python -m pytest tests/unit/ --cov=src/bot_v2 --cov-report=term-missing

# Stage 5: Production build
FROM python:3.11-slim as production

# Create non-root user
RUN groupadd -r trader && useradd -r -g trader trader

# Set working directory
WORKDIR /app

# Copy only necessary files from dependencies stage
COPY --from=dependencies /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=trader:trader src/bot_v2/ ./src/bot_v2/
COPY --chown=trader:trader config/ ./config/

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/cache \
    && chown -R trader:trader /app

# Security hardening
RUN chmod -R 755 /app/src \
    && chmod -R 777 /app/logs /app/data /app/cache

# Environment configuration
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV ENV=production
ENV LOG_LEVEL=INFO

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Switch to non-root user
USER trader

# Expose ports
EXPOSE 8080 8443 9090

# Production entrypoint
ENTRYPOINT ["python", "-m", "bot_v2"]

# Default command
CMD ["--mode", "production", "--config", "/app/config/production.yaml"]

# Stage 6: Security scanner (optional)
FROM production as security

# Install security scanning tools
USER root
RUN pip install --no-cache-dir safety bandit

# Run security checks
RUN safety check --json \
    && bandit -r /app/src -f json -o /tmp/bandit-report.json

# Switch back to non-root user
USER trader

# Labels for metadata
LABEL maintainer="Bot V2 Trading Team"
LABEL version="1.0.0"
LABEL description="Production-ready trading bot with ML intelligence"
LABEL org.opencontainers.image.source="https://github.com/yourusername/gpt-trader"
LABEL org.opencontainers.image.licenses="MIT"
