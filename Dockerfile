FROM python:3.14-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install dependencies into /app/.venv. Project source isn't installed
# (we run from the source tree), so --no-install-project is fine.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project


FROM python:3.14-slim AS runtime

RUN useradd -u 1000 -m -s /bin/bash app

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY --chown=app:app app ./app
COPY --chown=app:app migrations ./migrations
COPY --chown=app:app scripts ./scripts
COPY --chown=app:app alembic.ini ./
COPY --chown=app:app config.example.toml ./config.toml
COPY --chown=app:app docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

USER app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --start-interval=2s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8080/healthz',timeout=3).status==200 else 1)" \
    || exit 1

ENTRYPOINT ["/entrypoint.sh"]
