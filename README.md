# Мудрый Лось — Yandex.Alisa skill

LLM-backed Yandex.Alisa dialog skill. Activates via `Алиса, спроси у мудрого лося <вопрос>` and replies in a wise-elk persona.

## Setup

```sh
uv sync
cp .env.example .env   # fill in real values
uv run alembic upgrade head
uv run uvicorn app.main:app --reload
```

## Run tests

```sh
uv run pytest
```

## Create a link code for testing

```sh
uv run python -m scripts.issue_code --name "Test User"
# prints a 6-digit code; speak it to the skill once
```

## Smoke-test a local webhook

```sh
uv run python -m scripts.replay app/tests/fixtures/first_turn.json
```

## Configuration

- Secrets and environment-specific URLs live in `.env` (see `.env.example`).
- Tunables (wait phrases, timeouts, pagination) live in `config.toml`.
