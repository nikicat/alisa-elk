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

## Deployment (Docker + Traefik)

The container exposes plain HTTP on port 8080 inside Docker, and Traefik on
the host routes `https://<DOMAIN><ROUTE_PREFIX>/*` to it, stripping the
prefix before forwarding. All routing is declared as labels on the
container — no Traefik config files to touch.

### First-time setup

```sh
git clone https://github.com/nikicat/alisa-elk.git
cd alisa-elk

# Bind-mount target for SQLite. uid 1000 matches the container user.
mkdir -p data
sudo chown -R 1000:1000 data

# Fill in .env with real values.
cp .env.example .env
$EDITOR .env
```

Required `.env` values for deployment:

| Key | Example | Purpose |
|---|---|---|
| `YANDEX_SKILL_ID` | `d668...` | Reject webhook calls from other skills. |
| `WEBHOOK_PATH_SECRET` | random 32-hex | Secret path segment in the webhook URL. |
| `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` | router-specific | OpenAI-compatible LLM endpoint. |
| `DOMAIN` | `yourdomain.example` | Host part of the public URL. |
| `ROUTE_PREFIX` | `/elk` | Subpath under which the skill lives. |
| `ROOT_PATH` | `/elk` | Tell FastAPI its mount point (cosmetic). |
| `TRAEFIK_NETWORK` | `traefik` | The Docker network Traefik is attached to. |
| `TRAEFIK_ENTRYPOINT` | `websecure` | Traefik entrypoint name for HTTPS. |
| `TRAEFIK_CERTRESOLVER` | `letsencrypt` | Cert resolver name configured in Traefik. |

```sh
docker compose up -d --build
docker compose logs -f
```

Once running, the webhook URL to paste into Yandex Dialogs is:
```
https://<DOMAIN><ROUTE_PREFIX>/webhook/<WEBHOOK_PATH_SECRET>
```

### Issuing link codes in production

```sh
docker compose exec elk python -m scripts.issue_code --name "Имя"
```

### Updating

```sh
git pull
docker compose up -d --build
```

Compose's `restart: unless-stopped` keeps the container alive across reboots
and crashes. SQLite data persists in `./data/elk.db` on the host — back it
up with `cp data/elk.db data/elk.db.bak` (SQLite is fine being copied while
the app reads from it; for hot copies of a busy DB, use `sqlite3 ... .backup`).
