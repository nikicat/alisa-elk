"""POST a Yandex Dialogs JSON payload to the local /webhook.

Usage:
    uv run python -m scripts.replay path/to/payload.json
    uv run python -m scripts.replay path/to/payload.json --host http://localhost:8080
"""

import argparse
import json
import sys
import time

import httpx

from app.config import get_settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("payload", help="Path to a JSON file with the request body")
    parser.add_argument("--host", default="http://localhost:8080")
    parser.add_argument(
        "--secret",
        default=None,
        help="Webhook secret (defaults to WEBHOOK_PATH_SECRET env)",
    )
    args = parser.parse_args(argv)

    secret = args.secret or get_settings().WEBHOOK_PATH_SECRET
    url = f"{args.host}/webhook/{secret}"
    with open(args.payload, encoding="utf-8") as f:
        body = json.load(f)
    t0 = time.perf_counter()
    resp = httpx.post(url, json=body, timeout=10.0)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    print(f"status={resp.status_code} elapsed={elapsed_ms}ms", file=sys.stderr)
    print(json.dumps(resp.json(), ensure_ascii=False, indent=2))
    return 0 if resp.status_code == 200 else 1


if __name__ == "__main__":
    raise SystemExit(main())
