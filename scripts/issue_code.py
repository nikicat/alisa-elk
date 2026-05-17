"""Admin CLI: create a user and a 6-digit link code, print the code.

Usage:
    uv run python -m scripts.issue_code --name "Alice"
    uv run python -m scripts.issue_code --user-id 7    # existing user
"""

import argparse
import sys

from app import repo
from app.db import init_engine, session_scope


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="Display name for a new user")
    parser.add_argument(
        "--user-id", type=int, help="Existing user id (skip user creation)"
    )
    parser.add_argument("--ttl-minutes", type=int, default=15, help="Code lifetime")
    args = parser.parse_args(argv)

    if args.user_id is None and args.name is None:
        parser.error("provide --name (new user) or --user-id (existing)")

    init_engine()
    with session_scope() as db:
        if args.user_id is None:
            user = repo.create_user(db, display_name=args.name)
            db.commit()
            user_id = user.id
            print(f"created user id={user_id} name={args.name!r}", file=sys.stderr)
        else:
            user_id = args.user_id
        code = repo.create_link_code(db, user_id, ttl_minutes=args.ttl_minutes)
        db.commit()
    print(code)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
