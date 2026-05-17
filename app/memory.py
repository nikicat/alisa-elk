from datetime import datetime

from sqlalchemy.orm import Session

from app.repo import recent_turns


def load_recent_turns(
    db: Session,
    session_id: str,
    limit: int,
    *,
    since: datetime | None = None,
) -> list[dict[str, str]]:
    """Return last `limit` exchanges as OpenAI-format messages (user/assistant pairs)."""
    turns = recent_turns(db, session_id, limit, since=since)
    messages: list[dict[str, str]] = []
    for turn in turns:
        messages.append({"role": "user", "content": turn.request_text})
        messages.append({"role": "assistant", "content": turn.response_text})
    return messages
