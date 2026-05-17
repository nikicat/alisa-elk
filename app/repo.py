import json
import secrets
from datetime import UTC, datetime, timedelta

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.models import LinkCode, LinkedAccount, PendingRequest, TurnLog, User


def utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


# ---------- users / linking ----------


def create_user(db: Session, display_name: str | None = None) -> User:
    user = User(display_name=display_name, created_at=utcnow())
    db.add(user)
    db.flush()
    return user


def get_user_by_app_id(db: Session, application_id: str) -> User | None:
    row = db.execute(
        select(User)
        .join(LinkedAccount, LinkedAccount.user_id == User.id)
        .where(LinkedAccount.yandex_application_id == application_id)
    ).scalar_one_or_none()
    return row


def create_link_code(db: Session, user_id: int, ttl_minutes: int = 15) -> str:
    code = f"{secrets.randbelow(1_000_000):06d}"
    now = utcnow()
    db.add(
        LinkCode(
            code=code,
            user_id=user_id,
            created_at=now,
            expires_at=now + timedelta(minutes=ttl_minutes),
        )
    )
    db.flush()
    return code


def consume_link_code(
    db: Session, code: str, application_id: str
) -> User | None:
    """Atomically claim the code and bind application_id to the user.

    Returns the User on success; None if the code is unknown/expired/used, or
    if the device is already linked to a *different* user (refusing silent
    rebind protects shared family devices from being hijacked by anyone with
    a fresh code).
    """
    now = utcnow()
    result = db.execute(
        update(LinkCode)
        .where(
            LinkCode.code == code,
            LinkCode.used_at.is_(None),
            LinkCode.expires_at > now,
        )
        .values(used_at=now, used_by_app_id=application_id)
        .returning(LinkCode.user_id)
    )
    row = result.fetchone()
    if row is None:
        return None
    user_id: int = row[0]
    existing = db.execute(
        select(LinkedAccount).where(
            LinkedAccount.yandex_application_id == application_id
        )
    ).scalar_one_or_none()
    if existing is None:
        db.add(
            LinkedAccount(
                yandex_application_id=application_id,
                user_id=user_id,
                linked_at=now,
            )
        )
    elif existing.user_id == user_id:
        # Same user re-linking the same device — just refresh the timestamp.
        existing.linked_at = now
    else:
        # Device already bound to a different user. Code is consumed (so it
        # can't be replayed) but we refuse to rebind.
        return None
    db.flush()
    return db.get(User, user_id)


# ---------- turn log + memory ----------


def log_turn(
    db: Session,
    *,
    user_id: int | None,
    yandex_application_id: str,
    session_id: str,
    message_id: int,
    request_text: str,
    response_text: str,
    total_ms: int,
    llm_ms: int | None = None,
    llm_input_tokens: int | None = None,
    llm_output_tokens: int | None = None,
    pending_request_id: str | None = None,
) -> TurnLog:
    turn = TurnLog(
        ts=utcnow(),
        user_id=user_id,
        yandex_application_id=yandex_application_id,
        session_id=session_id,
        message_id=message_id,
        request_text=request_text,
        response_text=response_text,
        total_ms=total_ms,
        llm_ms=llm_ms,
        llm_input_tokens=llm_input_tokens,
        llm_output_tokens=llm_output_tokens,
        pending_request_id=pending_request_id,
    )
    db.add(turn)
    db.flush()
    return turn


def get_turn(db: Session, turn_id: int) -> TurnLog | None:
    return db.get(TurnLog, turn_id)


def recent_turns(
    db: Session,
    session_id: str,
    limit: int,
    *,
    since: datetime | None = None,
) -> list[TurnLog]:
    if limit <= 0:
        return []
    stmt = select(TurnLog).where(TurnLog.session_id == session_id)
    if since is not None:
        stmt = stmt.where(TurnLog.ts > since)
    stmt = stmt.order_by(TurnLog.ts.desc()).limit(limit)
    rows = db.execute(stmt).scalars().all()
    return list(reversed(rows))


# ---------- pending requests ----------


def create_pending(
    db: Session,
    *,
    pending_id: str,
    user_id: int,
    session_id: str,
    request_text: str,
    messages: list[dict],
) -> PendingRequest:
    now = utcnow()
    row = PendingRequest(
        id=pending_id,
        user_id=user_id,
        session_id=session_id,
        request_text=request_text,
        messages_json=json.dumps(messages, ensure_ascii=False),
        status="in_progress",
        wait_turns=1,
        created_at=now,
        updated_at=now,
    )
    db.add(row)
    db.flush()
    return row


def get_pending(db: Session, pending_id: str) -> PendingRequest | None:
    return db.get(PendingRequest, pending_id)


def mark_pending_ready(
    db: Session, pending_id: str, response_text: str
) -> None:
    db.execute(
        update(PendingRequest)
        .where(PendingRequest.id == pending_id)
        .values(
            status="ready",
            response_text=response_text,
            updated_at=utcnow(),
        )
    )


def mark_pending_error(
    db: Session, pending_id: str, error_text: str
) -> None:
    db.execute(
        update(PendingRequest)
        .where(PendingRequest.id == pending_id)
        .values(
            status="error",
            error_text=error_text,
            updated_at=utcnow(),
        )
    )


def mark_pending_aborted(db: Session, pending_id: str) -> None:
    db.execute(
        update(PendingRequest)
        .where(PendingRequest.id == pending_id)
        .values(status="aborted", updated_at=utcnow())
    )


def bump_pending_wait_turns(
    db: Session, pending_id: str, wait_turns: int
) -> None:
    db.execute(
        update(PendingRequest)
        .where(PendingRequest.id == pending_id)
        .values(wait_turns=wait_turns, updated_at=utcnow())
    )


def mark_orphaned_in_progress_as_error(db: Session) -> int:
    """Called on startup. Pending requests with in-process tasks did not survive."""
    result = db.execute(
        update(PendingRequest)
        .where(PendingRequest.status == "in_progress")
        .values(
            status="error",
            error_text="process restart",
            updated_at=utcnow(),
        )
    )
    return result.rowcount or 0
