import asyncio
import contextlib
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import structlog
from sqlalchemy.orm import Session

from app import config as config_mod
from app import persona, repo
from app.linking import detect_code
from app.llm import LLMClient, LLMResult
from app.memory import load_recent_turns
from app.pagination import chunk_for_alice
from app.schemas import AliceRequest, AliceResponse, Button, Response
from app.wait import PendingTaskRegistry, wait_for_or_keepalive

log = structlog.get_logger(__name__)


@dataclass
class HandlerDeps:
    session_factory: Callable[[], Session]
    llm: LLMClient
    registry: PendingTaskRegistry


def _normalize(text: str) -> str:
    return text.strip().lower().replace("ё", "е")


def _matches_any(command: str, words: frozenset[str]) -> bool:
    n = _normalize(command)
    if not n:
        return False
    if n in words:
        return True
    return any(
        n.startswith(w + " ") or n.endswith(" " + w) or f" {w} " in n for w in words
    )


def _make(
    text: str,
    *,
    session_state: dict[str, Any] | None = None,
    buttons: list[dict[str, Any]] | None = None,
    end_session: bool = False,
) -> AliceResponse:
    if len(text) > 1024:
        text = text[:1024]
    response = Response(text=text, tts=text, end_session=end_session)
    if buttons:
        response.buttons = [Button(**b) for b in buttons]
    return AliceResponse(
        response=response, session_state=session_state or {}, version="1.0"
    )


def _continuation_buttons() -> list[dict[str, Any]]:
    return [{"title": "Дальше", "hide": True}, {"title": "Хватит", "hide": True}]


async def _persist_result(
    task: asyncio.Task[LLMResult],
    pending_id: str,
    session_factory: Callable[[], Session],
    registry: PendingTaskRegistry,
) -> None:
    """Await the LLM task; persist its outcome to the pending_requests row.

    Runs as a fire-and-forget background task. Always closes its own DB session.
    """
    try:
        result = await task
        with session_factory() as db:
            repo.mark_pending_ready(db, pending_id, result.text)
            db.commit()
    except asyncio.CancelledError:
        with session_factory() as db:
            repo.mark_pending_aborted(db, pending_id)
            db.commit()
        raise
    except Exception as exc:
        log.warning("llm_task_error", pending_id=pending_id, error=str(exc))
        with session_factory() as db:
            repo.mark_pending_error(db, pending_id, str(exc)[:500])
            db.commit()
    finally:
        registry.discard(pending_id)


def _store_paginated(
    db: Session,
    *,
    response_text: str,
    user_id: int | None,
    application_id: str,
    session_id: str,
    message_id: int,
    request_text: str,
    total_ms: int,
    llm_ms: int | None,
    pending_request_id: str | None,
    chunk_chars: int,
) -> tuple[str, dict[str, Any] | None]:
    """Log the turn and return (first_chunk, cursor-or-None)."""
    chunks = chunk_for_alice(response_text, chunk_chars=chunk_chars)
    turn = repo.log_turn(
        db,
        user_id=user_id,
        yandex_application_id=application_id,
        session_id=session_id,
        message_id=message_id,
        request_text=request_text,
        response_text=response_text,
        total_ms=total_ms,
        llm_ms=llm_ms,
        pending_request_id=pending_request_id,
    )
    if len(chunks) == 1:
        return chunks[0], None
    first = chunks[0] + " " + persona.CONTINUE_PROMPT
    return first, {"turn_id": turn.id, "idx": 1}


def _clear_session_state(req: AliceRequest, *keys: str) -> AliceRequest:
    cleaned = {k: v for k, v in req.state.session.items() if k not in keys}
    return req.model_copy(
        update={"state": req.state.model_copy(update={"session": cleaned})}
    )


async def route(req: AliceRequest, deps: HandlerDeps) -> AliceResponse:
    """Public entry. Routes the request, then forwards context_since so the
    user's "forget the conversation" intent persists across subsequent turns."""
    context_since = req.state.session.get("context_since")
    resp = await _route_inner(req, deps)
    if context_since and not resp.response.end_session:
        resp.session_state.setdefault("context_since", context_since)
    return resp


async def _route_inner(req: AliceRequest, deps: HandlerDeps) -> AliceResponse:
    started_at = time.monotonic()
    settings = config_mod.get_settings()
    cfg = config_mod.get_config()

    # 1. Auth gate.
    if settings.YANDEX_SKILL_ID and req.session.skill_id != settings.YANDEX_SKILL_ID:
        log.warning(
            "skill_id_mismatch",
            got=req.session.skill_id,
            expected=settings.YANDEX_SKILL_ID,
        )
        return _make("Лес не узнаёт тебя.", end_session=True)

    command = req.request.command or ""
    original = req.request.original_utterance or command
    application_id = req.session.application.application_id
    session_id = req.session.session_id
    message_id = req.session.message_id

    session_state: dict[str, Any] = dict(req.state.session)
    pending_id = session_state.get("pending_id")
    cursor = session_state.get("cursor")
    context_since_raw = session_state.get("context_since")
    context_since: datetime | None = None
    if isinstance(context_since_raw, str):
        try:
            context_since = datetime.fromisoformat(context_since_raw)
        except ValueError:
            context_since = None

    # 2. Exit.
    if _matches_any(command, persona.EXIT_WORDS):
        if pending_id:
            deps.registry.cancel(pending_id)
        return _make(persona.FAREWELL, end_session=True)

    # 3. Help.
    if _matches_any(command, persona.HELP_WORDS):
        return _make(persona.HELP_TEXT)

    # 4. Reset conversation context (keep the dialog open).
    if _matches_any(command, persona.RESET_WORDS):
        if pending_id:
            deps.registry.cancel(pending_id)
        return _make(
            persona.RESET_OK,
            session_state={"context_since": repo.utcnow().isoformat()},
        )

    # 5. WAITING state.
    if pending_id:
        return await _handle_waiting(
            req=req,
            deps=deps,
            pending_id=pending_id,
            wait_turns=int(session_state.get("wait_turns", 1)),
            command=command,
            application_id=application_id,
            session_id=session_id,
            message_id=message_id,
            started_at=started_at,
        )

    # 6. CONTINUATION state.
    if cursor and _matches_any(command, persona.CONTINUE_WORDS):
        with deps.session_factory() as db:
            turn = repo.get_turn(db, int(cursor["turn_id"]))
            if turn is None:
                return _make("Я уже забыл, о чём шла речь. Спроси заново.")
            chunks = chunk_for_alice(
                turn.response_text, chunk_chars=cfg["pagination"]["chunk_chars"]
            )
            idx = int(cursor["idx"])
            if idx >= len(chunks):
                return _make("Это всё.")
            chunk = chunks[idx]
            if idx + 1 < len(chunks):
                return _make(
                    chunk + " " + persona.CONTINUE_PROMPT,
                    session_state={"cursor": {"turn_id": turn.id, "idx": idx + 1}},
                    buttons=_continuation_buttons(),
                )
            return _make(chunk)

    # 7. Resolve user — must come before code detection so a linked user
    #    dictating digits inside a question doesn't get hijacked into the
    #    link-code path.
    with deps.session_factory() as db:
        user = repo.get_user_by_app_id(db, application_id)

    # 8. Reverse-code linking — only meaningful when the device isn't linked.
    if user is None:
        code = detect_code(original, req.request.nlu.tokens)
        if code:
            with deps.session_factory() as db:
                linked = repo.consume_link_code(db, code, application_id)
                if linked is not None:
                    db.commit()
                    return _make(persona.LINK_OK)
                db.rollback()
                return _make(persona.LINK_BAD)
        if req.session.new and not command:
            return _make(persona.UNLINKED_GREETING)
        return _make(persona.UNLINKED_QUESTION)
    user_id = user.id

    # 9. Greeting fallback.
    if req.session.new and not command:
        return _make(persona.GREETING)
    if not command:
        return _make("Слушаю.")
    if len(command) > 500:
        command = command[:500]

    # 10. LLM dispatch.
    with deps.session_factory() as db:
        history = load_recent_turns(
            db, session_id, cfg["memory"]["recent_turns"], since=context_since
        )
    messages = [
        {"role": "system", "content": persona.SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": command},
    ]
    new_pending_id = uuid.uuid4().hex
    llm_t0 = time.monotonic()
    task: asyncio.Task[LLMResult] = asyncio.create_task(
        deps.llm.complete(
            messages,
            max_tokens=cfg["llm"]["max_tokens"],
            temperature=cfg["llm"]["temperature"],
        )
    )
    deps.registry.register(new_pending_id, task)
    timeout = float(cfg["llm"]["first_wait_timeout_s"])
    try:
        result = await wait_for_or_keepalive(task, timeout)
    except Exception as exc:
        deps.registry.discard(new_pending_id)
        log.warning("llm_first_call_error", error=str(exc))
        return _make(persona.LLM_ERROR)
    except BaseException:
        # ASGI cancellation (client disconnect, shutdown). The shielded task
        # would otherwise keep burning tokens with no consumer.
        if not task.done():
            task.cancel()
        deps.registry.discard(new_pending_id)
        raise

    if result is not None:
        # Fast path.
        deps.registry.discard(new_pending_id)
        llm_ms = int((time.monotonic() - llm_t0) * 1000)
        total_ms = int((time.monotonic() - started_at) * 1000)
        with deps.session_factory() as db:
            chunk, next_cursor = _store_paginated(
                db,
                response_text=result.text,
                user_id=user_id,
                application_id=application_id,
                session_id=session_id,
                message_id=message_id,
                request_text=command,
                total_ms=total_ms,
                llm_ms=llm_ms,
                pending_request_id=None,
                chunk_chars=cfg["pagination"]["chunk_chars"],
            )
            db.commit()
        new_state: dict[str, Any] = {}
        buttons: list[dict[str, Any]] | None = None
        if next_cursor:
            new_state["cursor"] = next_cursor
            buttons = _continuation_buttons()
        return _make(chunk, session_state=new_state, buttons=buttons)

    # Slow path: persist pending row, fire wrapper, return wait phrase.
    try:
        with deps.session_factory() as db:
            repo.create_pending(
                db,
                pending_id=new_pending_id,
                user_id=user_id,
                session_id=session_id,
                request_text=command,
                messages=messages,
            )
            db.commit()
    except Exception as exc:
        # Couldn't persist the pending row — cancel the orphaned LLM task
        # rather than leaking it.
        if not task.done():
            task.cancel()
        deps.registry.discard(new_pending_id)
        log.warning("pending_create_failed", error=str(exc))
        return _make(persona.LLM_ERROR)
    asyncio.create_task(
        _persist_result(task, new_pending_id, deps.session_factory, deps.registry)
    )
    return _make(
        cfg["persona"]["wait_phrases"][0],
        session_state={"pending_id": new_pending_id, "wait_turns": 1},
    )


async def _handle_waiting(
    *,
    req: AliceRequest,
    deps: HandlerDeps,
    pending_id: str,
    wait_turns: int,
    command: str,
    application_id: str,
    session_id: str,
    message_id: int,
    started_at: float,
) -> AliceResponse:
    cfg = config_mod.get_config()
    is_yes = _matches_any(command, persona.AFFIRMATIVE_WORDS)
    is_no = _matches_any(command, persona.NEGATIVE_WORDS)

    with deps.session_factory() as db:
        pending = repo.get_pending(db, pending_id)

    # Pending row missing or already aborted — fall through.
    if pending is None or pending.status == "aborted":
        deps.registry.discard(pending_id)
        cleared = _clear_session_state(req, "pending_id", "wait_turns")
        return await route(cleared, deps)

    if pending.status == "ready":
        deps.registry.discard(pending_id)
        return _ready_response(
            deps=deps,
            pending=pending,
            application_id=application_id,
            session_id=session_id,
            message_id=message_id,
            started_at=started_at,
            cfg=cfg,
        )

    if pending.status == "error":
        deps.registry.discard(pending_id)
        return _make(persona.LLM_ERROR)

    # Still in_progress.
    if is_no:
        deps.registry.cancel(pending_id)
        return _make(persona.ABORT_OK)

    if not is_yes:
        # Treat as new question: cancel old, recurse with cleared state.
        deps.registry.cancel(pending_id)
        cleared = _clear_session_state(req, "pending_id", "wait_turns")
        return await route(cleared, deps)

    # "да": wait some more on the still-running task.
    task = deps.registry.get(pending_id)
    if task is not None:
        timeout = float(cfg["llm"]["subsequent_wait_timeout_s"])
        with contextlib.suppress(Exception):
            await wait_for_or_keepalive(task, timeout)

    with deps.session_factory() as db:
        pending = repo.get_pending(db, pending_id)

    if pending is None:
        return _make(persona.LLM_ERROR)
    if pending.status == "ready":
        deps.registry.discard(pending_id)
        return _ready_response(
            deps=deps,
            pending=pending,
            application_id=application_id,
            session_id=session_id,
            message_id=message_id,
            started_at=started_at,
            cfg=cfg,
        )
    if pending.status == "error":
        deps.registry.discard(pending_id)
        return _make(persona.LLM_ERROR)

    # Still in progress: escalate or give up.
    next_wait_turns = wait_turns + 1
    max_turns = int(cfg["llm"]["max_wait_turns"])
    phrases: list[str] = cfg["persona"]["wait_phrases"]
    if next_wait_turns > max_turns:
        deps.registry.cancel(pending_id)
        return _make(cfg["persona"]["give_up_phrase"])
    phrase = phrases[min(next_wait_turns - 1, len(phrases) - 1)]
    with deps.session_factory() as db:
        repo.bump_pending_wait_turns(db, pending_id, next_wait_turns)
        db.commit()
    return _make(
        phrase,
        session_state={"pending_id": pending_id, "wait_turns": next_wait_turns},
    )


def _ready_response(
    *,
    deps: HandlerDeps,
    pending: Any,
    application_id: str,
    session_id: str,
    message_id: int,
    started_at: float,
    cfg: dict[str, Any],
) -> AliceResponse:
    total_ms = int((time.monotonic() - started_at) * 1000)
    with deps.session_factory() as db:
        chunk, next_cursor = _store_paginated(
            db,
            response_text=pending.response_text or "",
            user_id=pending.user_id,
            application_id=application_id,
            session_id=session_id,
            message_id=message_id,
            request_text=pending.request_text,
            total_ms=total_ms,
            llm_ms=None,
            pending_request_id=pending.id,
            chunk_chars=cfg["pagination"]["chunk_chars"],
        )
        db.commit()
    new_state: dict[str, Any] = {}
    buttons: list[dict[str, Any]] | None = None
    if next_cursor:
        new_state["cursor"] = next_cursor
        buttons = _continuation_buttons()
    return _make(chunk, session_state=new_state, buttons=buttons)
