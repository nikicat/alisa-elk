"""Nodes for the parent dialog graph.

Each node reads from `DialogState` and writes a partial update. Runtime
dependencies (LLM client, DB session factory, pending-task registry)
are passed through `config["configurable"]["deps"]` — LangGraph's
standard slot for per-invocation context — so the compiled graph
itself is stateless and shareable across all requests.

Phase 2 pulled the slow path off the `pending_requests` table: the
`asyncio.Task` parked in `PendingTaskRegistry` is the only authority
on whether a slow LLM call has produced a result. `idle_llm` snapshots
the dispatch context into `pending` and returns a wait phrase;
subsequent turns enter `check_pending`, which reads `task.result()`
once `task.done()` flips.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

import structlog
from langchain_core.runnables import RunnableConfig

from app import config as config_mod
from app import persona, repo
from app.dialog.state import CursorState, DialogState, PendingState
from app.dialog.tools import IDLE_TOOLS_OPENAI
from app.games import words as words_game
from app.linking import detect_code
from app.llm import LLMResult
from app.memory import load_recent_turns
from app.pagination import chunk_for_alice
from app.wait import wait_for_or_keepalive

if TYPE_CHECKING:
    from app.handler import HandlerDeps

log = structlog.get_logger(__name__)


# ---------- helpers ----------


def _deps(config: RunnableConfig) -> HandlerDeps:
    return config["configurable"]["deps"]  # type: ignore[index]


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


def _parse_context_since(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def _continuation_buttons() -> list[dict[str, Any]]:
    return [{"title": "Дальше", "hide": True}, {"title": "Хватит", "hide": True}]


def _empty_output() -> dict[str, Any]:
    """Default per-turn output skeleton — wiped on every turn before
    nodes start adding to it, so a previous turn's `end_session=True`
    can't leak into the next."""
    return {
        "last_bot_text": None,
        "response_buttons": None,
        "end_session": False,
        "session_state_extras": {},
    }


# ---------- nodes ----------


async def turn_init(state: DialogState, config: RunnableConfig) -> dict:
    """Reset per-turn output fields and stamp wall-clock start time."""
    return {
        **_empty_output(),
        "started_at": time.monotonic(),
    }


async def check_pending(state: DialogState, config: RunnableConfig) -> dict:
    """Slow-path follow-up: a previous turn left `pending_id` in state.

    Phase 2 moved result handoff out of `pending_requests` into the
    `asyncio.Task` parked in `PendingTaskRegistry`. We read `task.result()`
    once the task is done — no DB poll, no background persister.

    On affirmative follow-up, wait a bit more (`subsequent_wait_timeout_s`).
    On negative follow-up, cancel. Anything else cancels and recurses
    fresh — the entry router will re-dispatch the new command. Process
    restart (registry empty after restart) is treated as orphan-recurse.
    """
    deps = _deps(config)
    cfg = config_mod.get_config()
    command = state.get("user_input") or ""
    pending = state.get("pending") or {}
    pending_id = pending.get("pending_id")
    wait_turns = int(pending.get("wait_turns", 1))
    assert pending_id is not None, "check_pending entered without pending_id"

    is_yes = _matches_any(command, persona.AFFIRMATIVE_WORDS)
    is_no = _matches_any(command, persona.NEGATIVE_WORDS)
    is_exit = _matches_any(command, persona.EXIT_WORDS)

    # Exit during wait — end the dialog immediately.
    if is_exit:
        deps.registry.cancel(pending_id)
        return {
            "pending": None,
            "last_bot_text": persona.FAREWELL,
            "end_session": True,
        }

    task = deps.registry.get(pending_id)
    if task is None:
        # Orphan (process restart, or task already drained): drop the
        # ghost pending and let the router re-dispatch on this turn.
        return {"pending": None, "_recurse": True}

    if is_no:
        if not task.done():
            task.cancel()
        deps.registry.discard(pending_id)
        return {"pending": None, "last_bot_text": persona.ABORT_OK}

    if not is_yes:
        # Treat as new question: cancel old, recurse fresh.
        if not task.done():
            task.cancel()
        deps.registry.discard(pending_id)
        return {"pending": None, "_recurse": True}

    # "да" → wait some more on the running task.
    if not task.done():
        timeout = float(cfg["llm"]["subsequent_wait_timeout_s"])
        with contextlib.suppress(Exception):
            await wait_for_or_keepalive(task, timeout)

    if task.done():
        deps.registry.discard(pending_id)
        return _apply_task_result(state, pending, task, deps, cfg)

    # Still in progress: escalate or give up.
    next_wait_turns = wait_turns + 1
    max_turns = int(cfg["llm"]["max_wait_turns"])
    phrases: list[str] = cfg["persona"]["wait_phrases"]
    if next_wait_turns > max_turns:
        if not task.done():
            task.cancel()
        deps.registry.discard(pending_id)
        log.warning(
            "llm_give_up",
            session_id=state.get("session_id"),
            pending_id=pending_id,
            wait_turns=wait_turns,
        )
        return {"pending": None, "last_bot_text": cfg["persona"]["give_up_phrase"]}
    phrase = phrases[min(next_wait_turns - 1, len(phrases) - 1)]
    log.info(
        "llm_wait_phrase",
        path="escalate",
        session_id=state.get("session_id"),
        pending_id=pending_id,
        wait_turns=next_wait_turns,
    )
    return {
        "pending": {**pending, "wait_turns": next_wait_turns},
        "last_bot_text": phrase,
    }


def _apply_task_result(
    state: DialogState,
    pending: dict,
    task: asyncio.Task[LLMResult],
    deps: HandlerDeps,
    cfg: dict,
) -> dict:
    """Drain a done task into a DialogState update. Errors / cancellations
    surface as `persona.LLM_ERROR`; a `reset_context` tool call inside
    the result is translated into a clean reset turn."""
    pending_id = pending.get("pending_id")
    session_id = state.get("session_id") or ""
    try:
        result = task.result()
    except asyncio.CancelledError:
        log.info(
            "llm_task_cancelled",
            session_id=session_id,
            pending_id=pending_id,
        )
        return {"pending": None, "last_bot_text": persona.LLM_ERROR}
    except Exception as exc:
        log.warning(
            "llm_task_error",
            session_id=session_id,
            pending_id=pending_id,
            error=str(exc),
        )
        return {"pending": None, "last_bot_text": persona.LLM_ERROR}

    if "reset_context" in result.tool_calls:
        log.info(
            "llm_reset_tool",
            path="slow",
            session_id=session_id,
            pending_id=pending_id,
        )
        return {
            "pending": None,
            "last_bot_text": persona.RESET_OK,
            "context_since": repo.utcnow().isoformat(),
        }

    log.info(
        "llm_response",
        path="slow",
        response=result.text,
        session_id=session_id,
        pending_id=pending_id,
    )

    started_at = pending.get("llm_started_at") or state.get("started_at") or time.monotonic()
    total_ms = int((time.monotonic() - started_at) * 1000)
    application_id = pending.get("application_id") or state.get("application_id") or ""
    message_id = int(pending.get("message_id") or state.get("message_id") or 0)
    request_text = pending.get("request_text") or ""
    user_id = pending.get("user_id")

    with deps.session_factory() as db:
        chunk, next_cursor = _store_paginated(
            db,
            response_text=result.text,
            user_id=user_id,
            application_id=application_id,
            session_id=session_id,
            message_id=message_id,
            request_text=request_text,
            total_ms=total_ms,
            llm_ms=None,
            pending_request_id=pending_id,
            chunk_chars=cfg["pagination"]["chunk_chars"],
        )
        db.commit()
    update: dict[str, Any] = {"pending": None, "last_bot_text": chunk}
    if next_cursor:
        update["cursor"] = next_cursor
        update["response_buttons"] = _continuation_buttons()
    else:
        update["cursor"] = None
    return update




def _store_paginated(
    db,
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
) -> tuple[str, CursorState | None]:
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
    return first, CursorState(turn_id=turn.id, idx=1)


async def pagination_continue_node(
    state: DialogState,
    config: RunnableConfig,
) -> dict:
    """Player asked for the next chunk of the previous response."""
    deps = _deps(config)
    cfg = config_mod.get_config()
    cursor = state.get("cursor") or {}
    with deps.session_factory() as db:
        turn = repo.get_turn(db, int(cursor.get("turn_id", 0)))
    if turn is None:
        return {
            "cursor": None,
            "last_bot_text": "Я уже забыл, о чём шла речь. Спроси заново.",
        }
    chunks = chunk_for_alice(
        turn.response_text, chunk_chars=cfg["pagination"]["chunk_chars"]
    )
    idx = int(cursor.get("idx", 0))
    if idx >= len(chunks):
        return {"cursor": None, "last_bot_text": "Это всё."}
    chunk = chunks[idx]
    if idx + 1 < len(chunks):
        return {
            "cursor": CursorState(turn_id=turn.id, idx=idx + 1),
            "last_bot_text": chunk + " " + persona.CONTINUE_PROMPT,
            "response_buttons": _continuation_buttons(),
        }
    return {"cursor": None, "last_bot_text": chunk}


async def linking_node(state: DialogState, config: RunnableConfig) -> dict:
    """Device is unlinked. Either consume a dictated code or prompt."""
    deps = _deps(config)
    original = state.get("original_utterance") or state.get("user_input") or ""
    tokens = state.get("nlu_tokens") or []
    application_id = state.get("application_id") or ""
    code = detect_code(original, tokens)
    if code:
        with deps.session_factory() as db:
            linked = repo.consume_link_code(db, code, application_id)
            if linked is not None:
                db.commit()
                return {"last_bot_text": persona.LINK_OK}
            db.rollback()
            return {"last_bot_text": persona.LINK_BAD}
    command = (state.get("user_input") or "").strip()
    if state.get("new_session") and not command:
        return {"last_bot_text": persona.UNLINKED_GREETING}
    return {"last_bot_text": persona.UNLINKED_QUESTION}


async def greeting_node(state: DialogState, config: RunnableConfig) -> dict:  # noqa: ARG001
    return {"last_bot_text": persona.GREETING}


async def silence_node(state: DialogState, config: RunnableConfig) -> dict:  # noqa: ARG001
    return {"last_bot_text": "Слушаю."}


async def help_node(state: DialogState, config: RunnableConfig) -> dict:  # noqa: ARG001
    return {"last_bot_text": persona.HELP_TEXT}


async def exit_skill_node(state: DialogState, config: RunnableConfig) -> dict:
    """LLM said the user wants to leave. Also cancel any in-flight task."""
    deps = _deps(config)
    pending = state.get("pending") or {}
    pid = pending.get("pending_id")
    if pid:
        deps.registry.cancel(pid)
    return {
        "pending": None,
        "last_bot_text": persona.FAREWELL,
        "end_session": True,
    }


async def reset_context_node(state: DialogState, config: RunnableConfig) -> dict:
    """LLM said the user wants to wipe conversational memory."""
    deps = _deps(config)
    pending = state.get("pending") or {}
    pid = pending.get("pending_id")
    if pid:
        deps.registry.cancel(pid)
    return {
        "pending": None,
        "context_since": repo.utcnow().isoformat(),
        "last_bot_text": persona.RESET_OK,
    }


async def idle_llm(state: DialogState, config: RunnableConfig) -> dict:
    """Idle LLM dispatch: fast-path or wait-phrase handoff.

    Phase 2: the slow path no longer writes a `pending_requests` row or
    fires a background persister — the live `asyncio.Task` parked in
    `PendingTaskRegistry` is itself the result mailbox, and
    `check_pending` reads `task.result()` directly on the follow-up
    turn. The pending snapshot in `DialogState` is the only durable
    state that survives a restart; the task itself does not.
    """
    deps = _deps(config)
    cfg = config_mod.get_config()
    command = (state.get("user_input") or "").strip()
    if len(command) > 500:
        command = command[:500]
    application_id = state.get("application_id") or ""
    session_id = state.get("session_id") or ""
    message_id = int(state.get("message_id") or 0)
    context_since = _parse_context_since(state.get("context_since"))
    started_at = state.get("started_at") or time.monotonic()

    with deps.session_factory() as db:
        user = repo.get_user_by_app_id(db, application_id)
        history = load_recent_turns(
            db, session_id, cfg["memory"]["recent_turns"], since=context_since
        )
    user_id = user.id if user is not None else None

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
            tools=IDLE_TOOLS_OPENAI,
        )
    )
    deps.registry.register(new_pending_id, task)
    timeout = float(cfg["llm"]["first_wait_timeout_s"])
    try:
        result = await wait_for_or_keepalive(task, timeout)
    except Exception as exc:
        deps.registry.discard(new_pending_id)
        log.warning("llm_first_call_error", error=str(exc))
        return {"last_bot_text": persona.LLM_ERROR}
    except BaseException:
        if not task.done():
            task.cancel()
        deps.registry.discard(new_pending_id)
        raise

    if result is not None:
        # Fast path.
        deps.registry.discard(new_pending_id)
        llm_ms = int((time.monotonic() - llm_t0) * 1000)
        total_ms = int((time.monotonic() - started_at) * 1000)
        # Tool dispatch — return state markers; routing layer steers to
        # the matching short-circuit node (which produces the user text).
        if "reset_context" in result.tool_calls:
            log.info("llm_reset_tool", session_id=session_id, llm_ms=llm_ms)
            return {"_tool_call": "reset_context"}
        if "exit_skill" in result.tool_calls:
            log.info("llm_exit_tool", session_id=session_id, llm_ms=llm_ms)
            return {"_tool_call": "exit_skill"}
        if "help" in result.tool_calls:
            log.info("llm_help_tool", session_id=session_id, llm_ms=llm_ms)
            return {"_tool_call": "help"}
        if "enter_game" in result.tool_calls:
            log.info("llm_enter_game_tool", session_id=session_id, llm_ms=llm_ms)
            return {"_tool_call": "enter_game"}

        log.info(
            "llm_response",
            path="fast",
            response=result.text,
            session_id=session_id,
            llm_ms=llm_ms,
        )
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
        update: dict[str, Any] = {"last_bot_text": chunk}
        if next_cursor:
            update["cursor"] = next_cursor
            update["response_buttons"] = _continuation_buttons()
        else:
            update["cursor"] = None
        return update

    # Slow path: snapshot the dispatch context into state, leave the
    # task in the registry, return a wait phrase. The next turn's
    # `check_pending` reads `task.result()` directly.
    log.info(
        "llm_wait_phrase",
        path="first",
        session_id=session_id,
        pending_id=new_pending_id,
        wait_turns=1,
    )
    return {
        "pending": PendingState(
            pending_id=new_pending_id,
            wait_turns=1,
            request_text=command,
            user_id=user_id,
            application_id=application_id,
            message_id=message_id,
            llm_started_at=started_at,
        ),
        "last_bot_text": cfg["persona"]["wait_phrases"][0],
    }


# ---------- words subgraph nodes (wired via state.game) ----------


async def enter_game_node(state: DialogState, config: RunnableConfig) -> dict:  # noqa: ARG001
    """LLM emitted `enter_game(words)`. Set up the game state and let the
    bot pick the first word via the shared words_intro routine."""
    fresh = words_game.GameState(used=[], required_letter=None, last_cheat=None)
    intro_update = await words_game.words_intro({"game": fresh})  # type: ignore[arg-type]
    # words_intro returns {game, messages, last_bot_text} — that's
    # already a DialogState-compatible delta.
    return intro_update


async def words_classify_player_node(
    state: DialogState,
    config: RunnableConfig,
) -> dict:
    return await words_game.words_classify_player(state)  # type: ignore[arg-type]


async def words_validate_node(state: DialogState, config: RunnableConfig) -> dict:
    return await words_game.words_validate(state)  # type: ignore[arg-type]


async def words_bot_turn_node(state: DialogState, config: RunnableConfig) -> dict:
    return await words_game.words_bot_turn(state)  # type: ignore[arg-type]


async def words_resolve_challenge_node(
    state: DialogState,
    config: RunnableConfig,
) -> dict:
    return await words_game.words_resolve_challenge(state)  # type: ignore[arg-type]
