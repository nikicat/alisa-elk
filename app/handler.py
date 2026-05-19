"""Thin LangGraph driver — Phase 1 of the handler-to-langgraph migration.

`route(req, deps)` is the public entry point that the FastAPI webhook
calls. Internally it forwards the request into a compiled LangGraph
keyed by `req.session.session_id`, then unpacks the checkpointed state
into an `AliceResponse`. Dispatch logic, dialog state, and the slow-path
machinery all live inside the graph (`app/dialog`).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import structlog
from langchain_core.runnables import RunnableConfig
from sqlalchemy.orm import Session

from app.llm import LLMClient
from app.schemas import AliceRequest, AliceResponse, Button, Response
from app.wait import PendingTaskRegistry

log = structlog.get_logger(__name__)


@dataclass
class HandlerDeps:
    session_factory: Callable[[], Session]
    llm: LLMClient
    registry: PendingTaskRegistry
    graph: Any = (
        None  # `CompiledStateGraph`; typed loosely to dodge a hard import here.
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


def _inputs_from_req(req: AliceRequest) -> dict[str, Any]:
    """Build the per-turn DialogState slice from an AliceRequest.

    Persisted state (`messages`, `pending`, `cursor`, `context_since`,
    `game`) is restored by the checkpointer; we only need to inject
    transient request fields.
    """
    command = req.request.command or ""
    return {
        "user_input": command,
        "original_utterance": req.request.original_utterance or command,
        "nlu_tokens": list(req.request.nlu.tokens),
        "application_id": req.session.application.application_id,
        "session_id": req.session.session_id,
        "message_id": req.session.message_id,
        "new_session": req.session.new,
        # Clear transient routing markers before the turn runs.
        "_tool_call": None,
    }


def _bridge_session_state(req: AliceRequest, result: dict[str, Any]) -> dict[str, Any]:
    """Yandex session_state mirror.

    The LangGraph checkpointer owns the canonical dialog state, keyed
    by `thread_id == session_id`, so this mirror is technically
    redundant. We still publish `pending_id`/`wait_turns`/`cursor`/
    `context_since` because:

      - they make request/response traces readable without consulting
        the checkpoint DB, and
      - existing observability + tests inspect them directly.

    The mirror is read-only — incoming `state.session` is ignored;
    only the checkpoint is authoritative.
    """
    out: dict[str, Any] = {"thread_id": req.session.session_id}
    pending = result.get("pending") or {}
    if pending.get("pending_id"):
        out["pending_id"] = pending["pending_id"]
        out["wait_turns"] = pending.get("wait_turns", 1)
    cursor = result.get("cursor") or {}
    if cursor.get("turn_id"):
        out["cursor"] = {
            "turn_id": cursor["turn_id"],
            "idx": cursor.get("idx", 1),
        }
    if result.get("context_since"):
        out["context_since"] = result["context_since"]
    return out


async def route(req: AliceRequest, deps: HandlerDeps) -> AliceResponse:
    """Public entry point. Invokes the parent dialog graph and renders
    the resulting state into an `AliceResponse`."""
    command = req.request.command or ""
    if command:
        log.info(
            "user_phrase",
            command=command,
            session_id=req.session.session_id,
            application_id=req.session.application.application_id,
            message_id=req.session.message_id,
        )

    config: RunnableConfig = {
        "configurable": {
            "thread_id": req.session.session_id,
            "deps": deps,
        },
    }
    inputs = _inputs_from_req(req)
    result = await deps.graph.ainvoke(inputs, config=config)

    text = result.get("last_bot_text") or "Слушаю."
    end_session = bool(result.get("end_session"))
    buttons = result.get("response_buttons")

    session_state = _bridge_session_state(req, result)
    return _make(
        text,
        session_state=session_state,
        buttons=buttons,
        end_session=end_session,
    )
