"""DialogState — the parent graph's checkpointed state shape.

Mirrors what `app/handler.py:_route_inner` used to read out of
`AliceRequest.state.session`. Persistence is by `thread_id =
req.session.session_id`, so the lifetime matches Yandex's session.

Game state is carried as a nested `GameState` (defined in
`app/games/words.py`). Pending/cursor are kept as flat sub-dicts; we
will collapse them into typed pydantic models if/when the surface gets
wider, but Phase 1 keeps them in the legacy shape so a checkpoint
written by either codebase stays interchangeable for ease of rollback.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from app.games.words import GameState


class PendingState(TypedDict, total=False):
    pending_id: str
    wait_turns: int


class CursorState(TypedDict, total=False):
    turn_id: int
    idx: int


class DialogState(TypedDict, total=False):
    # ----- inputs assembled by handler.py for each turn -----
    user_input: str | None
    original_utterance: str | None
    nlu_tokens: list[str]
    application_id: str
    session_id: str
    message_id: int
    new_session: bool
    started_at: float

    # ----- persisted dialog state -----
    messages: Annotated[list[BaseMessage], add_messages]
    pending: PendingState | None
    cursor: CursorState | None
    context_since: str | None  # ISO-8601 or None
    game: GameState | None

    # ----- output the driver consumes for each turn -----
    last_bot_text: str | None
    response_buttons: list[dict[str, Any]] | None
    end_session: bool
    session_state_extras: dict[str, Any]

    # ----- transient per-turn routing markers (not user-facing) -----
    _tool_call: str | None
