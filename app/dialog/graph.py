"""Parent dialog graph.

Composes the legacy handler's dispatch tree as a LangGraph state
machine. Game nodes from `app.games.words` are embedded directly so
their `game`/`messages` updates merge into the parent state without
needing a nested subgraph compile.

Topology, in priority order from `entry_router`:

  pending      → check_pending       (LLM classifies; may recurse)
  cursor       → pagination_continue (LLM classifies; may recurse)
  unlinked     → linking_node
  in_game      → words_classify_player_node → …
  empty+new    → greeting_node
  empty        → silence_node
  command      → idle_llm            (LLM dispatch with full tool surface)

Phase 3 dropped the EXIT/HELP/RESET/CONTINUE/AFFIRMATIVE/NEGATIVE
keyword shortcuts; every intent now reaches an LLM dispatch. The idle
LLM uses the full tool surface (`reset_context` / `exit_skill` /
`help` / `enter_game`); the wait and pagination branches use focused
classifier tool sets defined in `app.dialog.tools`.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from app.dialog.nodes import (
    check_pending,
    enter_game_node,
    exit_skill_node,
    greeting_node,
    help_node,
    idle_llm,
    linking_node,
    pagination_continue_node,
    reset_context_node,
    silence_node,
    turn_init,
    words_bot_turn_node,
    words_classify_player_node,
    words_resolve_challenge_node,
    words_validate_node,
)
from app.dialog.state import DialogState

# ---------- routers ----------


def _has_pending(state: DialogState) -> bool:
    pending = state.get("pending") or {}
    return bool(pending.get("pending_id"))


def _has_cursor(state: DialogState) -> bool:
    cursor = state.get("cursor") or {}
    return bool(cursor.get("turn_id"))


def _in_game(state: DialogState) -> bool:
    return state.get("game") is not None


def _unlinked(state: DialogState, config: RunnableConfig) -> bool:
    from app import repo

    application_id = state.get("application_id") or ""
    deps = config["configurable"]["deps"]  # type: ignore[index]
    with deps.session_factory() as db:
        return repo.get_user_by_app_id(db, application_id) is None


EntryTarget = Literal[
    "waiting",
    "paginate",
    "linking",
    "in_game",
    "greeting",
    "silence",
    "idle",
]


def route_from_entry(
    state: DialogState,
    config: RunnableConfig,
) -> EntryTarget:
    command = (state.get("user_input") or "").strip()
    if _has_pending(state):
        return "waiting"
    if _has_cursor(state) and command:
        return "paginate"
    if _unlinked(state, config):
        return "linking"
    if _in_game(state):
        return "in_game"
    if not command:
        return "greeting" if state.get("new_session") else "silence"
    return "idle"


def route_after_branch(state: DialogState) -> Literal["recurse", "done"]:
    """Shared post-branch router for `waiting` and `paginate`.

    If the branch cleared its triggering state (pending/cursor) and
    produced no user-facing text, hand the same turn back to
    entry_router so the idle path can re-dispatch the user's command."""
    if state.get("last_bot_text"):
        return "done"
    return "recurse"


def route_after_idle(
    state: DialogState,
) -> Literal["reset", "exit", "help", "enter_game", "done"]:
    tool = state.get("_tool_call")  # type: ignore[typeddict-item]
    if tool == "reset_context":
        return "reset"
    if tool == "exit_skill":
        return "exit"
    if tool == "help":
        return "help"
    if tool == "enter_game":
        return "enter_game"
    return "done"


def route_after_classify(
    state: DialogState,
) -> Literal["resolve_challenge", "to_validate", "done"]:
    g = state.get("game")
    if g is None:
        return "done"  # exit_game tool fired
    if g.get("last_cheat") == "challenge":
        return "resolve_challenge"
    return "to_validate"


def route_after_validate(
    state: DialogState,
) -> Literal["bot_turn", "done"]:
    g = state.get("game")
    if g is None:
        return "done"
    if g.get("last_cheat"):
        return "done"
    return "bot_turn"


# ---------- assembly ----------


async def _entry_router_node(state: DialogState, config: RunnableConfig) -> dict:  # noqa: ARG001
    return {}


def _assemble() -> StateGraph:
    g = StateGraph(DialogState)
    g.add_node("turn_init", turn_init)
    g.add_node("entry_router", _entry_router_node)
    g.add_node("waiting", check_pending)
    g.add_node("paginate", pagination_continue_node)
    g.add_node("linking", linking_node)
    g.add_node("greeting", greeting_node)
    g.add_node("silence", silence_node)
    g.add_node("idle_llm", idle_llm)
    g.add_node("reset_context", reset_context_node)
    g.add_node("exit_skill", exit_skill_node)
    g.add_node("help", help_node)
    g.add_node("enter_game", enter_game_node)

    # Embedded words subgraph (same node functions as app.games.words).
    g.add_node("words_classify_player", words_classify_player_node)
    g.add_node("words_validate", words_validate_node)
    g.add_node("words_bot_turn", words_bot_turn_node)
    g.add_node("words_resolve_challenge", words_resolve_challenge_node)

    g.add_edge(START, "turn_init")
    g.add_edge("turn_init", "entry_router")

    g.add_conditional_edges(
        "entry_router",
        route_from_entry,
        {
            "waiting": "waiting",
            "paginate": "paginate",
            "linking": "linking",
            "in_game": "words_classify_player",
            "greeting": "greeting",
            "silence": "silence",
            "idle": "idle_llm",
        },
    )
    g.add_conditional_edges(
        "waiting",
        route_after_branch,
        {"recurse": "entry_router", "done": END},
    )
    g.add_conditional_edges(
        "paginate",
        route_after_branch,
        {"recurse": "entry_router", "done": END},
    )
    g.add_conditional_edges(
        "idle_llm",
        route_after_idle,
        {
            "reset": "reset_context",
            "exit": "exit_skill",
            "help": "help",
            "enter_game": "enter_game",
            "done": END,
        },
    )
    g.add_conditional_edges(
        "words_classify_player",
        route_after_classify,
        {
            "to_validate": "words_validate",
            "resolve_challenge": "words_resolve_challenge",
            "done": END,
        },
    )
    g.add_conditional_edges(
        "words_validate",
        route_after_validate,
        {"bot_turn": "words_bot_turn", "done": END},
    )
    for node in (
        "exit_skill",
        "help",
        "reset_context",
        "linking",
        "greeting",
        "silence",
        "enter_game",
        "words_bot_turn",
        "words_resolve_challenge",
    ):
        g.add_edge(node, END)
    return g


def build_parent_graph(checkpointer):
    """Compile the parent dialog graph with the supplied checkpointer.

    The checkpointer is normally `AsyncSqliteSaver(data/dialog.db)`
    built once at app startup. Tests pass `MemorySaver()` per-test for
    isolation.
    """
    return _assemble().compile(checkpointer=checkpointer)
