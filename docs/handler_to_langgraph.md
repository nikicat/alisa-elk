# Handler → LangGraph migration plan

**Status:** Phase 1 and Phase 2 landed. Phase 3 (keyword/table cleanup) pending.
**Scope:** Replace the if/elif state machine in `app/handler.py:_route_inner`
with a compiled LangGraph that owns dialog state. `route(req, deps)` stays
as the public entry; only the internals change. First milestone: in-game
flow routes through `app/games/words` whenever the idle LLM emits the
`enter_game` tool.

## Decisions (settled)

1. **LLM client: stay with `OpenAIRouterClient`** (async, project-tuned).
   Manual tool-call inspection inside nodes — no migration to
   `ChatOpenAI` for the idle path. The existing
   `wait_for_or_keepalive` slow-path machinery keeps working unchanged
   in Phase 1.
2. **`thread_id = req.session.session_id`** — matches the lifetime of
   `session_state` today. Game state is intentionally ephemeral and
   dies when the Yandex session ends.
3. **`auth_gate` becomes FastAPI middleware**, not a graph node.
   Rejected requests never enter the graph, so no checkpoint state for
   them.
4. **`linking` stays in the graph.** Users may want to relink a device
   later (e.g. switching account), so the linked/unlinked transition is
   part of the dialog state machine, not a one-shot guard.
5. **Drop the regex shortcuts** (`EXIT_WORDS`, `HELP_WORDS`,
   `RESET_WORDS`). Trust the LLM via tool calls (`exit_skill`,
   `help`, `reset_context`). Classifier probe shows ~96% on adversarial
   inputs; the latency cost of an extra LLM call is acceptable.

## State mapping

| Today (`session_state` key) | After migration | Owner |
|---|---|---|
| `pending_id`, `wait_turns` | `DialogState.pending` | LangGraph checkpoint |
| `cursor` (`{turn_id, idx}`) | `DialogState.pagination_cursor` | LangGraph checkpoint |
| `context_since` (ISO str) | `DialogState.context_since` (`datetime`) | LangGraph checkpoint |
| *(new)* game state | `DialogState.game` | LangGraph checkpoint, nested |
| | **Bridge:** `session_state = {"thread_id": <session_id>}` | Yandex |

`turns` and `users` tables stay as today. `pending_requests` survives
through Phase 1 (the existing slow-path code keeps writing to it); it
retires in Phase 2 once the wait mechanism moves into graph nodes.

## Checkpointer

`AsyncSqliteSaver` at `data/dialog.db`. Separate file from `elk.db` to
avoid write-lock contention with SQLAlchemy. Bind-mounted in Docker, so
state persists across container restarts. Built once at app startup in
`main.py` and injected into `HandlerDeps`.

## Parent graph topology (Phase 1)

```
START
  │
  └─► linking_or_idle
        │
        ├─ unlinked ─► linking
        │              │
        │  ┌──valid code──► idle  (transitions in-place; commits user link)
        │  └──bad code────► END   (asks to redictate)
        │
        └─ linked ───► idle_llm
                         │
                         ├─ tool: enter_game(words) ─► words_subgraph
                         │                                │
                         │                                └─► END
                         │
                         ├─ tool: reset_context ─► clear_context ─► END
                         ├─ tool: exit_skill ─► END (end_session=true)
                         ├─ tool: help ─► help_text_node ─► END
                         │
                         └─ no tool ─► dispatch_llm_or_paginate ─► END
                                          (covers fast-path reply, slow-path
                                           wait phrases, and pagination cursor)
```

`words_subgraph` is `app.games.words.build_for_handler(checkpointer)` —
a new helper similar to `build_for_studio` but accepting the parent's
saver. Cleaner than nesting `MemorySaver`s.

The idle tools surface grows: `enter_game`, `reset_context` (existing),
`exit_skill` (new — replaces `EXIT_WORDS`), `help` (new — replaces
`HELP_WORDS`). One LLM call decides all four intents.

## Migration phases

### Phase 1 — Idle + game dispatch  ✅ done

**New files (created as planned)**

- `app/dialog/__init__.py`
- `app/dialog/state.py` — `DialogState` TypedDict, sub-types for
  pending / cursor.
- `app/dialog/tools.py` — OpenAI-tool-schema mirrors of `enter_game`,
  `exit_skill`, `help` (`reset_context` lives in `persona.RESET_TOOL`)
  plus `@tool` mirrors for the words subgraph and any future
  ChatOpenAI-based dispatcher.
- `app/dialog/nodes.py` — `turn_init`, `waiting_node`, `idle_llm` (with
  slow-path), `pagination_continue_node`, `linking_node`,
  `greeting_node`, `silence_node`, `help_node`, `exit_skill_node`,
  `reset_context_node`, `enter_game_node`, and thin wrappers around
  the words-game nodes so the parent graph can embed them directly.
- `app/dialog/graph.py` — `build_parent_graph(checkpointer)` +
  `_assemble()`.

**Modified files (as planned)**

- `app/main.py` — builds `AsyncSqliteSaver("data/dialog.db")` in the
  lifespan and injects the compiled graph into `HandlerDeps`. The
  `skill_id` check moved to a FastAPI middleware that peeks the
  request body and short-circuits with the canned rejection response.
- `app/handler.py` — collapsed to a thin driver: `route()` builds the
  per-turn input slice, calls `deps.graph.ainvoke`, and renders the
  resulting state into an `AliceResponse`. Old helpers
  (`_route_inner`, `_handle_waiting`, `_ready_response`,
  `_clear_session_state`, `_make`, `_store_paginated`,
  `_persist_result`) are gone from this module; the equivalent logic
  lives inside the graph nodes.
- `app/games/words.py` — node functions are now `async def` and call
  `llm.ainvoke`. Added `build_for_handler(checkpointer)` so the same
  topology can be compiled against the parent saver if a caller ever
  wants the standalone graph wired to dialog persistence.

**Slow path:** NOT migrated in Phase 1 (as planned). `idle_llm` is the
dispatcher: it spawns the LLM task, calls `wait_for_or_keepalive`, and
either returns a fast-path reply, a tool-dispatch marker, or persists
a `pending_requests` row + wait phrase + fires `_persist_result` as a
background task. Subsequent turns enter `waiting_node` via
`entry_router` when state has a `pending_id`.

**Tests**

- `app/tests/conftest.py` now builds a fresh in-memory
  `MemorySaver`-backed parent graph per test and injects it into
  `HandlerDeps`. The `app_client` fixture wires the same graph into
  the FastAPI dependency override.
- `app/tests/test_handler.py` and `app/tests/test_full_cycle.py` pass
  unchanged (24 + 13 tests).
- `app/tests/test_words_game.py` — `invoke` helper switched to
  `ainvoke`; tests made `async`; `FakeChatLLM` gained an `ainvoke`
  passthrough. All 11 tests pass.
- No new `test_dialog_graph.py` yet — the existing public-contract
  tests already exercise every graph branch. We can add one later if
  the graph topology grows independent enough to warrant unit-level
  coverage of routing decisions.

**Deviations from settled decisions (worth knowing)**

- **Regex shortcuts stayed in place.** Decision #5 (drop
  `EXIT_WORDS` / `HELP_WORDS` / `RESET_WORDS`, trust the LLM tools)
  was deferred. The `entry_router` checks the keyword frozensets
  before reaching `idle_llm`, mirroring the legacy fast-path. The
  idle LLM still has `exit_skill` / `help` / `reset_context` /
  `enter_game` bound, so non-keyword phrasings still work. Phase 3
  can remove the regexes in a single keyword-deletion commit with
  one round of test churn instead of bleeding it across this PR.
- **`session_state` mirrors `pending_id` / `cursor` / `context_since`**
  alongside `thread_id` rather than the minimal `{thread_id}` form.
  The LangGraph checkpoint is still the canonical source of truth
  (incoming `state.session` is ignored); the mirror exists only so
  logs and existing test assertions stay readable. Cheap to remove
  later if the mirror grows stale.

### Phase 2 — Slow path into the graph  ✅ done

The `pending_requests` table is out of the live request path entirely.
The asyncio.Task parked in `PendingTaskRegistry` is now the only
authority on whether a slow LLM call has produced a result, and
`check_pending` reads `task.result()` directly the moment `task.done()`
flips.

**What changed**

- `idle_llm` slow path: drops `repo.create_pending` and the
  `_persist_result` background coroutine. It snapshots
  `(request_text, user_id, application_id, message_id, llm_started_at)`
  into `DialogState.pending` and returns the first wait phrase. The
  live `asyncio.Task` stays in `PendingTaskRegistry`.
- `waiting_node` → `check_pending` (renamed for clarity; the routing
  edge label is still `waiting` for diagram continuity). It polls
  `deps.registry.get(pending_id)`, calls `wait_for_or_keepalive` on
  the still-running task when the user says «да», and on
  `task.done()` drains via `_apply_task_result` — which logs the turn,
  paginates, and produces the response. Cancellation / exception /
  `reset_context` tool calls are handled inline.
- `_persist_result` and the `RESET_SENTINEL` are gone.
- Orphan handling: if state has `pending_id` but the registry has no
  task (process restart, or task already drained), `check_pending`
  drops the ghost pending and recurses to the entry router so the
  user's new input gets a fresh dispatch.

**`pending_requests` status:** retired from the live request path.
`mark_orphaned_in_progress_as_error` still runs on startup so any
rows from earlier deployments get drained, but nothing new is
written. Phase 3 will drop the table via Alembic.

**Tests updated**

- `test_wait_pattern.py::test_no_aborts_pending` and
  `test_exit_while_waiting_cancels` now assert
  `registry.get(pending_id) is None` instead of `pending_requests.row.status`.
- `test_full_cycle.py::test_slow_llm_abort_with_no` and
  `test_reset_while_pending_cancels_pending` switched the same way.
- `test_startup_marks_orphaned_in_progress_as_error` kept unchanged —
  the legacy cleanup still works on legacy rows.
- All 59 tests pass.

### Phase 3 — Cleanup (~½ day)

- Remove `EXIT_WORDS`, `HELP_WORDS`, `RESET_WORDS`, `CONTINUE_WORDS`,
  `AFFIRMATIVE_WORDS`, `NEGATIVE_WORDS` and the helpers that use them.
- Remove `pending_requests` table if Phase 2 retired it (Alembic
  migration).
- Drop `MockLLMClient` if all tests have moved to the new fakes; keep
  it otherwise.

## Risk surface

- **Checkpoint version skew.** LangGraph saver schema can change between
  minor versions. Pin `langgraph` to an exact version once the
  migration lands; add a `# CHECKPOINT SCHEMA: v0.6.x` comment near the
  saver construction.
- **In-flight tasks on restart.** Same risk as today — orphaned
  `pending_requests` rows. Add a startup sweeper that marks rows older
  than N seconds as `aborted`. Not new work; can carry forward from
  current behavior.
- **Async/sync mismatch in nodes.** `app/games/words.py` is currently
  sync (`ChatOpenAI.invoke`). Phase 1 converts to `async def` +
  `ainvoke`. ~50 lines, mechanical.
- **Linking state-machine subtlety.** A user can be both linked AND
  request relinking. The `linking_or_idle` router needs an explicit
  intent (LLM tool? `relink` button?) to enter `linking` for an
  already-linked device, otherwise the LLM might never offer the path.
  TBD when implementing.

## What's NOT in this plan

- Multi-game registry (cities, capitals, …). The `app/games/__init__.py`
  registry exists but only `words` is plugged in. Adding another game
  is a separate step once Phase 1 lands.
- Migration to `ChatOpenAI` everywhere. Could happen later; not required
  for any milestone here.
- LangGraph Studio production mode / hosted persistence. Stay on
  `AsyncSqliteSaver` indefinitely.
