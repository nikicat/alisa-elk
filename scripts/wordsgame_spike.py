"""LangGraph spike for the words game ("игра в слова").

Standalone CLI that drives a tiny hierarchical FSM against the configured
LLM router. The point is to prove out the architecture (game state in
code, LLM as renderer, tool-call-as-transition, SQLite checkpointer) end
to end before touching `app/handler.py`.

Graph shape (one StateGraph, hierarchy expressed by routing on `game`):

      START
        │
        ▼
   entry_router ──── game is None ────► idle_llm ───┐
        │                                 │         │
        │ game set                tool: enter_game  │ no tool call
        │                                 │         ▼
        ▼                                 ▼        END (persona reply)
   words_validate                  words_intro ───► END (bot's first word)
        │
        ├─ cheat/exit ─► END (call-out or game-over message)
        │
        └─ ok ─► words_bot_turn ─► END (bot's next word)

State is checkpointed to data/langgraph.db; the CLI passes a thread_id
and resumes the same conversation across runs.

Usage:
    uv run python -m scripts.wordsgame_spike                # default thread
    uv run python -m scripts.wordsgame_spike --thread foo
    uv run python -m scripts.wordsgame_spike --reset        # wipe checkpoint
    uv run python -m scripts.wordsgame_spike --debug        # show routing
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, TypedDict, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app import persona
from app.config import get_settings

SKIP_LETTERS = frozenset({"ь", "ъ", "ы"})

# Bot picks its moves deterministically from this noun dictionary. The
# same dictionary is also queried to resolve player challenges (since the
# bot only plays words from the dictionary, an honest challenge always
# loses — but the LLM-driven classifier may still emit challenge_word on
# the player's behalf, and players who don't trust the bot need to be
# taught a lesson somehow).
DICTIONARY_PATH = Path(__file__).parent / "words_dict.tsv"

# Seeded by the CLI / tests; module-global so nodes can be plain
# state-only functions (LangGraph contract).
_RNG = random.Random()


@dataclass(frozen=True)
class WordDict:
    """Read view over words_dict.tsv."""

    by_letter: dict[str, tuple[str, ...]]
    all_words: frozenset[str]


_DICTIONARY_CACHE: WordDict | None = None


def load_dictionary() -> WordDict:
    """Lazy-load and cache the noun dictionary. Tests monkeypatch this."""
    global _DICTIONARY_CACHE
    if _DICTIONARY_CACHE is not None:
        return _DICTIONARY_CACHE
    by_letter: dict[str, list[str]] = {}
    all_words: set[str] = set()
    for raw_line in DICTIONARY_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 2 or parts[1].strip() != "n":
            continue
        word = parts[0].strip().lower()
        if not word:
            continue
        # The chain rule normalises ё→е nowhere, so keep words verbatim.
        by_letter.setdefault(word[0], []).append(word)
        all_words.add(word)
    _DICTIONARY_CACHE = WordDict(
        by_letter={k: tuple(sorted(v)) for k, v in by_letter.items()},
        all_words=frozenset(all_words),
    )
    return _DICTIONARY_CACHE


class GameState(TypedDict):
    used: list[str]
    required_letter: str | None
    last_cheat: str | None  # None | "wrong_letter" | "repeat"


class DialogState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_input: str | None
    game: GameState | None
    last_bot_text: str | None


# ---------------------------------------------------------------- tools


@tool
def enter_game(name: Literal["words"]) -> str:
    """Вызови, когда пользователь предлагает поиграть в словесную игру:
    «давай поиграем в слова», «сыграем в слова», «игра в слова».
    Передавай name='words'."""
    return f"entering {name}"


@tool
def exit_game() -> str:
    """Вызови, когда пользователь хочет выйти из текущей игры:
    «хватит играть», «выйдем из игры», «давай заканчивать игру»."""
    return "exited"


@tool
def challenge_word() -> str:
    """Вызови, когда игрок сомневается в твоём предыдущем слове или
    просит проверить его: «сомневаюсь», «такого слова нет»,
    «проверь своё слово», «это не слово», «выдумал». После вызова
    мы проверяем твоё последнее слово по словарю и объявляем победителя."""
    return "challenged"


IDLE_TOOLS = [enter_game, exit_game]
IN_GAME_TOOLS = [challenge_word, exit_game]


# ---------------------------------------------------------------- helpers


def required_start(word: str) -> str | None:
    for ch in reversed(word.lower().strip()):
        if "а" <= ch <= "я" or ch == "ё":
            if ch in SKIP_LETTERS:
                continue
            return ch
    return None


def extract_word(text: str) -> str | None:
    matches = re.findall(r"[а-яё]{2,}", text.lower())
    return matches[-1] if matches else None


def make_llm(*, temperature: float, max_tokens: int) -> ChatOpenAI:
    s = get_settings()
    base = s.LLM_BASE_URL.rstrip("/")
    # ChatOpenAI wants the bare /v1 base; strip a trailing /chat/completions
    # if someone set the full URL in .env (we accept both like OpenAIRouterClient).
    if base.endswith("/chat/completions"):
        base = base[: -len("/chat/completions")]
    return ChatOpenAI(
        base_url=base,
        api_key=s.LLM_API_KEY,
        model=s.LLM_MODEL,
        temperature=temperature,
        max_tokens=max_tokens,  # pyright: ignore[reportCallIssue]
    )


# ---------------------------------------------------------------- nodes


def idle_llm(state: DialogState) -> dict:
    """Persona reply with enter_game/exit_game tools bound."""
    llm = make_llm(temperature=0.6, max_tokens=160).bind_tools(IDLE_TOOLS)
    user = state.get("user_input") or ""
    msgs: list[BaseMessage] = [
        SystemMessage(persona.SYSTEM_PROMPT),
        *state.get("messages", []),
        HumanMessage(user),
    ]
    reply: AIMessage = llm.invoke(msgs)
    update: dict = {
        "messages": [HumanMessage(user), reply],
        "user_input": None,
    }
    tool_calls = getattr(reply, "tool_calls", None) or []
    if tool_calls:
        tc = tool_calls[0]
        if tc["name"] == "enter_game":
            update["game"] = GameState(used=[], required_letter=None, last_cheat=None)
            # words_intro produces the user-visible text on this same turn.
            update["last_bot_text"] = None
        elif tc["name"] == "exit_game":
            # We weren't in a game — be polite, don't crash.
            update["last_bot_text"] = "Сейчас никакой игры не идёт."
    else:
        update["last_bot_text"] = reply.content or "(тишина)"
    return update


def words_intro(state: DialogState) -> dict:
    """Bot's first word — picked uniformly from the dictionary."""
    dictionary = load_dictionary()
    pool = sorted(dictionary.all_words)
    word = _RNG.choice(pool)
    return {
        "game": GameState(
            used=[word], required_letter=required_start(word), last_cheat=None
        ),
        "messages": [AIMessage(word)],
        "last_bot_text": f"Поехали. Моё слово: {word}.",
    }


def words_classify_player(state: DialogState) -> dict:
    """LLM-driven intent classifier.

    Looks at the player's message with [challenge_word, exit_game] tools
    bound. If a tool is emitted, that becomes the transition; otherwise we
    pass through and let `words_validate` treat the message as a normal
    word move. The classifier is the only LLM call inside an active game.
    """
    raw = (state.get("user_input") or "").strip()
    if not raw:
        return {}  # nothing to classify
    llm = make_llm(temperature=0.0, max_tokens=24).bind_tools(IN_GAME_TOOLS)
    sys_msg = SystemMessage(
        "Идёт игра «в слова». Сообщение ниже — реплика собеседника. "
        "Если он сомневается в твоём предыдущем слове или просит проверить "
        "его — вызови challenge_word. Если он хочет выйти из игры или "
        "закончить — вызови exit_game. Иначе считай, что он сделал свой "
        "ход в игре, и ничего не вызывай — просто ответь любым словом."
    )
    reply = llm.invoke([sys_msg, HumanMessage(raw)])
    tool_calls = getattr(reply, "tool_calls", None) or []
    name = tool_calls[0].get("name") if tool_calls else None
    game = state.get("game")
    if name == "challenge_word" and game is not None:
        return {"game": {**game, "last_cheat": "challenge"}}
    if name == "exit_game":
        return {
            "game": None,
            "user_input": None,
            "last_bot_text": "Хорошо, заканчиваем игру. Возвращаемся к беседе.",
        }
    return {}  # pass-through to words_validate


def words_resolve_challenge(state: DialogState) -> dict:
    """Look up the bot's last word in the dictionary; declare a winner."""
    game = state.get("game")
    assert game is not None
    used = game["used"]
    if not used:
        return {
            "game": {**game, "last_cheat": None},
            "user_input": None,
            "last_bot_text": "А что проверять? Я ещё ничего не назвал.",
        }
    bot_word = used[-1]
    dictionary = load_dictionary()
    if bot_word in dictionary.all_words:
        return {
            "game": None,
            "user_input": None,
            "last_bot_text": (
                f"Слово «{bot_word}» — настоящее, проверено словарём. "
                "Ты ошибся, лес тебе не верит. Я выиграл."
            ),
        }
    return {
        "game": None,
        "user_input": None,
        "last_bot_text": (
            f"Поймал! Слова «{bot_word}» в моём словаре нет. Твоя победа."
        ),
    }


def words_validate(state: DialogState) -> dict:
    """Pure chain-rule logic. Exit and challenge intents are handled
    upstream by `words_classify_player` via tool calls."""
    game = state["game"]
    assert game is not None
    raw = (state.get("user_input") or "").strip().lower()
    user_word = extract_word(raw)
    if user_word is None:
        # Mark as a "cheat" so route_after_validate ends the turn here
        # instead of falling through to words_bot_turn (which would
        # overwrite our message with a fresh bot word).
        return {
            "game": {**game, "last_cheat": "no_word"},
            "user_input": None,
            "last_bot_text": "Я не расслышал слова. Назови ещё раз.",
        }
    if user_word in game["used"]:
        return {
            "game": {**game, "last_cheat": "repeat"},
            "user_input": None,
            "last_bot_text": (
                f"Слово «{user_word}» уже было. Назови другое на «{game['required_letter']}»."
            ),
        }
    expected = game["required_letter"]
    if expected and not user_word.startswith(expected):
        return {
            "game": {**game, "last_cheat": "wrong_letter"},
            "user_input": None,
            "last_bot_text": (
                f"Слово должно начинаться на «{expected}». Попробуй ещё раз."
            ),
        }
    # Valid player move — record and pass turn to bot.
    return {
        "game": GameState(
            used=game["used"] + [user_word],
            required_letter=required_start(user_word),
            last_cheat=None,
        ),
        "messages": [HumanMessage(user_word)],
        "user_input": None,
    }


def words_bot_turn(state: DialogState) -> dict:
    """Deterministic bot move — pick any unused dictionary word starting
    with the required letter. If none remain, surrender (player wins)."""
    game = state["game"]
    assert game is not None and game["required_letter"] is not None
    dictionary = load_dictionary()
    used: set[str] = set(game["used"])
    candidates = [
        w for w in dictionary.by_letter.get(game["required_letter"], ())
        if w not in used
    ]
    if not candidates:
        return {
            "game": None,
            "last_bot_text": (
                f"Сдаюсь — слов на букву «{game['required_letter']}» "
                "у меня больше нет. Твоя победа."
            ),
        }
    word = _RNG.choice(candidates)
    return {
        "game": GameState(
            used=game["used"] + [word],
            required_letter=required_start(word),
            last_cheat=None,
        ),
        "messages": [AIMessage(word)],
        "last_bot_text": word,
    }


# ---------------------------------------------------------------- routing


def route_entry(state: DialogState) -> Literal["in_game", "idle"]:
    return "in_game" if state.get("game") else "idle"


def route_after_idle(state: DialogState) -> Literal["enter_game", "done"]:
    g = state.get("game")
    # game was just set by idle_llm AND we haven't picked a first word yet
    if g is not None and not g["used"]:
        return "enter_game"
    return "done"


def route_after_validate(state: DialogState) -> Literal["bot_turn", "done"]:
    g = state.get("game")
    if g is None:
        return "done"  # exited
    if g.get("last_cheat"):
        return "done"  # called out — wait for fresh user word next turn
    return "bot_turn"


def route_after_classify(
    state: DialogState,
) -> Literal["resolve_challenge", "to_validate", "done"]:
    g = state.get("game")
    if g is None:
        return "done"  # exit_game tool fired
    if g.get("last_cheat") == "challenge":
        return "resolve_challenge"
    return "to_validate"


# ---------------------------------------------------------------- graph


def _entry_router_node(state: DialogState) -> dict:  # noqa: ARG001
    return {}  # pass-through; routing decision happens on the edge


def _assemble_graph() -> StateGraph:
    """Build the topology — nodes and edges, not yet compiled."""
    g = StateGraph(DialogState)
    g.add_node("entry_router", _entry_router_node)
    g.add_node("idle_llm", idle_llm)
    g.add_node("words_intro", words_intro)
    g.add_node("words_classify_player", words_classify_player)
    g.add_node("words_validate", words_validate)
    g.add_node("words_bot_turn", words_bot_turn)
    g.add_node("words_resolve_challenge", words_resolve_challenge)

    g.add_edge(START, "entry_router")
    g.add_conditional_edges(
        "entry_router",
        route_entry,
        {"in_game": "words_classify_player", "idle": "idle_llm"},
    )
    g.add_conditional_edges(
        "idle_llm",
        route_after_idle,
        {"enter_game": "words_intro", "done": END},
    )
    g.add_edge("words_intro", END)
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
    g.add_edge("words_bot_turn", END)
    g.add_edge("words_resolve_challenge", END)
    return g


def build_graph(checkpointer):
    """Compile with the given checkpointer (CLI / production use)."""
    return _assemble_graph().compile(checkpointer=checkpointer)


def build_for_studio():
    """Entrypoint for `langgraph dev` — the dev runtime supplies its own
    persistence layer, so we compile without a checkpointer."""
    return _assemble_graph().compile()


# ---------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="LangGraph spike: words-game FSM driven from a CLI."
    )
    parser.add_argument("--thread", default="cli", help="checkpoint thread id")
    parser.add_argument(
        "--reset", action="store_true", help="wipe the checkpoint DB before starting"
    )
    parser.add_argument("--db", default="data/langgraph.db", help="checkpoint file path")
    parser.add_argument("--debug", action="store_true", help="print routing decisions")
    args = parser.parse_args(argv)

    settings = get_settings()
    if not settings.LLM_API_KEY or settings.LLM_API_KEY in {"sk-dev", "sk-replace-me"}:
        print("error: LLM_API_KEY missing or placeholder. Set it in .env.", file=sys.stderr)
        return 2

    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if args.reset and db_path.exists():
        db_path.unlink()

    print(f"model  : {settings.LLM_MODEL}")
    print(f"thread : {args.thread}")
    print(f"db     : {db_path}")
    print("type Ctrl-D or 'quit' to exit.\n")

    with SqliteSaver.from_conn_string(str(db_path)) as cp:
        graph = build_graph(cp)
        if args.debug:
            print(graph.get_graph().draw_ascii())
            print()
        config: RunnableConfig = {"configurable": {"thread_id": args.thread}}

        # Optional greeting — only print if it's a brand new thread.
        existing = graph.get_state(config)
        if not existing.values:
            print("Лось: Лес слушает. Задавай вопрос.")

        while True:
            try:
                user = input("Ты:  ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not user:
                continue
            if user.lower() in {"quit", "exit", ":q"}:
                break
            # LangGraph merges partial updates into the checkpointed state,
            # but pyright wants the full TypedDict shape here.
            result = graph.invoke(cast(DialogState, {"user_input": user}), config=config)
            text = result.get("last_bot_text") or "(тишина)"
            print(f"Лось: {text}")
            if args.debug:
                g = result.get("game")
                if g:
                    print(f"  [game] used={g['used']} →{g['required_letter']} cheat={g['last_cheat']}")
                else:
                    print("  [game] None")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
