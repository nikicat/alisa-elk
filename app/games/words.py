"""Words-game ("игра в слова") subskill — LangGraph dialog FSM.

A hierarchical state machine that handles both the idle persona chat
and an in-game subgraph. Bot moves are deterministic picks from
`words_dict.tsv` (no LLM cost). The only in-game LLM call is the
classifier in `words_classify_player`, which both routes the player's
intent and extracts the candidate noun when applicable.

Bound classifier tools: noun_attempt(word), not_a_noun, challenge_word,
exit_game. On noun_attempt, the LLM hands back the extracted word; code
downstream only runs the chain-rule check (first letter + dedup).

Graph shape (one StateGraph; hierarchy is routing on `state.game`):

      START
        │
        ▼
   entry_router ─── game=None ──► idle_llm ──► words_intro ──► END
        │                            │                          (bot's first word)
        │                            └──► END (plain persona reply)
        │
        └─── game set ───► words_classify_player
                                │
                                ├─► resolve_challenge ──► END  (challenge → win/lose)
                                ├─► END  (exit_game / not_a_noun; scolded)
                                └─► words_validate  (noun_attempt → chain check)
                                         │
                                         ├─► END  (wrong_letter / repeat)
                                         └─► words_bot_turn ──► END

The module is runnable standalone for manual testing:

    uv run python -m app.games.words                 # default thread
    uv run python -m app.games.words --thread foo
    uv run python -m app.games.words --reset         # wipe checkpoint
    uv run python -m app.games.words --debug         # show routing

It also exposes `build_for_studio()` for `langgraph dev` and the
node/state/tool symbols that `app/handler.py` will import once we wire
this into the main dispatch in step (3).
"""

from __future__ import annotations

import argparse
import contextlib
import random
import re
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, TypedDict, cast

import structlog
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app import persona
from app.config import get_settings

log = structlog.get_logger(__name__)

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
    # Per-turn marker read by route_after_classify / route_after_validate.
    # "challenge" routes to resolve_challenge; any other truthy value ends
    # the turn after the scold message is set.
    last_cheat: str | None  # None | "not_a_noun" | "wrong_letter" | "repeat" | "challenge"
    # Set by words_classify_player when the LLM fires noun_attempt with an
    # extracted candidate; consumed (and cleared) by words_validate.
    candidate_word: str | None


class DialogState(TypedDict, total=False):
    # Matches `app.dialog.state.DialogState`'s totality so the parent
    # graph can hand its state straight to these nodes without casts.
    # session_id is read by logging but not declared here — it lives on
    # the parent state, which is structurally a superset of this one.
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
def noun_attempt(word: str) -> str:
    """Вызови, когда реплика игрока — попытка назвать существительное
    в именительном падеже единственного числа. В аргумент `word`
    передай само слово в нижнем регистре, без знаков препинания
    («мой ответ — яблоко» → word="яблоко»). Если слово незнакомое
    или выдуманное, но выглядит как существительное (например,
    «вертля»), всё равно noun_attempt — словарь проверим сами."""
    return f"noun {word}"


@tool
def challenge_word() -> str:
    """Вызови ТОЛЬКО когда в реплике игрока есть явные слова сомнения
    в ТВОЁМ предыдущем слове: «сомневаюсь», «такого слова нет»,
    «проверь своё слово», «это не слово», «выдумал», «врёшь».
    Одно слово, даже выдуманное, — это noun_attempt, а НЕ challenge."""
    return "challenged"


@tool
def not_a_noun() -> str:
    """Вызови, когда реплика — НЕ существительное в именительном падеже
    единственного числа. Сюда подходит: другая часть речи (глагол
    «бегать», прилагательное «красивый», местоимение «той», «его»,
    «себя», предлог), любой косвенный падеж («книги», «столу»),
    множественное число («штаны», «ножницы»), имя собственное или
    географическое название («Москва», «Иван»), бессмыслица
    («иририри», «wow ???»), либо реплика без слова (тишина, эмоция
    без существительного)."""
    return "not a noun"


IDLE_TOOLS = [enter_game, exit_game]
IN_GAME_TOOLS = [noun_attempt, not_a_noun, challenge_word, exit_game]


# ---------------------------------------------------------------- helpers


def required_start(word: str) -> str | None:
    for ch in reversed(word.lower().strip()):
        if "а" <= ch <= "я" or ch == "ё":
            if ch in SKIP_LETTERS:
                continue
            return ch
    return None


@contextlib.contextmanager
def _elapsed_ms() -> Iterator[Callable[[], int]]:
    """Yield a callable that returns ms elapsed since the `with` was entered."""
    t0 = time.monotonic()
    yield lambda: int((time.monotonic() - t0) * 1000)


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


async def idle_llm(state: DialogState) -> dict:
    """Persona reply with enter_game/exit_game tools bound."""
    llm = make_llm(temperature=0.6, max_tokens=160).bind_tools(IDLE_TOOLS)
    user = state.get("user_input") or ""
    msgs: list[BaseMessage] = [
        SystemMessage(persona.SYSTEM_PROMPT),
        *state.get("messages", []),
        HumanMessage(user),
    ]
    reply: AIMessage = await llm.ainvoke(msgs)
    update: dict = {
        "messages": [HumanMessage(user), reply],
        "user_input": None,
    }
    tool_calls = getattr(reply, "tool_calls", None) or []
    if tool_calls:
        tc = tool_calls[0]
        if tc["name"] == "enter_game":
            update["game"] = GameState(
                used=[], required_letter=None, last_cheat=None, candidate_word=None
            )
            # words_intro produces the user-visible text on this same turn.
            update["last_bot_text"] = None
        elif tc["name"] == "exit_game":
            # We weren't in a game — be polite, don't crash.
            update["last_bot_text"] = "Сейчас никакой игры не идёт."
    else:
        update["last_bot_text"] = reply.content or "(тишина)"
    return update


async def words_intro(state: DialogState) -> dict:
    """Bot's first word — picked uniformly from the dictionary."""
    slog = log.bind(session_id=state.get("session_id"))
    dictionary = load_dictionary()
    pool = sorted(dictionary.all_words)
    word = _RNG.choice(pool)
    slog.info(
        "words_intro",
        bot_word=word,
        required_letter=required_start(word),
    )
    return {
        "game": GameState(
            used=[word],
            required_letter=required_start(word),
            last_cheat=None,
            candidate_word=None,
        ),
        "messages": [AIMessage(word)],
        "last_bot_text": f"Поехали. Моё слово: {word}.",
    }


_CLASSIFY_PROMPT_TEMPLATE = (
    "Идёт игра «в слова». Правила: каждый по очереди называет "
    "существительное в именительном падеже единственного числа, "
    "начинающееся на последнюю значимую букву предыдущего слова "
    "(ь, ъ, ы пропускаются). Слова не повторяются.\n\n"
    "Контекст хода:\n"
    "— твоё последнее слово: «{bot_word}»\n"
    "— требуемая первая буква следующего слова: «{required}»\n\n"
    "Реплика игрока приходит следующим сообщением. Определи, что он "
    "имеет в виду, и вызови ровно один инструмент: noun_attempt(word), "
    "not_a_noun, challenge_word или exit_game. Описание у каждого "
    "инструмента — следуй ему буквально."
)


def _not_a_noun_update(game: GameState) -> dict:
    required = game["required_letter"] or ""
    tail = f" Назови существительное на «{required}»." if required else ""
    return {
        "game": {**game, "last_cheat": "not_a_noun", "candidate_word": None},
        "user_input": None,
        "last_bot_text": (
            "Это не существительное в именительном падеже "
            f"единственного числа.{tail}"
        ),
    }


async def words_classify_player(state: DialogState) -> dict:
    """Single LLM call: classify the player's reply AND, if it's a
    noun attempt, extract the normalized candidate word.

    Bound tools: noun_attempt(word), not_a_noun, challenge_word, exit_game.
    The system prompt includes the bot's last word and the required first
    letter so the classifier can reason about challenges and rule context.
    """
    raw = (state.get("user_input") or "").strip()
    game = state.get("game")
    slog = log.bind(session_id=state.get("session_id"))
    if not raw or game is None:
        return {}
    bot_word = game["used"][-1] if game["used"] else "—"
    required = game["required_letter"] or "—"
    llm = make_llm(temperature=0.0, max_tokens=64).bind_tools(IN_GAME_TOOLS)
    sys_msg = SystemMessage(
        _CLASSIFY_PROMPT_TEMPLATE.format(bot_word=bot_word, required=required)
    )
    with _elapsed_ms() as ms:
        try:
            reply = await llm.ainvoke([sys_msg, HumanMessage(raw)])
        except Exception as exc:
            slog.warning("words_classify_error", error=str(exc), llm_ms=ms())
            # Treat LLM failure as "couldn't classify" — scold and stay.
            return _not_a_noun_update(game)
        tool_calls = getattr(reply, "tool_calls", None) or []
        name = tool_calls[0].get("name") if tool_calls else None
        args = tool_calls[0].get("args") if tool_calls else None
        slog.info(
            "words_classify",
            intent=name or "no_tool",
            args=args,
            response=reply.content or "",
            llm_ms=ms(),
        )

    if name == "exit_game":
        return {
            "game": None,
            "user_input": None,
            "last_bot_text": "Хорошо, заканчиваем игру. Возвращаемся к беседе.",
        }
    if name == "challenge_word":
        return {"game": {**game, "last_cheat": "challenge", "candidate_word": None}}
    if name == "noun_attempt":
        word = ((args or {}).get("word") or "").strip().lower()
        if word and re.fullmatch(r"[а-яё]+", word):
            return {
                "game": {**game, "last_cheat": None, "candidate_word": word},
                "user_input": None,
            }
        # LLM fired noun_attempt but didn't extract a clean cyrillic word.
        # Treat as if it had said not_a_noun.
        return _not_a_noun_update(game)
    # not_a_noun, unknown tool, or no tool at all — all scold and stay.
    return _not_a_noun_update(game)


async def words_resolve_challenge(state: DialogState) -> dict:
    """Look up the bot's last word in the dictionary; declare a winner."""
    game = state.get("game")
    assert game is not None
    slog = log.bind(session_id=state.get("session_id"))
    used = game["used"]
    if not used:
        slog.info("words_challenge", outcome="nothing_to_check")
        return {
            "game": {**game, "last_cheat": None},
            "user_input": None,
            "last_bot_text": "А что проверять? Я ещё ничего не назвал.",
        }
    bot_word = used[-1]
    dictionary = load_dictionary()
    if bot_word in dictionary.all_words:
        slog.info("words_challenge", outcome="bot_wins", bot_word=bot_word)
        return {
            "game": None,
            "user_input": None,
            "last_bot_text": (
                f"Слово «{bot_word}» — настоящее, проверено словарём. "
                "Ты ошибся, лес тебе не верит. Я выиграл."
            ),
        }
    slog.info("words_challenge", outcome="player_wins", bot_word=bot_word)
    return {
        "game": None,
        "user_input": None,
        "last_bot_text": (
            f"Поймал! Слова «{bot_word}» в моём словаре нет. Твоя победа."
        ),
    }


async def words_validate(state: DialogState) -> dict:
    """Pure chain-rule logic. Reads the candidate word that the classifier
    extracted via `noun_attempt(word=...)`; never re-extracts from the raw
    message. Exit / challenge / not-a-noun decisions live upstream."""
    game = state.get("game")
    assert game is not None
    slog = log.bind(session_id=state.get("session_id"))
    user_word = (game.get("candidate_word") or "").strip().lower()
    if not user_word:
        # Defensive: route_after_classify should only land us here with
        # a candidate set. Treat a missing candidate as not_a_noun.
        slog.info("words_validate", outcome="no_candidate")
        return _not_a_noun_update(game)
    if user_word in game["used"]:
        slog.info("words_validate", outcome="repeat", player_word=user_word)
        return {
            "game": {**game, "last_cheat": "repeat", "candidate_word": None},
            "user_input": None,
            "last_bot_text": (
                f"Слово «{user_word}» уже было. Назови другое на «{game['required_letter']}»."
            ),
        }
    expected = game["required_letter"]
    if expected and not user_word.startswith(expected):
        slog.info(
            "words_validate",
            outcome="wrong_letter",
            player_word=user_word,
            required=expected,
        )
        return {
            "game": {**game, "last_cheat": "wrong_letter", "candidate_word": None},
            "user_input": None,
            "last_bot_text": (
                f"Слово должно начинаться на «{expected}». Попробуй ещё раз."
            ),
        }
    slog.info("words_validate", outcome="accepted", player_word=user_word)
    # Valid player move — record and pass turn to bot.
    return {
        "game": GameState(
            used=game["used"] + [user_word],
            required_letter=required_start(user_word),
            last_cheat=None,
            candidate_word=None,
        ),
        "messages": [HumanMessage(user_word)],
        "user_input": None,
    }


async def words_bot_turn(state: DialogState) -> dict:
    """Deterministic bot move — pick any unused dictionary word starting
    with the required letter. If none remain, surrender (player wins)."""
    game = state.get("game")
    assert game is not None and game["required_letter"] is not None
    slog = log.bind(session_id=state.get("session_id"))
    dictionary = load_dictionary()
    used: set[str] = set(game["used"])
    candidates = [
        w
        for w in dictionary.by_letter.get(game["required_letter"], ())
        if w not in used
    ]
    if not candidates:
        slog.info(
            "words_bot_turn", outcome="surrender", required=game["required_letter"]
        )
        return {
            "game": None,
            "last_bot_text": (
                f"Сдаюсь — слов на букву «{game['required_letter']}» "
                "у меня больше нет. Твоя победа."
            ),
        }
    word = _RNG.choice(candidates)
    slog.info(
        "words_bot_turn",
        outcome="played",
        bot_word=word,
        required_letter=required_start(word),
    )
    return {
        "game": GameState(
            used=game["used"] + [word],
            required_letter=required_start(word),
            last_cheat=None,
            candidate_word=None,
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
    cheat = g.get("last_cheat")
    if cheat == "challenge":
        return "resolve_challenge"
    if cheat:
        return "done"  # classifier-set cheat (e.g. not_a_noun) — turn already rendered
    return "to_validate"


# ---------------------------------------------------------------- graph


async def _entry_router_node(state: DialogState) -> dict:
    del state
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


def build_for_handler(checkpointer):
    """Compile the standalone words graph using the parent handler's saver.

    Phase 1 of the LangGraph migration uses this for the standalone CLI
    and (eventually) any place that wants the spike graph wired to the
    same persistence layer as the parent dialog graph. The parent graph
    in `app/dialog/graph.py` does not embed this compiled subgraph — it
    composes the words nodes directly to keep state-merging trivial.
    """
    return _assemble_graph().compile(checkpointer=checkpointer)


# ---------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="LangGraph spike: words-game FSM driven from a CLI."
    )
    parser.add_argument("--thread", default="cli", help="checkpoint thread id")
    parser.add_argument(
        "--reset", action="store_true", help="wipe the checkpoint DB before starting"
    )
    parser.add_argument(
        "--db", default="data/langgraph.db", help="checkpoint file path"
    )
    parser.add_argument("--debug", action="store_true", help="print routing decisions")
    args = parser.parse_args(argv)

    settings = get_settings()
    if not settings.LLM_API_KEY or settings.LLM_API_KEY in {"sk-dev", "sk-replace-me"}:
        print(
            "error: LLM_API_KEY missing or placeholder. Set it in .env.",
            file=sys.stderr,
        )
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
            except EOFError, KeyboardInterrupt:
                print()
                break
            if not user:
                continue
            if user.lower() in {"quit", "exit", ":q"}:
                break
            # LangGraph merges partial updates into the checkpointed state,
            # but pyright wants the full TypedDict shape here.
            result = graph.invoke(
                cast(DialogState, {"user_input": user}), config=config
            )
            text = result.get("last_bot_text") or "(тишина)"
            print(f"Лось: {text}")
            if args.debug:
                g = result.get("game")
                if g:
                    print(
                        f"  [game] used={g['used']} →{g['required_letter']} cheat={g['last_cheat']}"
                    )
                else:
                    print("  [game] None")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
