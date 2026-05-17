"""Behaviour tests for app/games/words.py — the words-game FSM.

We patch `make_llm` with a scripted FakeChatLLM that returns canned
AIMessages, so these run offline. `_RNG` is seeded and the dictionary is
swapped for a small in-memory fixture so bot moves are predictable.

LLM call budget per turn under the current design:
  - Idle (no game): 1 call to idle_llm.
  - In-game: 1 call to words_classify_player. Bot moves are deterministic
    dictionary picks — no LLM call.

Failure modes exercised:
  - player picks a wrong-letter word
  - player repeats an already-used word
  - player types gibberish (no Russian word) — regression for the bug
    where the chain silently advanced past a "didn't hear you" message
  - player intent classified as exit_game (tool call)
  - player intent classified as challenge_word, bot's word IS in dict
    → player loses
  - player intent classified as challenge_word, bot's word is NOT in
    dict → player wins
  - bot exhausts its dictionary pool for the required letter → surrenders
  - idle persona reply with no tool call ends the turn
"""

from __future__ import annotations

import random
from typing import Any, cast

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver

from app.games import words as spike


class FakeChatLLM:
    """Mimics the slice of ChatOpenAI the spike uses: bind_tools + invoke."""

    def __init__(self, responses: list[AIMessage]) -> None:
        self._responses = list(responses)
        self.calls: list[list[Any]] = []

    def bind_tools(self, _tools: list[Any]) -> FakeChatLLM:
        return self

    def invoke(self, messages: list[Any]) -> AIMessage:
        self.calls.append(messages)
        if not self._responses:
            raise AssertionError(
                "FakeChatLLM ran out of scripted responses — "
                "graph took an unexpected LLM-calling branch."
            )
        return self._responses.pop(0)

    async def ainvoke(self, messages: list[Any]) -> AIMessage:
        return self.invoke(messages)


@pytest.fixture
def fake_llm(monkeypatch):
    """Install a FakeChatLLM with a scripted list of AIMessage responses."""

    def install(responses: list[AIMessage]) -> FakeChatLLM:
        fake = FakeChatLLM(responses)
        monkeypatch.setattr(spike, "make_llm", lambda **_: fake)
        return fake

    return install


@pytest.fixture
def small_dict(monkeypatch):
    """Swap the dictionary for a small predictable set.

    Keys cover the chains exercised below. `арбуз` is intentionally
    excluded so we can test the challenge-wins-on-fake-word path.
    """
    fixture = spike.WordDict(
        by_letter={
            "з": ("зонт", "змея"),
            "о": ("олень",),
            "ь": (),  # never picked — required_start skips
            "м": ("море",),
            "е": ("ель",),
        },
        all_words=frozenset({"зонт", "змея", "олень", "море", "ель"}),
    )
    monkeypatch.setattr(spike, "_DICTIONARY_CACHE", fixture)
    return fixture


@pytest.fixture(autouse=True)
def _seeded_rng(monkeypatch):
    """Make `_RNG.choice` deterministic across tests."""
    monkeypatch.setattr(spike, "_RNG", random.Random(0))


@pytest.fixture
def graph():
    return spike.build_graph(MemorySaver())


def cfg(thread: str = "t") -> dict:
    return {"configurable": {"thread_id": thread}}


async def invoke(graph: Any, user_input: str, thread: str = "t") -> dict:
    return await graph.ainvoke(
        cast(spike.DialogState, {"user_input": user_input}), config=cfg(thread)
    )


def enter_game_call() -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "enter_game",
                "args": {"name": "words"},
                "id": "tc-enter",
            }
        ],
    )


def exit_game_call() -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": "exit_game", "args": {}, "id": "tc-exit"}],
    )


def challenge_call() -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": "challenge_word", "args": {}, "id": "tc-challenge"}],
    )


def no_tool_reply(text: str = "ход") -> AIMessage:
    """Classifier output meaning 'not a tool, treat as a normal move'."""
    return AIMessage(content=text)


# --------------------------------------------------------------- happy path


async def test_enter_game_picks_first_word_from_dictionary(fake_llm, small_dict, graph):
    fake_llm([enter_game_call()])
    result = await invoke(graph, "давай поиграем в слова")
    assert result["game"] is not None
    assert result["game"]["used"]  # one bot word
    bot_first = result["game"]["used"][0]
    assert bot_first in small_dict.all_words
    assert "Поехали" in result["last_bot_text"]


def _state_game(graph) -> dict:
    state = graph.get_state(cfg()).values
    return state["game"]


async def test_valid_player_move_advances_chain(fake_llm, graph):
    """Player word valid → classifier says no-tool → validate appends →
    bot picks deterministically from dictionary. Uses the real 554-word
    dictionary, which has follow-ups for every reachable letter."""
    fake_llm([enter_game_call(), no_tool_reply()])
    await invoke(graph, "давай поиграем")
    bot_first = _state_game(graph)["used"][0]
    required = spike.required_start(bot_first)
    assert required is not None
    dictionary = spike.load_dictionary()
    candidates = [w for w in dictionary.by_letter.get(required, ()) if w != bot_first]
    if not candidates:
        pytest.skip(f"no follow-up in dict for «{required}» (bot picked {bot_first!r})")
    player_word = candidates[0]
    next_required = spike.required_start(player_word)
    result = await invoke(graph, player_word)
    used = result["game"]["used"]
    assert used[0] == bot_first
    assert used[1] == player_word
    assert len(used) == 3
    assert used[2].startswith(next_required or "")
    assert result["game"]["last_cheat"] is None


@pytest.mark.usefixtures("small_dict")
async def test_player_wrong_letter_does_not_advance(fake_llm, graph):
    fake_llm(
        [
            enter_game_call(),
            no_tool_reply(),  # classifier: no tool, pass to validate
        ]
    )
    await invoke(graph, "давай")
    # Bot's first word starts with some letter L; player must answer with
    # L. We send something that almost certainly won't match (single char
    # not in any required letter we care about).
    bot_first = _state_game(graph)["used"][0]
    required = spike.required_start(bot_first)
    bad_letter = "ы" if required != "ы" else "э"
    # Find a Cyrillic word starting with `bad_letter` (it's not in the
    # small_dict, but extract_word doesn't care; validate compares letters).
    fake_llm_again_msg = f"{bad_letter}нечто"
    result = await invoke(graph, fake_llm_again_msg)
    assert result["game"]["last_cheat"] == "wrong_letter"
    assert fake_llm_again_msg not in result["game"]["used"]
    assert f"начинаться на «{required}»" in result["last_bot_text"]


@pytest.mark.usefixtures("small_dict")
async def test_player_repeat_does_not_advance(fake_llm, graph):
    fake_llm(
        [
            enter_game_call(),
            no_tool_reply(),  # classifier when player repeats
        ]
    )
    await invoke(graph, "давай")
    bot_first = _state_game(graph)["used"][0]
    result = await invoke(graph, bot_first)  # echo the bot's word
    assert result["game"]["last_cheat"] == "repeat"
    assert result["game"]["used"] == [bot_first]  # unchanged
    assert "уже было" in result["last_bot_text"]


@pytest.mark.usefixtures("small_dict")
async def test_gibberish_does_not_advance_chain(fake_llm, graph):
    """Regression: gibberish must terminate at "не расслышал"."""
    fake_llm(
        [
            enter_game_call(),
            no_tool_reply(),
        ]
    )
    await invoke(graph, "давай")
    used_before = list(_state_game(graph)["used"])
    result = await invoke(graph, "wow ???")
    assert result["game"]["last_cheat"] == "no_word"
    assert result["game"]["used"] == used_before
    assert "не расслышал" in result["last_bot_text"]


# --------------------------------------------------------------- tool intents


@pytest.mark.usefixtures("small_dict")
async def test_exit_via_classifier_ends_game(fake_llm, graph):
    fake_llm(
        [
            enter_game_call(),
            exit_game_call(),  # classifier emits exit_game
        ]
    )
    await invoke(graph, "давай")
    result = await invoke(graph, "хватит играть")
    assert result["game"] is None
    assert "заканчиваем" in result["last_bot_text"].lower()


@pytest.mark.usefixtures("small_dict")
async def test_challenge_loses_when_bot_word_is_in_dictionary(fake_llm, graph):
    """Bot always picks dictionary words → an honest challenge always loses."""
    fake_llm(
        [
            enter_game_call(),
            challenge_call(),  # classifier emits challenge_word
        ]
    )
    await invoke(graph, "давай")
    bot_first = _state_game(graph)["used"][0]
    result = await invoke(graph, "сомневаюсь, такого слова нет")
    assert result["game"] is None  # game ended on resolve
    assert "Я выиграл" in result["last_bot_text"]
    assert bot_first in result["last_bot_text"]


async def test_challenge_wins_when_bot_word_is_not_in_dictionary(
    fake_llm, monkeypatch, graph
):
    """If the bot somehow plays a word not in the dictionary (e.g., the
    dictionary shrinks mid-game), a player challenge wins."""
    # Game starts with a normal dictionary, then we remove the bot's word
    # from the dictionary before the player challenges.
    fixture = spike.WordDict(
        by_letter={"з": ("зонт",)},
        all_words=frozenset({"зонт"}),
    )
    monkeypatch.setattr(spike, "_DICTIONARY_CACHE", fixture)
    fake_llm(
        [
            enter_game_call(),
            challenge_call(),
        ]
    )
    await invoke(graph, "давай")
    # Strip the bot's word from the dictionary
    bot_first = _state_game(graph)["used"][0]
    pruned = spike.WordDict(
        by_letter={
            k: tuple(w for w in v if w != bot_first)
            for k, v in fixture.by_letter.items()
        },
        all_words=fixture.all_words - {bot_first},
    )
    monkeypatch.setattr(spike, "_DICTIONARY_CACHE", pruned)
    result = await invoke(graph, "сомневаюсь")
    assert result["game"] is None
    assert "Твоя победа" in result["last_bot_text"]
    assert bot_first in result["last_bot_text"]


# --------------------------------------------------------------- bot exhaustion


async def test_bot_surrenders_when_no_words_left_for_letter(
    fake_llm, monkeypatch, graph
):
    """Player wins by exhausting bot's pool for the required letter."""
    # Tiny dictionary: only one 'м' word and one 'е' word. Player will
    # answer with a word ending in а letter where the bot has nothing.
    fixture = spike.WordDict(
        by_letter={
            "м": ("море",),  # bot's intro will be "море"
            "е": ("ель",),
            "я": (),
        },
        all_words=frozenset({"море", "ель"}),
    )
    monkeypatch.setattr(spike, "_DICTIONARY_CACHE", fixture)
    monkeypatch.setattr(spike, "_RNG", random.Random(0))

    fake_llm(
        [
            enter_game_call(),
            no_tool_reply(),  # player's word goes through validate
        ]
    )
    await invoke(graph, "давай")
    # Bot picked "море" or "ель"; force a chain dead-end by playing
    # something ending in a letter not in fixture.
    bot_first = _state_game(graph)["used"][0]
    # The player's word must start with required_start(bot_first) and
    # end with a letter that has no dictionary entries.
    # bot_first is "море" (ends 'е') or "ель" (ends 'ь'→'л' after skip).
    if bot_first == "море":
        player_word = "ехидна"  # starts 'е' (matches), ends 'а' → empty pool
    else:
        player_word = "лиса"  # starts 'л' (matches), ends 'а' → empty pool
    result = await invoke(graph, player_word)
    assert result["game"] is None
    assert "Твоя победа" in result["last_bot_text"]


# --------------------------------------------------------------- idle paths


async def test_idle_plain_reply_no_tool_call(fake_llm, graph):
    fake_llm([AIMessage(content="Лес шумит, я слушаю.")])
    result = await invoke(graph, "что такое осень")
    assert result.get("game") is None
    assert result["last_bot_text"] == "Лес шумит, я слушаю."


async def test_exit_game_tool_call_outside_game_is_graceful(fake_llm, graph):
    fake_llm([exit_game_call()])
    result = await invoke(graph, "хватит")
    assert result.get("game") is None
    assert "не идёт" in result["last_bot_text"]
