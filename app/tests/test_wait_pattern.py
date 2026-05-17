"""Wait-pattern tests: slow LLM keep-alive, да/нет flow, escalation, abort."""

import time

import pytest

from app import config as cfg_mod
from app import persona, repo
from app.tests.mock_alice import AliceSession
from app.tests.mock_llm import MockLLMClient

FAST_CONFIG = {
    "llm": {
        "temperature": 0.6,
        "max_tokens": 280,
        "first_wait_timeout_s": 0.1,
        "subsequent_wait_timeout_s": 0.1,
        "max_wait_turns": 3,
    },
    "persona": {
        "wait_phrases": [
            "Надо подумать, подождёшь?",
            "Очень сложный вопрос, подождёшь ещё?",
            "Совсем заковыристо, потерпишь?",
        ],
        "give_up_phrase": "Лес сегодня молчит. Спроси чуть позже или о другом.",
    },
    "memory": {"recent_turns": 4},
    "pagination": {"chunk_chars": 1000},
}


@pytest.fixture(autouse=True)
def _fast_cfg(monkeypatch):
    monkeypatch.setattr(cfg_mod, "get_config", lambda: FAST_CONFIG)


def _setup_linked_user(db, alice: AliceSession) -> int:
    user = repo.create_user(db, display_name="Test")
    db.commit()
    code = repo.create_link_code(db, user.id)
    db.commit()
    repo.consume_link_code(db, code, alice.application_id)
    db.commit()
    return user.id


def _yield(seconds: float = 0.05) -> None:
    """Give background tasks a chance to run.

    The TestClient runs in a thread; the FastAPI event loop runs there too,
    so a small real sleep here lets background asyncio tasks make progress.
    """
    time.sleep(seconds)


def test_fast_llm_no_wait_phrase(alice, db, mock_llm: MockLLMClient):
    _setup_linked_user(db, alice)
    mock_llm.respond_instantly("Быстрый ответ.")
    alice.say("вопрос")
    alice.assert_text_equals("Быстрый ответ.")
    alice.assert_no_pending()


def test_slow_llm_first_wait_phrase(alice, db, mock_llm: MockLLMClient):
    _setup_linked_user(db, alice)
    mock_llm.respond_after(0.5, "Медленный ответ.")
    alice.say("вопрос")
    alice.assert_text_equals(FAST_CONFIG["persona"]["wait_phrases"][0])
    alice.assert_has_pending()


def test_yes_returns_ready_answer(alice, db, mock_llm: MockLLMClient):
    _setup_linked_user(db, alice)
    mock_llm.respond_after(0.2, "Лесная мудрость.")
    mock_llm.call_tool_instantly("wait_more")  # classifier on "да"
    alice.say("вопрос")
    alice.assert_has_pending()
    # Let the LLM finish; then user agrees to wait.
    _yield(0.4)
    alice.say("да")
    alice.assert_text_equals("Лесная мудрость.")
    alice.assert_no_pending()


def test_yes_still_in_flight_escalates(alice, db, mock_llm: MockLLMClient):
    _setup_linked_user(db, alice)
    # LLM never returns within test (long delay).
    mock_llm.respond_after(60.0, "irrelevant")
    mock_llm.call_tool_instantly("wait_more")  # classifier on "да"
    alice.say("вопрос")
    alice.assert_text_equals(FAST_CONFIG["persona"]["wait_phrases"][0])
    alice.say("да")
    alice.assert_text_equals(FAST_CONFIG["persona"]["wait_phrases"][1])
    alice.assert_has_pending()


def test_no_aborts_pending(alice, db, mock_llm: MockLLMClient, registry):
    _setup_linked_user(db, alice)
    mock_llm.respond_after(60.0, "irrelevant")
    mock_llm.call_tool_instantly("cancel_pending")  # classifier on "нет"
    alice.say("вопрос")
    pending_id = alice.last_session_state["pending_id"]
    alice.say("нет")
    alice.assert_text_equals(persona.ABORT_OK)
    alice.assert_no_pending()
    _yield(0.1)  # let cancellation reach the asyncio.Task
    # Phase 2: cancellation is observable via the registry directly —
    # the task is dropped, and the still-running asyncio.Task has been
    # marked cancelled. No DB row to consult.
    assert registry.get(pending_id) is None


def test_new_question_while_waiting_cancels_old(alice, db, mock_llm: MockLLMClient):
    _setup_linked_user(db, alice)
    mock_llm.respond_after(60.0, "first never returns")
    # Classifier sees the second utterance — no matching tool, treated as new.
    mock_llm.respond_instantly("classifier sees a new question")
    mock_llm.respond_instantly("Свежий ответ.")
    alice.say("первый вопрос")
    alice.assert_has_pending()
    alice.say("совсем другой вопрос")
    alice.assert_text_equals("Свежий ответ.")
    alice.assert_no_pending()


def test_ne_prefix_does_not_abort_wait(alice, db, mock_llm: MockLLMClient):
    """Phase 3: the LLM classifier decides yes/no/exit/new during a wait
    turn. "не понимаю" looks like neither a wait_more nor a cancel — the
    classifier emits no tool, and the turn is dispatched as a fresh
    question."""
    _setup_linked_user(db, alice)
    mock_llm.respond_after(60.0, "old question never returns")
    mock_llm.respond_instantly("classifier treats this as new")
    mock_llm.respond_instantly("Новый ответ.")
    alice.say("первый вопрос")
    alice.assert_has_pending()
    alice.say("не понимаю что происходит")
    alice.assert_text_equals("Новый ответ.")
    alice.assert_no_pending()


def test_max_wait_turns_gives_up(alice, db, mock_llm: MockLLMClient):
    _setup_linked_user(db, alice)
    mock_llm.respond_after(60.0, "never")
    # One classifier "wait_more" per follow-up turn.
    mock_llm.call_tool_instantly("wait_more")
    mock_llm.call_tool_instantly("wait_more")
    mock_llm.call_tool_instantly("wait_more")
    alice.say("вопрос")  # wait_phrases[0], wait_turns=1
    alice.say("да")  # wait_phrases[1], wait_turns=2
    alice.say("да")  # wait_phrases[2], wait_turns=3
    alice.say("да")  # exceeds max_wait_turns -> give-up phrase
    alice.assert_text_equals(FAST_CONFIG["persona"]["give_up_phrase"])
    alice.assert_no_pending()


def test_llm_error_during_wait(alice, db, mock_llm: MockLLMClient):
    _setup_linked_user(db, alice)
    mock_llm.raise_after(0.2, RuntimeError("llm boom"))
    mock_llm.call_tool_instantly("wait_more")  # classifier on "да"
    alice.say("вопрос")
    alice.assert_has_pending()
    _yield(0.4)
    alice.say("да")
    alice.assert_text_equals(persona.LLM_ERROR)


def test_exit_while_waiting_cancels(alice, db, mock_llm: MockLLMClient, registry):
    _setup_linked_user(db, alice)
    mock_llm.respond_after(60.0, "never")
    mock_llm.call_tool_instantly("exit_skill")  # classifier on "хватит"
    alice.say("вопрос")
    pending_id = alice.last_session_state["pending_id"]
    alice.say("хватит")
    alice.assert_end_session()
    alice.assert_text_equals(persona.FAREWELL)
    _yield(0.1)  # let cancellation reach the asyncio.Task
    assert registry.get(pending_id) is None
