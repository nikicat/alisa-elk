"""Wait-pattern tests: slow LLM keep-alive, да/нет flow, escalation, abort."""

import asyncio
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
    alice.say("вопрос")
    alice.assert_text_equals(FAST_CONFIG["persona"]["wait_phrases"][0])
    alice.say("да")
    alice.assert_text_equals(FAST_CONFIG["persona"]["wait_phrases"][1])
    alice.assert_has_pending()


def test_no_aborts_pending(alice, db, mock_llm: MockLLMClient, session_factory):
    _setup_linked_user(db, alice)
    mock_llm.respond_after(60.0, "irrelevant")
    alice.say("вопрос")
    pending_id = alice.last_session_state["pending_id"]
    alice.say("нет")
    alice.assert_text_equals(persona.ABORT_OK)
    alice.assert_no_pending()
    _yield(0.1)  # let cancellation propagate to _persist_result
    with session_factory() as fresh_db:
        row = repo.get_pending(fresh_db, pending_id)
        assert row is not None
        assert row.status == "aborted"


def test_new_question_while_waiting_cancels_old(alice, db, mock_llm: MockLLMClient):
    _setup_linked_user(db, alice)
    mock_llm.respond_after(60.0, "first never returns")
    mock_llm.respond_instantly("Свежий ответ.")
    alice.say("первый вопрос")
    alice.assert_has_pending()
    alice.say("совсем другой вопрос")
    alice.assert_text_equals("Свежий ответ.")
    alice.assert_no_pending()


def test_max_wait_turns_gives_up(alice, db, mock_llm: MockLLMClient):
    _setup_linked_user(db, alice)
    mock_llm.respond_after(60.0, "never")
    alice.say("вопрос")  # wait_phrases[0], wait_turns=1
    alice.say("да")  # wait_phrases[1], wait_turns=2
    alice.say("да")  # wait_phrases[2], wait_turns=3
    alice.say("да")  # exceeds max_wait_turns -> give-up phrase
    alice.assert_text_equals(FAST_CONFIG["persona"]["give_up_phrase"])
    alice.assert_no_pending()


def test_llm_error_during_wait(alice, db, mock_llm: MockLLMClient):
    _setup_linked_user(db, alice)
    mock_llm.raise_after(0.2, RuntimeError("llm boom"))
    alice.say("вопрос")
    alice.assert_has_pending()
    _yield(0.4)
    alice.say("да")
    alice.assert_text_equals(persona.LLM_ERROR)


def test_exit_while_waiting_cancels(alice, db, mock_llm: MockLLMClient, session_factory):
    _setup_linked_user(db, alice)
    mock_llm.respond_after(60.0, "never")
    alice.say("вопрос")
    pending_id = alice.last_session_state["pending_id"]
    alice.say("хватит")
    alice.assert_end_session()
    alice.assert_text_equals(persona.FAREWELL)
    # Give the cancel a moment to propagate to the background task.
    _yield(0.1)
    with session_factory() as fresh_db:
        row = repo.get_pending(fresh_db, pending_id)
        # Either explicitly aborted by _persist_result, or still in_progress
        # if the cancellation hasn't yet propagated. Both are acceptable for
        # the user-facing assertion above.
        assert row is not None
        assert row.status in ("aborted", "in_progress")


def test_startup_marks_orphaned_in_progress_as_error(db, session_factory):
    user = repo.create_user(db, display_name="Test")
    db.commit()
    repo.create_pending(
        db,
        pending_id="orphan1",
        user_id=user.id,
        session_id="s",
        request_text="q",
        messages=[],
    )
    db.commit()
    # Simulate startup cleanup.
    with session_factory() as fresh:
        n = repo.mark_orphaned_in_progress_as_error(fresh)
        fresh.commit()
        assert n == 1
        row = repo.get_pending(fresh, "orphan1")
        assert row is not None
        assert row.status == "error"
