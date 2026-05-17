"""End-to-end tests: mock Alice drives the FastAPI app, mock LLM behind handler.

Each test verifies a complete user flow including state propagation across turns.
Per-turn wall time is asserted to be under 4500 ms (with mocks, it's <100 ms).
"""

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


def _create_user_with_code(db) -> tuple[int, str]:
    user = repo.create_user(db, display_name="Test")
    db.commit()
    code = repo.create_link_code(db, user.id)
    db.commit()
    return user.id, code


def _assert_fast(elapsed_ms: int) -> None:
    assert elapsed_ms < 4500, f"turn took {elapsed_ms} ms (over budget)"


def _timed_say(alice: AliceSession, text: str) -> tuple[dict, int]:
    t0 = time.perf_counter()
    resp = alice.say(text)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    _assert_fast(elapsed_ms)
    return resp, elapsed_ms


def test_happy_path_linked_user(alice, db, mock_llm: MockLLMClient):
    user = repo.create_user(db, display_name="Test")
    db.commit()
    code = repo.create_link_code(db, user.id)
    db.commit()
    repo.consume_link_code(db, code, alice.application_id)
    db.commit()

    mock_llm.respond_instantly("Тишина — это голос леса.")
    _timed_say(alice, "что такое тишина")
    alice.assert_text_equals("Тишина — это голос леса.")
    alice.assert_no_pending()


def test_linking_flow_then_ask(alice, db, mock_llm: MockLLMClient):
    user_id, code = _create_user_with_code(db)

    # 1. First turn: unlinked greeting.
    _timed_say(alice, "")
    alice.assert_text_contains("код")

    # 2. Speak the code.
    _timed_say(alice, code)
    alice.assert_text_equals(persona.LINK_OK)

    # 3. Now ask a question — should hit LLM.
    mock_llm.respond_instantly("Зима — это сон леса.")
    _timed_say(alice, "что такое зима")
    alice.assert_text_equals("Зима — это сон леса.")


def test_multi_turn_memory_end_to_end(alice, db, mock_llm: MockLLMClient):
    user = repo.create_user(db, display_name="Test")
    db.commit()
    code = repo.create_link_code(db, user.id)
    db.commit()
    repo.consume_link_code(db, code, alice.application_id)
    db.commit()

    mock_llm.respond_instantly("Осень — это дрёма леса.")
    mock_llm.respond_instantly("Зима — это его сон.")
    _timed_say(alice, "что такое осень")
    _timed_say(alice, "а зима?")

    second_messages = mock_llm.calls[1]
    user_msgs = [m["content"] for m in second_messages if m["role"] == "user"]
    asst_msgs = [m["content"] for m in second_messages if m["role"] == "assistant"]
    assert "что такое осень" in user_msgs
    assert "Осень — это дрёма леса." in asst_msgs
    assert user_msgs[-1] == "а зима?"


def test_pagination_full_cycle(alice, db, mock_llm: MockLLMClient):
    user = repo.create_user(db, display_name="Test")
    db.commit()
    code = repo.create_link_code(db, user.id)
    db.commit()
    repo.consume_link_code(db, code, alice.application_id)
    db.commit()

    long_response = (". ".join(f"Часть {i}" for i in range(1, 200))) + "."
    assert len(long_response) > 1000
    mock_llm.respond_instantly(long_response)

    _timed_say(alice, "расскажи длинную сказку")
    assert "cursor" in alice.last_session_state
    assert persona.CONTINUE_PROMPT in alice.last_text

    _timed_say(alice, "дальше")
    assert alice.last_text


def test_slow_llm_happy_ending(alice, db, mock_llm: MockLLMClient):
    user = repo.create_user(db, display_name="Test")
    db.commit()
    code = repo.create_link_code(db, user.id)
    db.commit()
    repo.consume_link_code(db, code, alice.application_id)
    db.commit()

    mock_llm.respond_after(0.3, "Долгий, но мудрый ответ.")

    _timed_say(alice, "сложный вопрос")
    alice.assert_text_equals(FAST_CONFIG["persona"]["wait_phrases"][0])
    alice.assert_has_pending()

    time.sleep(0.4)  # let LLM finish
    _timed_say(alice, "да")
    alice.assert_text_equals("Долгий, но мудрый ответ.")
    alice.assert_no_pending()


def test_slow_llm_abort_with_no(alice, db, mock_llm: MockLLMClient, session_factory):
    user = repo.create_user(db, display_name="Test")
    db.commit()
    code = repo.create_link_code(db, user.id)
    db.commit()
    repo.consume_link_code(db, code, alice.application_id)
    db.commit()

    mock_llm.respond_after(60.0, "never delivered")

    _timed_say(alice, "очень сложный вопрос")
    pending_id = alice.last_session_state["pending_id"]
    _timed_say(alice, "нет")
    alice.assert_text_equals(persona.ABORT_OK)
    time.sleep(0.1)
    with session_factory() as fresh_db:
        row = repo.get_pending(fresh_db, pending_id)
        assert row is not None
        assert row.status == "aborted"


def test_exit_during_wait_ends_session(alice, db, mock_llm: MockLLMClient):
    user = repo.create_user(db, display_name="Test")
    db.commit()
    code = repo.create_link_code(db, user.id)
    db.commit()
    repo.consume_link_code(db, code, alice.application_id)
    db.commit()

    mock_llm.respond_after(60.0, "never")
    _timed_say(alice, "сложный вопрос")
    _timed_say(alice, "хватит")
    alice.assert_end_session()
    alice.assert_text_equals(persona.FAREWELL)
