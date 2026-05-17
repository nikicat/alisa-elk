"""Unit tests for handler dispatch logic. LLM is always mocked."""

from datetime import timedelta

from app import persona, repo
from app.linking import detect_code
from app.pagination import chunk_for_alice
from app.tests.mock_alice import AliceSession

# ---------- helpers ----------


def _make_test_user(db) -> int:
    user = repo.create_user(db, display_name="Test")
    db.commit()
    return user.id


def _link(db, user_id: int, app_id: str) -> None:
    code = repo.create_link_code(db, user_id)
    db.commit()
    user = repo.consume_link_code(db, code, app_id)
    assert user is not None
    db.commit()


# ---------- detect_code unit tests ----------


def test_detect_code_joined():
    assert detect_code("482917", ["482917"]) == "482917"


def test_detect_code_in_sentence():
    assert detect_code("мой код 482917 пожалуйста", []) == "482917"


def test_detect_code_spaced_tokens():
    assert detect_code("", ["4", "8", "2", "9", "1", "7"]) == "482917"


def test_detect_code_spaced_with_noise():
    # ASR sometimes wedges noise tokens between digits — only contiguous digits count.
    assert detect_code("", ["4", "8", "э", "2", "9", "1", "7"]) is None


def test_detect_code_too_short():
    assert detect_code("12345", ["12345"]) is None


def test_detect_code_too_long_run():
    # Seven digits — not a valid code; pattern requires exactly 6 in a row.
    assert detect_code("1234567", ["1234567"]) is None


def test_detect_code_missing():
    assert detect_code("привет лось", ["привет", "лось"]) is None


# ---------- consume_link_code atomicity ----------


def test_consume_link_code_single_use(db):
    user_id = _make_test_user(db)
    code = repo.create_link_code(db, user_id)
    db.commit()
    a = repo.consume_link_code(db, code, "app-A")
    db.commit()
    b = repo.consume_link_code(db, code, "app-B")
    db.commit()
    assert a is not None
    assert b is None


def test_consume_link_code_expired(db):
    user_id = _make_test_user(db)
    code = repo.create_link_code(db, user_id, ttl_minutes=15)
    # Force expiry by rewinding.
    from app.models import LinkCode

    row = db.get(LinkCode, code)
    row.expires_at = repo.utcnow() - timedelta(seconds=1)
    db.commit()
    result = repo.consume_link_code(db, code, "app-X")
    assert result is None


# ---------- chunk_for_alice ----------


def test_chunk_short_text_one_piece():
    assert chunk_for_alice("короткий ответ", chunk_chars=1000) == ["короткий ответ"]


def test_chunk_splits_at_sentence_boundary():
    text = "Первое предложение. Второе предложение. Третье предложение."
    chunks = chunk_for_alice(text, chunk_chars=30)
    assert len(chunks) >= 2
    assert all(len(c) <= 30 for c in chunks)


def test_chunk_hard_split_overflow_sentence():
    sentence = "А" * 250
    chunks = chunk_for_alice(sentence, chunk_chars=100)
    assert len(chunks) == 3
    assert "".join(chunks) == sentence


# ---------- dispatch via TestClient + mock Alice ----------


def test_skill_id_mismatch_ends_session(alice: AliceSession, app_client):
    payload = alice._build_payload("привет")
    payload["session"]["skill_id"] = "wrong-skill"
    resp = app_client.post(alice.webhook_path, json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["response"]["end_session"] is True


def test_first_turn_empty_greets_unlinked(alice: AliceSession):
    alice.say("")
    alice.assert_text_contains("код")


def test_unlinked_user_with_question_gets_link_prompt(alice: AliceSession):
    alice.say("что такое осень")
    alice.assert_text_equals(persona.UNLINKED_QUESTION)


def test_help_command(alice: AliceSession, db, session_factory):
    user_id = _make_test_user(db)
    _link(db, user_id, alice.application_id)
    alice.say("помощь")
    assert "Мудрый Лось" in alice.last_text


def test_exit_command_ends_session(alice: AliceSession, db):
    user_id = _make_test_user(db)
    _link(db, user_id, alice.application_id)
    alice.say("хватит")
    alice.assert_end_session()
    alice.assert_text_equals(persona.FAREWELL)


def test_first_turn_with_question_calls_llm(alice, db, mock_llm):
    user_id = _make_test_user(db)
    _link(db, user_id, alice.application_id)
    mock_llm.respond_instantly("Тишина — это голос леса.")
    alice.say("что такое тишина")
    alice.assert_text_equals("Тишина — это голос леса.")
    assert len(mock_llm.calls) == 1
    # System prompt always first.
    assert mock_llm.calls[0][0]["role"] == "system"
    assert mock_llm.calls[0][-1] == {"role": "user", "content": "что такое тишина"}


def test_link_code_via_voice(alice, db):
    user_id = _make_test_user(db)
    code = repo.create_link_code(db, user_id)
    db.commit()
    alice.say(code)
    alice.assert_text_equals(persona.LINK_OK)
    # Now they can ask a question.
    alice.responses.clear()


def test_link_code_bad(alice: AliceSession):
    alice.say("123456")
    alice.assert_text_equals(persona.LINK_BAD)


def test_multi_turn_memory_passes_history(alice, db, mock_llm):
    user_id = _make_test_user(db)
    _link(db, user_id, alice.application_id)
    mock_llm.respond_instantly("Осень — это дрёма леса.")
    mock_llm.respond_instantly("Зима — это его сон.")
    alice.say("что такое осень")
    alice.say("а зима?")
    # Second call should include the first exchange in messages.
    second_call = mock_llm.calls[1]
    user_messages = [m for m in second_call if m["role"] == "user"]
    assistant_messages = [m for m in second_call if m["role"] == "assistant"]
    assert any(m["content"] == "что такое осень" for m in user_messages)
    assert any(m["content"] == "Осень — это дрёма леса." for m in assistant_messages)
    assert user_messages[-1]["content"] == "а зима?"


def test_linked_user_with_digits_in_question_does_not_link(alice, db, mock_llm):
    """Once linked, dictating 6 digits inside a question must not trigger
    the link-code path (the digits are part of the question, not a code)."""
    user_id = _make_test_user(db)
    _link(db, user_id, alice.application_id)
    mock_llm.respond_instantly("Сто лет — это много, человек.")
    alice.say("сколько было ему 482917 лет")
    alice.assert_text_equals("Сто лет — это много, человек.")
    assert len(mock_llm.calls) == 1


def test_consume_link_code_rejects_rebind_to_different_user(db):
    """A device already linked to user A cannot be silently rebound to user B
    by anyone with a fresh code."""
    user_a = repo.create_user(db, display_name="A")
    user_b = repo.create_user(db, display_name="B")
    db.commit()
    code_a = repo.create_link_code(db, user_a.id)
    code_b = repo.create_link_code(db, user_b.id)
    db.commit()
    first = repo.consume_link_code(db, code_a, "shared-device")
    db.commit()
    assert first is not None and first.id == user_a.id
    rebind = repo.consume_link_code(db, code_b, "shared-device")
    db.commit()
    assert rebind is None  # refused
    # And the original binding survives.
    from sqlalchemy import select

    from app.models import LinkedAccount

    bound = db.execute(
        select(LinkedAccount).where(
            LinkedAccount.yandex_application_id == "shared-device"
        )
    ).scalar_one()
    assert bound.user_id == user_a.id


def test_pagination_long_response_chunks(alice, db, mock_llm):
    user_id = _make_test_user(db)
    _link(db, user_id, alice.application_id)
    long = ("Лес шумит. " * 200).strip()  # ~2200 chars
    mock_llm.respond_instantly(long)
    alice.say("расскажи длинную сказку")
    assert "cursor" in alice.last_session_state
    assert persona.CONTINUE_PROMPT in alice.last_text
    # Continuation.
    alice.say("дальше")
    # Either another chunk + prompt, or final chunk with no cursor.
    assert alice.responses[-1]["response"]["text"]
