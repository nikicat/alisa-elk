"""Mock Yandex Dialogs client driving the webhook through a realistic session.

Tracks message_id, session_id, and round-trips session_state from the previous
response into the next request, mirroring how real Alice behaves.
"""

import uuid
from typing import Any

from fastapi.testclient import TestClient


class AliceSession:
    def __init__(
        self,
        client: TestClient,
        *,
        webhook_path: str,
        skill_id: str,
        application_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        self.client = client
        self.webhook_path = webhook_path
        self.skill_id = skill_id
        self.application_id = application_id or f"app-{uuid.uuid4().hex[:12]}"
        self.session_id = session_id or f"sess-{uuid.uuid4().hex[:12]}"
        self.message_id = 0
        self._session_state: dict[str, Any] = {}
        self._user_state: dict[str, Any] = {}
        self._application_state: dict[str, Any] = {}
        self.responses: list[dict[str, Any]] = []
        self._new = True

    def _build_payload(self, command: str) -> dict[str, Any]:
        original = command
        tokens = command.split() if command else []
        return {
            "meta": {
                "locale": "ru-RU",
                "timezone": "Europe/Moscow",
                "client_id": "test",
                "interfaces": {},
            },
            "session": {
                "message_id": self.message_id,
                "session_id": self.session_id,
                "skill_id": self.skill_id,
                "new": self._new,
                "application": {"application_id": self.application_id},
                "user": None,
            },
            "request": {
                "command": command,
                "original_utterance": original,
                "type": "SimpleUtterance",
                "markup": {},
                "nlu": {"tokens": tokens, "entities": [], "intents": {}},
            },
            "state": {
                "session": self._session_state,
                "user": self._user_state,
                "application": self._application_state,
            },
            "version": "1.0",
        }

    def say(self, command: str) -> dict[str, Any]:
        payload = self._build_payload(command)
        resp = self.client.post(self.webhook_path, json=payload)
        assert resp.status_code == 200, (resp.status_code, resp.text)
        data = resp.json()
        self.responses.append(data)
        self._session_state = data.get("session_state") or {}
        if "user_state_update" in data and data["user_state_update"] is not None:
            self._user_state.update(data["user_state_update"])
        self.message_id += 1
        self._new = False
        # Sessions end via response.end_session; subsequent say() starts a new one.
        if data["response"].get("end_session"):
            self.session_id = f"sess-{uuid.uuid4().hex[:12]}"
            self.message_id = 0
            self._session_state = {}
            self._new = True
        return data

    # --- convenience ---

    @property
    def last_text(self) -> str:
        return self.responses[-1]["response"]["text"]

    @property
    def last_session_state(self) -> dict[str, Any]:
        return self.responses[-1].get("session_state") or {}

    def assert_text_contains(self, substr: str) -> None:
        text = self.last_text
        assert substr in text, f"expected {substr!r} in {text!r}"

    def assert_text_equals(self, expected: str) -> None:
        assert self.last_text == expected, (self.last_text, expected)

    def assert_end_session(self) -> None:
        assert self.responses[-1]["response"]["end_session"] is True

    def assert_has_pending(self) -> None:
        assert "pending_id" in self.last_session_state, self.last_session_state

    def assert_no_pending(self) -> None:
        assert "pending_id" not in self.last_session_state, self.last_session_state
