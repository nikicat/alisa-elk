import asyncio
from collections.abc import Callable
from dataclasses import dataclass

from app.llm import LLMError, LLMResult


@dataclass
class _Behaviour:
    delay_s: float
    text: str | None
    raise_exc: Exception | None
    on_call: Callable[[list[dict]], None] | None = None
    tool_calls: list[str] | None = None


class MockLLMClient:
    """Configurable fake LLM. One MockLLMClient instance can be scripted across
    multiple consecutive calls to test mixed behaviours within a session.
    """

    def __init__(self) -> None:
        self._behaviours: list[_Behaviour] = []
        self.calls: list[list[dict]] = []
        self._default = _Behaviour(0.0, "Лесная мудрость в одной строке.", None)

    # --- builders ---

    def respond_instantly(self, text: str) -> MockLLMClient:
        self._behaviours.append(_Behaviour(0.0, text, None))
        return self

    def call_tool_instantly(self, *names: str) -> MockLLMClient:
        self._behaviours.append(_Behaviour(0.0, "", None, tool_calls=list(names)))
        return self

    def call_tool_after(self, delay_s: float, *names: str) -> MockLLMClient:
        self._behaviours.append(_Behaviour(delay_s, "", None, tool_calls=list(names)))
        return self

    def respond_after(self, delay_s: float, text: str) -> MockLLMClient:
        self._behaviours.append(_Behaviour(delay_s, text, None))
        return self

    def raise_after(self, delay_s: float, exc: Exception) -> MockLLMClient:
        self._behaviours.append(_Behaviour(delay_s, None, exc))
        return self

    def script(self, behaviours: list[_Behaviour]) -> MockLLMClient:
        self._behaviours.extend(behaviours)
        return self

    def set_default(self, text: str) -> MockLLMClient:
        self._default = _Behaviour(0.0, text, None)
        return self

    # --- protocol ---

    async def complete(
        self,
        messages: list[dict],
        *,
        max_tokens: int,
        temperature: float,
        tools: list[dict] | None = None,
    ) -> LLMResult:
        self.calls.append(list(messages))
        behaviour = self._behaviours.pop(0) if self._behaviours else self._default
        if behaviour.delay_s > 0:
            await asyncio.sleep(behaviour.delay_s)
        if behaviour.raise_exc is not None:
            raise behaviour.raise_exc
        return LLMResult(
            text=behaviour.text or "",
            input_tokens=10,
            output_tokens=20,
            tool_calls=behaviour.tool_calls or [],
        )

    async def aclose(self) -> None:
        return None


def make_llm_error(text: str = "boom") -> Exception:
    return LLMError(text)
