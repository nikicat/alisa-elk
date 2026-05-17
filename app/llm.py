from typing import Protocol

import httpx


class LLMTimeoutError(Exception):
    pass


class LLMError(Exception):
    pass


class LLMResult:
    __slots__ = ("text", "input_tokens", "output_tokens", "tool_calls")

    def __init__(
        self,
        text: str,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        tool_calls: list[str] | None = None,
    ) -> None:
        self.text = text
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.tool_calls = tool_calls or []


class LLMClient(Protocol):
    async def complete(
        self,
        messages: list[dict],
        *,
        max_tokens: int,
        temperature: float,
        tools: list[dict] | None = None,
    ) -> LLMResult: ...


class OpenAIRouterClient:
    """OpenAI /v1/chat/completions-compatible client.

    Works with any router that speaks the OpenAI shape: LiteLLM, OpenRouter, etc.
    Configure via base_url, api_key, model.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=5.0),
            limits=httpx.Limits(max_keepalive_connections=4, keepalive_expiry=60),
        )
        self._owns_client = client is None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def complete(
        self,
        messages: list[dict],
        *,
        max_tokens: int,
        temperature: float,
        tools: list[dict] | None = None,
    ) -> LLMResult:
        if self.base_url.endswith("/chat/completions"):
            url = self.base_url
        else:
            url = f"{self.base_url}/chat/completions"
        body: dict = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if tools:
            body["tools"] = tools
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            resp = await self._client.post(url, json=body, headers=headers)
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(str(exc)) from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"transport: {exc}") from exc
        if resp.status_code >= 400:
            raise LLMError(f"http {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError) as exc:
            raise LLMError(f"bad shape: {exc}") from exc
        raw_tool_calls = message.get("tool_calls") or []
        tool_call_names = [
            tc.get("function", {}).get("name")
            for tc in raw_tool_calls
            if tc.get("function", {}).get("name")
        ]
        text = (message.get("content") or "").strip()
        if not text and not tool_call_names:
            raise LLMError("bad shape: empty content and no tool calls")
        usage = data.get("usage") or {}
        return LLMResult(
            text=text,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            tool_calls=tool_call_names,
        )
