"""Probe the configured LLM router for tool-calling behaviour.

Sends a small set of fixed prompts that should and should not trigger the
`reset_context` tool, then prints a pass/fail report. Useful when swapping
models or routers to confirm intent detection still works.

Usage:
    uv run python -m scripts.probe_llm
    uv run python -m scripts.probe_llm --model gemini-3-flash-preview
    uv run python -m scripts.probe_llm --show-text  # also print text replies
"""

import argparse
import sys
from dataclasses import dataclass

import httpx

from app.config import get_settings

RESET_TOOL = {
    "type": "function",
    "function": {
        "name": "reset_context",
        "description": (
            "Call this when the user wants to forget the conversation so far "
            "and start fresh, e.g. 'забудь', 'забудь всё', 'начнём заново', "
            "'сменим тему'. Do NOT call for normal questions or small abort "
            "requests."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}

PRIMING_HISTORY = [
    {"role": "system", "content": "Ты — Мудрый Лось, отвечаешь по-русски кратко."},
    {"role": "user", "content": "что такое осень"},
    {"role": "assistant", "content": "Осень — дрёма леса."},
]


@dataclass(frozen=True)
class Case:
    command: str
    should_reset: bool


CASES: list[Case] = [
    # Plain questions — never reset.
    Case("а зима?", False),
    Case("почему люди забывают сны", False),
    Case("как пишется слово забудь", False),
    Case("забывание это нормальный процесс?", False),
    # Phrases that mention the trigger words but mean something else.
    Case("не забудь рассказать ещё про осень", False),
    Case("что значит фраза забудь и иди дальше", False),
    Case("напомни мне не забыть полить цветы", False),
    Case("расскажи историю где герой забудь о своей жизни", False),
    Case("как правильно сменить тему разговора", False),
    Case("не надо забывать прошлое", False),
    # Genuine reset requests.
    Case("забудь", True),
    Case("забудь всё", True),
    Case("сменим тему", True),
    Case("хочу поговорить о другом", True),
    Case("начнём заново", True),
]


def probe(
    client: httpx.Client,
    base_url: str,
    api_key: str,
    model: str,
    command: str,
    temperature: float,
) -> tuple[bool, str]:
    """Send one prompt and return (tool_called, text_reply)."""
    body = {
        "model": model,
        "messages": [*PRIMING_HISTORY, {"role": "user", "content": command}],
        "tools": [RESET_TOOL],
        "temperature": temperature,
        "max_tokens": 200,
    }
    r = client.post(
        base_url,
        headers={"Authorization": f"Bearer {api_key}"},
        json=body,
        timeout=20,
    )
    r.raise_for_status()
    msg = r.json()["choices"][0]["message"]
    tool_calls = msg.get("tool_calls") or []
    called = any(
        tc.get("function", {}).get("name") == "reset_context" for tc in tool_calls
    )
    text = msg.get("content") or ""
    return called, text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Probe the configured LLM for reset_context tool-calling."
    )
    parser.add_argument("--model", help="Override LLM_MODEL from env/.env")
    parser.add_argument("--base-url", help="Override LLM_BASE_URL from env/.env")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument(
        "--show-text",
        action="store_true",
        help="Print the model's text reply for non-reset cases",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    base_url = args.base_url or settings.LLM_BASE_URL
    model = args.model or settings.LLM_MODEL
    api_key = settings.LLM_API_KEY
    if not api_key or api_key in {"sk-dev", "sk-replace-me"}:
        print(
            "error: LLM_API_KEY missing or placeholder. Set it in .env.",
            file=sys.stderr,
        )
        return 2

    print(f"model: {model}")
    print(f"base : {base_url}")
    print(f"temp : {args.temperature}\n")

    passed = failed = 0
    with httpx.Client() as client:
        for case in CASES:
            try:
                called, text = probe(
                    client, base_url, api_key, model, case.command, args.temperature
                )
            except httpx.HTTPError as exc:
                print(f"[ERROR] {case.command!r}: {exc}")
                failed += 1
                continue
            ok = called == case.should_reset
            verdict = "PASS" if ok else "FAIL"
            tag = "RESET" if called else "TEXT "
            expected = "RESET" if case.should_reset else "TEXT "
            print(f"[{verdict}] [{tag}] (want {expected}) {case.command!r}")
            if args.show_text and not called and text:
                snippet = text.replace("\n", " ")[:120]
                print(f"          → {snippet}")
            if ok:
                passed += 1
            else:
                failed += 1

    total = passed + failed
    print(f"\n{passed}/{total} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
