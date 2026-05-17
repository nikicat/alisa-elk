"""Probe the words-game intent classifier with the configured LLM.

The whole spike architecture rests on `words_classify_player` correctly
routing player utterances to one of three buckets:

  - challenge_word tool → player is challenging the bot's last word
  - exit_game tool      → player is asking to leave the game
  - no tool emitted     → player is making a normal word move

If Flash-tier models fumble this, the spike is unviable. This probe
sends scripted phrases through the real classifier node (not a mock),
reports per-case PASS/FAIL plus a confusion matrix, and supports
`--runs N` to measure noise.

Usage:
    uv run python -m scripts.probe_wordsgame_classifier
    uv run python -m scripts.probe_wordsgame_classifier --model gemini-2.5-pro
    uv run python -m scripts.probe_wordsgame_classifier --runs 3
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from dataclasses import dataclass
from typing import cast

from app.config import get_settings, reset_caches
from app.games import words as spike


@dataclass(frozen=True)
class Case:
    phrase: str
    expected: str  # "challenge" | "exit" | "move"


CASES: list[Case] = [
    # --- challenges ---
    Case("сомневаюсь, такого слова нет", "challenge"),
    Case("проверь своё слово", "challenge"),
    Case("это не слово", "challenge"),
    Case("выдумал ты это слово", "challenge"),
    Case("такого слова не существует", "challenge"),
    Case("проверим словарь", "challenge"),
    Case("я не верю, что это слово", "challenge"),
    # --- exits ---
    Case("хватит играть", "exit"),
    Case("давай заканчивать игру", "exit"),
    Case("выйти из игры", "exit"),
    Case("стоп", "exit"),
    Case("закончим", "exit"),
    Case("надоело, давай прекратим", "exit"),
    # --- plain moves (no tool should fire) ---
    Case("арбуз", "move"),
    Case("яблоко", "move"),
    Case("медведь", "move"),
    Case("самолёт", "move"),
    Case("море", "move"),
    Case("я скажу яблоко", "move"),
    # --- distractors: contain trigger substrings but mean something else ---
    Case("выйди из тени", "move"),  # "выйди" ≠ exit_game
    Case("заканчивается лето", "move"),  # "заканчив" ≠ exit_game
    Case("сомневающийся человек", "move"),  # "сомнев" ≠ challenge
    Case("я думаю это слово красивое", "move"),  # "это слово" ≠ challenge
    Case("проверка работает", "move"),  # "провер" ≠ challenge
]


def classify_once(phrase: str) -> str:
    """Invoke words_classify_player on a minimal in-game state and
    bucket the resulting update."""
    state = cast(
        spike.DialogState,
        {
            "messages": [],
            "user_input": phrase,
            "game": spike.GameState(
                used=["арбуз"], required_letter="з", last_cheat=None
            ),
            "last_bot_text": None,
        },
    )
    update = spike.words_classify_player(state)
    game = update.get("game")
    if game is None and update.get("last_bot_text"):
        return "exit"
    if isinstance(game, dict) and game.get("last_cheat") == "challenge":
        return "challenge"
    return "move"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Probe the words-game intent classifier."
    )
    parser.add_argument("--model", help="Override LLM_MODEL from env/.env")
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of runs per case; majority vote (use 3+ for noisy models).",
    )
    args = parser.parse_args(argv)

    if args.model:
        os.environ["LLM_MODEL"] = args.model
        reset_caches()

    settings = get_settings()
    if not settings.LLM_API_KEY or settings.LLM_API_KEY in {"sk-dev", "sk-replace-me"}:
        print(
            "error: LLM_API_KEY missing or placeholder. Set it in .env.",
            file=sys.stderr,
        )
        return 2

    print(f"model    : {settings.LLM_MODEL}")
    print(f"base     : {settings.LLM_BASE_URL}")
    print(f"runs/case: {args.runs}")
    print(f"cases    : {len(CASES)}\n")

    passed = failed = 0
    confusion: dict[tuple[str, str], int] = {}
    buckets = ("challenge", "exit", "move")
    for case in CASES:
        results = [classify_once(case.phrase) for _ in range(args.runs)]
        counter = Counter(results)
        majority = counter.most_common(1)[0][0]
        ok = majority == case.expected
        verdict = "PASS" if ok else "FAIL"
        if args.runs == 1:
            print(
                f"[{verdict}] [{majority:<9}] (want {case.expected:<9}) {case.phrase!r}"
            )
        else:
            dist = ", ".join(f"{k}={v}" for k, v in counter.most_common())
            print(
                f"[{verdict}] [{majority:<9}] (want {case.expected:<9}) "
                f"{case.phrase!r}  [{dist}]"
            )
        key = (case.expected, majority)
        confusion[key] = confusion.get(key, 0) + 1
        if ok:
            passed += 1
        else:
            failed += 1

    total = passed + failed
    print(f"\n{passed}/{total} passed")

    print("\nconfusion (rows = expected, cols = actual):")
    header = "  " + " " * 11 + "  ".join(f"{b:<9}" for b in buckets)
    print(header)
    for e in buckets:
        cells = "  ".join(f"{confusion.get((e, a), 0):<9}" for a in buckets)
        print(f"  want {e:<8}  {cells}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
