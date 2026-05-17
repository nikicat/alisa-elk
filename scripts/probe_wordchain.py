"""Probe an LLM for word-chain ("игра в слова") consistency.

Runs N sessions of up to M rounds each. The "player" picks valid words
from a small Russian noun list; the bot must reply with a noun starting
with the last letter of the player's word, and must not repeat any word
already used in the session. Prints per-session and overall stats on how
many bot turns were valid.

Two modes:
  - casual (default): system prompt = the real elk persona, player opens
    with "давай поиграем в слова, начинай", bot starts the chain. This is
    the realistic Alice flow — no rules in the system prompt.
  - strict (--strict-rules): system prompt spells out the game rules
    explicitly and player supplies a seed word. Lets you separate
    "doesn't understand the game" from "can't track turns".

Usage:
    uv run python -m scripts.probe_wordchain
    uv run python -m scripts.probe_wordchain --sessions 5 --rounds 10
    uv run python -m scripts.probe_wordchain --strict-rules
    uv run python -m scripts.probe_wordchain --model gemini-2.5-pro --verbose
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from dataclasses import dataclass, field

import httpx

from app import persona
from app.config import get_settings

# Letters that "don't count" as the chain letter — if a word ends in one,
# look one position further back. Standard игра-в-слова convention.
SKIP_LETTERS = frozenset({"ь", "ъ", "ы"})

# Small wordlist organised by starting letter. Common, unambiguous nouns,
# singular nominative. Enough variety that the player almost never runs
# out of options inside a 10–15 round session.
WORDLIST: dict[str, list[str]] = {
    "а": ["арбуз", "апельсин", "автомобиль", "артист", "азбука", "аист", "аптека", "акула"],
    "б": ["банан", "баран", "барсук", "береза", "билет", "бочка", "букварь", "бутылка"],
    "в": ["ваза", "валенок", "вагон", "ведро", "вертолет", "ветер", "виноград", "волк", "ворона"],
    "г": ["газета", "гитара", "голубь", "гора", "гриб", "груша", "гусь"],
    "д": ["дверь", "дельфин", "дерево", "диван", "дождь", "доктор", "дом", "дорога", "дуб"],
    "е": ["енот", "ежевика", "елка"],
    "ж": ["жаба", "жемчуг", "жираф", "журнал", "желудь"],
    "з": ["заяц", "зебра", "зеркало", "змея", "золото", "зонт", "звезда"],
    "и": ["игла", "икра", "индюк", "иголка", "ива"],
    "к": ["кабан", "картина", "карандаш", "кенгуру", "кит", "книга", "корова", "кот", "кран", "крыша"],
    "л": ["лампа", "лебедь", "лев", "лес", "лимон", "лиса", "ложка", "лошадь", "луна", "лужа"],
    "м": ["магазин", "мак", "малина", "медведь", "мел", "мешок", "молоко", "мост", "мяч"],
    "н": ["небо", "нога", "нос", "нота", "носок", "ножик"],
    "о": ["облако", "овца", "огонь", "окно", "олень", "орех", "осел", "остров"],
    "п": ["парус", "паук", "перо", "петух", "пингвин", "поезд", "поле", "потолок", "птица"],
    "р": ["радуга", "ракета", "ракушка", "ребенок", "река", "репа", "рис", "рукав", "рыба"],
    "с": ["сад", "самолет", "свеча", "свинья", "скала", "слон", "снег", "собака", "солнце", "стол"],
    "т": ["тарелка", "телевизор", "тигр", "топор", "торт", "трава", "троллейбус", "туча"],
    "у": ["ужин", "улитка", "улица", "утка", "ухо", "утюг"],
    "ф": ["фабрика", "фасоль", "ферма", "фонарь", "фонтан", "футбол"],
    "х": ["халат", "хвост", "хлеб", "холм", "хомяк"],
    "ц": ["цапля", "цветок", "цемент", "цирк", "цыпленок"],
    "ч": ["чайник", "чашка", "человек", "черепаха", "чердак"],
    "ш": ["шапка", "шарик", "шкаф", "школа", "шляпа", "шоколад", "шуба"],
    "щ": ["щенок", "щука", "щит"],
    "э": ["экран", "элемент", "экскаватор", "эскимо"],
    "ю": ["юбка", "юг", "юла", "юноша"],
    "я": ["яблоко", "ягода", "якорь", "ястреб", "ящерица"],
}

SEEDS = ["солнце", "лес", "дорога", "книга", "море", "ветер", "облако", "птица"]

CASUAL_OPENER = "давай поиграем в слова, начинай"

STRICT_SYSTEM_PROMPT = (
    "Мы играем в игру «слова». Правила: каждый игрок по очереди называет одно "
    "существительное в единственном числе именительного падежа, которое "
    "начинается на последнюю букву предыдущего слова. Если предыдущее слово "
    "заканчивается на «ь», «ъ» или «ы», берётся предпоследняя буква. "
    "Слова повторять нельзя. Сообщения от собеседника — это его ход. "
    "В ответ называй ровно ОДНО слово, без пояснений и знаков препинания."
)


def required_start(word: str) -> str | None:
    """Return the chain letter for `word`, skipping ь/ъ/ы."""
    for ch in reversed(word.lower().strip()):
        if "а" <= ch <= "я" or ch == "ё":
            if ch in SKIP_LETTERS:
                continue
            return ch
    return None


def extract_word(text: str) -> str | None:
    """Pick the bot's intended word from its reply.

    In casual mode the elk persona will often wrap the word in a short
    sentence ("Хорошо, давай. Лес."). We take the last cyrillic token of
    length ≥ 2 — that matches how a human would interpret the answer and
    how Alice's TTS leaves the word ringing in the user's ear.
    """
    matches = re.findall(r"[а-яё]{2,}", text.lower())
    return matches[-1] if matches else None


# Phrases an honest bot is likely to use when it spots that the player
# broke the rules. Used as a positive signal for cheat detection on top of
# the behavioral check (did the bot refuse to chain off the cheated word).
CHEAT_KEYWORDS = re.compile(
    r"не\s+на\s+ту|не\s+та\s+бук|неправ|ошиб|должн[оа]|"
    r"уже\s+(было|называл)|повтор|не\s+подход|не\s+годит|"
    r"правил|жульнич|чит",
    re.IGNORECASE,
)


@dataclass
class RoundResult:
    bot_word: str | None
    raw: str
    ok: bool
    reason: str  # ok | wrong_letter | repeat | empty | http_error | cheat_detected | cheat_accepted
    expected_letter: str | None = None
    cheat_kind: str | None = None  # None | "wrong_letter"


@dataclass
class SessionResult:
    seed: str  # the seed word (strict) or "(casual)" placeholder
    mode: str  # "casual" or "strict"
    rounds: list[RoundResult] = field(default_factory=list)
    stopped_reason: str = ""  # max_rounds | bot_fail | no_player_word | http_error | cheat_probe

    @property
    def chain_rounds(self) -> list[RoundResult]:
        return [r for r in self.rounds if r.cheat_kind is None]

    @property
    def cheat_rounds(self) -> list[RoundResult]:
        return [r for r in self.rounds if r.cheat_kind is not None]

    @property
    def valid_rounds(self) -> int:
        return sum(1 for r in self.chain_rounds if r.ok)


def player_pick(letter: str, used: set[str], rng: random.Random) -> str | None:
    options = [w for w in WORDLIST.get(letter, []) if w not in used]
    if not options:
        return None
    return rng.choice(options)


def pick_cheat_word(
    required_letter: str, used: set[str], rng: random.Random
) -> str | None:
    """Pick a word that deliberately starts with the WRONG letter."""
    candidates: list[str] = []
    for letter, words in WORDLIST.items():
        if letter == required_letter:
            continue
        candidates.extend(w for w in words if w not in used)
    if not candidates:
        return None
    return rng.choice(candidates)


def call_llm(
    client: httpx.Client,
    url: str,
    api_key: str,
    model: str,
    messages: list[dict],
    temperature: float,
) -> str:
    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 32,
    }
    r = client.post(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        json=body,
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"].get("content") or ""


def run_session(
    *,
    client: httpx.Client,
    url: str,
    api_key: str,
    model: str,
    mode: str,
    seed: str | None,
    max_rounds: int,
    temperature: float,
    rng: random.Random,
    stop_on_fail: bool,
    cheat_rate: float,
) -> SessionResult:
    if mode == "strict":
        assert seed is not None
        used: set[str] = {seed}
        messages: list[dict] = [
            {"role": "system", "content": STRICT_SYSTEM_PROMPT},
            {"role": "user", "content": seed},
        ]
        prev_word: str | None = seed
        seed_label = seed
    else:  # casual
        used = set()
        messages = [
            {"role": "system", "content": persona.SYSTEM_PROMPT},
            {"role": "user", "content": CASUAL_OPENER},
        ]
        prev_word = None  # bot starts the chain, no constraint on round 1
        seed_label = "(casual)"
    result = SessionResult(seed=seed_label, mode=mode)

    for _ in range(max_rounds):
        expected = required_start(prev_word) if prev_word else None
        try:
            raw = call_llm(client, url, api_key, model, messages, temperature)
        except httpx.HTTPError as exc:
            result.rounds.append(
                RoundResult(None, str(exc), False, "http_error", expected)
            )
            result.stopped_reason = "http_error"
            break

        bot_word = extract_word(raw)
        if bot_word is None:
            rr = RoundResult(None, raw, False, "empty", expected)
        elif expected and not bot_word.startswith(expected):
            rr = RoundResult(bot_word, raw, False, "wrong_letter", expected)
        elif bot_word in used:
            rr = RoundResult(bot_word, raw, False, "repeat", expected)
        else:
            rr = RoundResult(bot_word, raw, True, "ok", expected)
        result.rounds.append(rr)

        if not rr.ok:
            result.stopped_reason = "bot_fail"
            if stop_on_fail or bot_word is None:
                break
            # Continue: pretend the bot's claimed word is valid so the game
            # can proceed and we can observe later turns too.

        assert bot_word is not None
        used.add(bot_word)
        messages.append({"role": "assistant", "content": bot_word})

        next_letter = required_start(bot_word)
        if next_letter is None:
            result.stopped_reason = "no_player_word"
            break

        # Cheat probe: with probability `cheat_rate`, the player breaks the
        # rule. We then send one more bot turn, grade detection, and end
        # the session — the chain is dead either way.
        if cheat_rate > 0 and rng.random() < cheat_rate:
            cheat_word = pick_cheat_word(next_letter, used, rng)
            if cheat_word is not None:
                messages.append({"role": "user", "content": cheat_word})
                try:
                    raw = call_llm(
                        client, url, api_key, model, messages, temperature
                    )
                except httpx.HTTPError as exc:
                    result.rounds.append(
                        RoundResult(
                            None, str(exc), False, "http_error",
                            next_letter, cheat_kind="wrong_letter",
                        )
                    )
                    result.stopped_reason = "http_error"
                    break
                probe_word = extract_word(raw)
                cheat_chain_letter = required_start(cheat_word)
                keyword_hit = bool(CHEAT_KEYWORDS.search(raw))
                accepted = (
                    probe_word is not None
                    and cheat_chain_letter is not None
                    and probe_word.startswith(cheat_chain_letter)
                    and not keyword_hit
                )
                reason = "cheat_accepted" if accepted else "cheat_detected"
                result.rounds.append(
                    RoundResult(
                        probe_word, raw, not accepted, reason,
                        next_letter, cheat_kind="wrong_letter",
                    )
                )
                result.stopped_reason = "cheat_probe"
                break

        next_word = player_pick(next_letter, used, rng)
        if next_word is None:
            result.stopped_reason = "no_player_word"
            break
        used.add(next_word)
        messages.append({"role": "user", "content": next_word})
        prev_word = next_word
    else:
        result.stopped_reason = "max_rounds"

    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Probe an LLM's ability to play игра-в-слова over multiple sessions."
    )
    parser.add_argument("--sessions", type=int, default=5, help="number of sessions to run")
    parser.add_argument("--rounds", type=int, default=10, help="max bot turns per session")
    parser.add_argument("--model", help="Override LLM_MODEL from env/.env")
    parser.add_argument("--base-url", help="Override LLM_BASE_URL from env/.env")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for reproducibility")
    parser.add_argument("--verbose", action="store_true", help="print every bot turn")
    parser.add_argument(
        "--continue-on-fail",
        action="store_true",
        help="Don't stop a session on bot's first mistake; keep going so we can measure later turns too.",
    )
    parser.add_argument(
        "--strict-rules",
        action="store_true",
        help="Spell out the game rules in the system prompt and seed the chain with a starting word. "
        "Default is casual mode: real elk persona + 'давай поиграем в слова, начинай'.",
    )
    parser.add_argument(
        "--cheat-rate",
        type=float,
        default=0.0,
        help="Probability per player turn that the player intentionally picks a word "
        "starting with the WRONG letter, to test whether the bot calls it out. "
        "Each cheat ends the session. Try 0.3.",
    )
    args = parser.parse_args(argv)
    mode = "strict" if args.strict_rules else "casual"
    if not 0.0 <= args.cheat_rate <= 1.0:
        print("error: --cheat-rate must be in [0, 1]", file=sys.stderr)
        return 2

    settings = get_settings()
    base_url = (args.base_url or settings.LLM_BASE_URL).rstrip("/")
    url = base_url if base_url.endswith("/chat/completions") else base_url + "/chat/completions"
    model = args.model or settings.LLM_MODEL
    api_key = settings.LLM_API_KEY
    if not api_key or api_key in {"sk-dev", "sk-replace-me"}:
        print("error: LLM_API_KEY missing or placeholder. Set it in .env.", file=sys.stderr)
        return 2

    print(f"model     : {model}")
    print(f"base      : {url}")
    print(f"mode      : {mode}")
    print(f"temp      : {args.temperature}")
    print(f"sessions  : {args.sessions}")
    print(f"max rounds: {args.rounds}")
    print(f"cheat rate: {args.cheat_rate}\n")

    rng = random.Random(args.seed)
    sessions: list[SessionResult] = []
    with httpx.Client() as client:
        for i in range(args.sessions):
            seed_word = rng.choice(SEEDS) if mode == "strict" else None
            sr = run_session(
                client=client,
                url=url,
                api_key=api_key,
                model=model,
                mode=mode,
                seed=seed_word,
                max_rounds=args.rounds,
                temperature=args.temperature,
                rng=rng,
                stop_on_fail=not args.continue_on_fail,
                cheat_rate=args.cheat_rate,
            )
            sessions.append(sr)
            cheat_tag = ""
            if sr.cheat_rounds:
                cr = sr.cheat_rounds[0]
                cheat_tag = f" cheat={'DETECTED' if cr.ok else 'ACCEPTED'}"
            print(
                f"[session {i+1}] seed={sr.seed!r} "
                f"valid={sr.valid_rounds}/{len(sr.chain_rounds)} "
                f"stopped={sr.stopped_reason}{cheat_tag}"
            )
            if args.verbose:
                for j, r in enumerate(sr.rounds, 1):
                    tag = "OK" if r.ok else r.reason.upper()
                    want = f"→{r.expected_letter}" if r.expected_letter else ""
                    marker = "*" if r.cheat_kind else " "
                    raw_snip = r.raw.replace("\n", " ")[:60]
                    print(f"   {j:2d}.{marker}[{tag:16s}] {want:>4s}  {r.bot_word or '?'!r}  raw={raw_snip!r}")

    chain_total = sum(len(s.chain_rounds) for s in sessions)
    chain_valid = sum(s.valid_rounds for s in sessions)
    cheats_attempted = sum(len(s.cheat_rounds) for s in sessions)
    cheats_detected = sum(1 for s in sessions for r in s.cheat_rounds if r.ok)
    by_reason: dict[str, int] = {}
    for s in sessions:
        for r in s.rounds:
            by_reason[r.reason] = by_reason.get(r.reason, 0) + 1
    chain_pct = (chain_valid / chain_total * 100) if chain_total else 0.0
    cheat_pct = (cheats_detected / cheats_attempted * 100) if cheats_attempted else 0.0

    print("\n--- summary ---")
    print(f"chain turns       : {chain_total}")
    print(f"chain valid       : {chain_valid} ({chain_pct:.1f}%)")
    if cheats_attempted:
        print(f"cheats attempted  : {cheats_attempted}")
        print(f"cheats detected   : {cheats_detected} ({cheat_pct:.1f}%)")
    print("by reason:")
    for k, v in sorted(by_reason.items()):
        print(f"   {k:16s} {v}")
    chain_clean = chain_total > 0 and chain_valid == chain_total
    cheats_clean = cheats_attempted == 0 or cheats_detected == cheats_attempted
    return 0 if chain_clean and cheats_clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
