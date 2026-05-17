import re

# Russian dictation can render "482917" as "482917" or "4 8 2 9 1 7" or
# "сорок восемь / два девять / один семь". We accept two forms:
#   1. Six digits joined: any run of exactly 6 contiguous digits.
#   2. Six single-digit tokens in order (with optional non-digit noise between).
# Word-form digits ("один два три") are not handled in v1.

_RUN_OF_6 = re.compile(r"\b(\d{6})\b")
_DIGIT_TOKEN = re.compile(r"^\d$")


def detect_code(original_utterance: str, tokens: list[str]) -> str | None:
    """Return a 6-digit code if the utterance contains one, else None."""
    if original_utterance:
        match = _RUN_OF_6.search(original_utterance)
        if match:
            return match.group(1)
    digits: list[str] = []
    for tok in tokens:
        if _DIGIT_TOKEN.match(tok):
            digits.append(tok)
            if len(digits) == 6:
                return "".join(digits)
        elif tok.isdigit() and len(tok) == 6:
            return tok
        else:
            digits.clear()
    return None
