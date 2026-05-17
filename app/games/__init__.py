"""Game subskills for the elk skill.

Each game module exposes a LangGraph subgraph plus the tool definitions
that drive entry/exit, so `handler.py` can dispatch into it when the
in-idle LLM emits the matching tool call. For now only `words` is
implemented; the dict here is the registry future games will plug into.
"""

from app.games import words

GAMES = {
    "words": words,
}

__all__ = ["GAMES", "words"]
