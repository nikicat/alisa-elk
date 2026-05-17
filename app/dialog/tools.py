"""Idle-tools surface for the parent dialog graph.

Phase 1 keeps the OpenAI-tool-schema form (passed to
`OpenAIRouterClient.complete(..., tools=[...])`) because the dispatch
path still uses the project-tuned httpx client rather than ChatOpenAI.
The same names are also re-exported as LangChain `@tool` decorators so
they can be bound to a ChatOpenAI in the words subgraph if/when we
unify the LLM client (Phase 2+).
"""

from __future__ import annotations

from langchain_core.tools import tool

from app import persona

# OpenAI-tool-schema definitions ----------------------------------------------

ENTER_GAME_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "enter_game",
        "description": (
            "Вызови, когда пользователь предлагает поиграть в словесную игру: "
            "«давай поиграем в слова», «сыграем в слова», «игра в слова». "
            "Передавай name='words'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "enum": ["words"]},
            },
            "required": ["name"],
        },
    },
}

EXIT_SKILL_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "exit_skill",
        "description": (
            "Вызови, когда пользователь хочет закончить разговор и попрощаться: "
            "«хватит», «выход», «пока», «до свидания», «выйти», «стоп»."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}

HELP_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "help",
        "description": (
            "Вызови, когда пользователь просит помощь, не понимает что делать "
            "или как пользоваться: «помощь», «помоги», «что ты умеешь», "
            "«как пользоваться»."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}

# Toolset advertised to the idle LLM dispatcher.
# `reset_context` lives in `persona.RESET_TOOL` for historical reasons.
IDLE_TOOLS_OPENAI: list[dict] = [
    persona.RESET_TOOL,
    ENTER_GAME_TOOL,
    EXIT_SKILL_TOOL,
    HELP_TOOL,
]


# LangChain-style `@tool` mirrors (used by the words subgraph and any
# future ChatOpenAI-based dispatcher) --------------------------------------


@tool
def reset_context() -> str:
    """Call when the user wants to forget the conversation so far."""
    return "context cleared"


@tool
def exit_skill() -> str:
    """Call when the user wants to end the skill session."""
    return "exiting"


@tool
def help() -> str:  # noqa: A001
    """Call when the user asks for help with the skill."""
    return "help shown"
