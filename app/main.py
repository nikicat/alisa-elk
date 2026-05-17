import logging
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Annotated

import structlog
from fastapi import Depends, FastAPI, HTTPException, Path, Request
from fastapi.responses import JSONResponse
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.config import get_settings
from app.db import get_session_factory, init_engine
from app.dialog.graph import build_parent_graph
from app.handler import HandlerDeps, route
from app.llm import OpenAIRouterClient
from app.schemas import AliceRequest, AliceResponse
from app.wait import get_registry

# CHECKPOINT SCHEMA: v1.2.x  (langgraph + langgraph-checkpoint-sqlite v3.x)
DIALOG_DB_PATH = "data/dialog.db"


def configure_logging(level: str) -> None:
    logging.basicConfig(level=level)
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.LogfmtRenderer(),
        ]
    )


_lifespan_state: dict = {}


def get_handler_deps() -> HandlerDeps:
    settings = get_settings()
    if not hasattr(get_handler_deps, "_llm"):
        get_handler_deps._llm = OpenAIRouterClient(  # type: ignore[attr-defined]
            base_url=settings.LLM_BASE_URL,
            api_key=settings.LLM_API_KEY,
            model=settings.LLM_MODEL,
        )
    return HandlerDeps(
        session_factory=get_session_factory(),
        llm=get_handler_deps._llm,  # type: ignore[attr-defined]
        registry=get_registry(),
        graph=_lifespan_state.get("graph"),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    settings = get_settings()
    configure_logging(settings.LOG_LEVEL)
    init_engine()

    async with AsyncExitStack() as stack:
        saver = await stack.enter_async_context(
            AsyncSqliteSaver.from_conn_string(DIALOG_DB_PATH)
        )
        _lifespan_state["graph"] = build_parent_graph(saver)
        try:
            yield
        finally:
            _lifespan_state.pop("graph", None)
            get_registry().cancel_all()
            if hasattr(get_handler_deps, "_llm"):
                await get_handler_deps._llm.aclose()  # type: ignore[attr-defined]


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None)


@app.middleware("http")
async def skill_id_gate(request: Request, call_next):
    """Reject Alice requests carrying the wrong skill_id before they
    enter the graph. Keeps checkpointer state clean of probe traffic."""
    if not request.url.path.startswith("/webhook/"):
        return await call_next(request)
    skill_id = get_settings().YANDEX_SKILL_ID
    if not skill_id:
        return await call_next(request)
    # Lightweight peek: read the body, validate, then restore the stream
    # so FastAPI's downstream parser sees an untouched request.
    body = await request.body()

    async def _replay():  # noqa: ANN202
        return {"type": "http.request", "body": body, "more_body": False}

    request._receive = _replay  # type: ignore[attr-defined]
    try:
        import json

        payload = json.loads(body)
        got = payload.get("session", {}).get("skill_id")
    except Exception:
        return await call_next(request)
    if got and got != skill_id:
        structlog.get_logger().warning("skill_id_mismatch", got=got, expected=skill_id)
        return JSONResponse(
            content={
                "response": {
                    "text": "Лес не узнаёт тебя.",
                    "tts": "Лес не узнаёт тебя.",
                    "end_session": True,
                },
                "session_state": {},
                "version": "1.0",
            },
        )
    return await call_next(request)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"ok": "1"}


@app.post("/webhook/{secret}", response_model=None)
async def webhook(
    secret: Annotated[str, Path(min_length=1, max_length=128)],
    payload: AliceRequest,
    deps: Annotated[HandlerDeps, Depends(get_handler_deps)],
) -> AliceResponse:
    if secret != get_settings().WEBHOOK_PATH_SECRET:
        raise HTTPException(status_code=404)
    try:
        return await route(payload, deps)
    except Exception as exc:
        structlog.get_logger().error("handler_unhandled", error=str(exc))
        raise
