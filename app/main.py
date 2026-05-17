import logging
from contextlib import asynccontextmanager
from typing import Annotated

import structlog
from fastapi import Depends, FastAPI, HTTPException, Path

from app import repo
from app.config import get_settings
from app.db import init_engine, session_scope, get_session_factory
from app.handler import HandlerDeps, route
from app.llm import OpenAIRouterClient
from app.schemas import AliceRequest, AliceResponse
from app.wait import get_registry


def configure_logging(level: str) -> None:
    logging.basicConfig(level=level)
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ]
    )


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
    )


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    settings = get_settings()
    configure_logging(settings.LOG_LEVEL)
    init_engine()
    with session_scope() as db:
        cleared = repo.mark_orphaned_in_progress_as_error(db)
        db.commit()
    if cleared:
        structlog.get_logger().info("startup_marked_orphans", count=cleared)
    yield
    # Best-effort: cancel any background tasks.
    get_registry().cancel_all()
    if hasattr(get_handler_deps, "_llm"):
        await get_handler_deps._llm.aclose()  # type: ignore[attr-defined]


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None)


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
