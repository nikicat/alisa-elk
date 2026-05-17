"""Pytest fixtures: in-memory SQLite, configured FastAPI app, MockLLM injection."""

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import MemorySaver
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import config as cfg_mod
from app import db as db_mod
from app import main as main_mod
from app.dialog.graph import build_parent_graph
from app.handler import HandlerDeps
from app.models import Base
from app.tests.mock_alice import AliceSession
from app.tests.mock_llm import MockLLMClient
from app.wait import PendingTaskRegistry, reset_registry_for_tests

TEST_SKILL_ID = "test-skill-id"
TEST_WEBHOOK_SECRET = "test-secret"


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("YANDEX_SKILL_ID", TEST_SKILL_ID)
    monkeypatch.setenv("WEBHOOK_PATH_SECRET", TEST_WEBHOOK_SECRET)
    monkeypatch.setenv("LLM_BASE_URL", "http://unused")
    monkeypatch.setenv("LLM_API_KEY", "unused")
    monkeypatch.setenv("LLM_MODEL", "mock")
    monkeypatch.setenv("DB_URL", "sqlite:///:memory:")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    monkeypatch.setenv("CONFIG_TOML_PATH", "config.toml")
    cfg_mod.reset_caches()
    yield
    cfg_mod.reset_caches()


@pytest.fixture
def engine():
    """Shared in-memory SQLite engine — required because each connection sees its
    own database otherwise."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def session_factory(engine):
    SessionLocal = sessionmaker(
        bind=engine, autoflush=False, autocommit=False, expire_on_commit=False
    )
    return SessionLocal


@pytest.fixture
def db(session_factory):
    db = session_factory()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def mock_llm():
    return MockLLMClient()


@pytest.fixture
def registry():
    reset_registry_for_tests()
    reg = PendingTaskRegistry()
    yield reg
    reg.cancel_all()


@pytest.fixture
def dialog_graph():
    """Fresh compiled parent graph with an in-memory checkpointer per-test."""
    return build_parent_graph(MemorySaver())


@pytest.fixture
def deps(session_factory, mock_llm, registry, dialog_graph):
    return HandlerDeps(
        session_factory=session_factory,
        llm=mock_llm,
        registry=registry,
        graph=dialog_graph,
    )


@pytest.fixture
def app_client(monkeypatch, session_factory, mock_llm, registry, dialog_graph):
    """FastAPI TestClient with mocked LLM, in-memory DB, isolated registry."""

    def fake_init_engine(db_url=None):  # noqa: ARG001
        return session_factory.kw["bind"]

    monkeypatch.setattr(db_mod, "init_engine", fake_init_engine)
    monkeypatch.setattr(db_mod, "get_session_factory", lambda: session_factory)

    def fake_get_handler_deps() -> HandlerDeps:
        return HandlerDeps(
            session_factory=session_factory,
            llm=mock_llm,
            registry=registry,
            graph=dialog_graph,
        )

    main_mod.app.dependency_overrides[main_mod.get_handler_deps] = fake_get_handler_deps

    with TestClient(main_mod.app) as client:
        yield client

    main_mod.app.dependency_overrides.clear()


@pytest.fixture
def alice(app_client) -> AliceSession:
    return AliceSession(
        client=app_client,
        webhook_path=f"/webhook/{TEST_WEBHOOK_SECRET}",
        skill_id=TEST_SKILL_ID,
    )
