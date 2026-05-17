from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Meta(BaseModel):
    model_config = ConfigDict(extra="allow")
    locale: str = "ru-RU"
    timezone: str = "UTC"
    client_id: str = ""
    interfaces: dict[str, Any] = Field(default_factory=dict)


class Application(BaseModel):
    model_config = ConfigDict(extra="allow")
    application_id: str


class SessionUser(BaseModel):
    model_config = ConfigDict(extra="allow")
    user_id: str | None = None


class Session(BaseModel):
    model_config = ConfigDict(extra="allow")
    message_id: int
    session_id: str
    skill_id: str
    new: bool
    application: Application
    user: SessionUser | None = None


class NluToken(BaseModel):
    model_config = ConfigDict(extra="allow")


class Nlu(BaseModel):
    model_config = ConfigDict(extra="allow")
    tokens: list[str] = Field(default_factory=list)
    entities: list[dict[str, Any]] = Field(default_factory=list)
    intents: dict[str, Any] = Field(default_factory=dict)


class Request(BaseModel):
    model_config = ConfigDict(extra="allow")
    command: str = ""
    original_utterance: str = ""
    type: str = "SimpleUtterance"
    markup: dict[str, Any] = Field(default_factory=dict)
    nlu: Nlu = Field(default_factory=Nlu)


class State(BaseModel):
    model_config = ConfigDict(extra="allow")
    session: dict[str, Any] = Field(default_factory=dict)
    user: dict[str, Any] = Field(default_factory=dict)
    application: dict[str, Any] = Field(default_factory=dict)


class AliceRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    meta: Meta = Field(default_factory=Meta)
    session: Session
    request: Request
    state: State = Field(default_factory=State)
    version: str = "1.0"


class Button(BaseModel):
    title: str
    hide: bool = True
    payload: dict[str, Any] | None = None
    url: str | None = None


class Response(BaseModel):
    text: str
    tts: str | None = None
    buttons: list[Button] = Field(default_factory=list)
    end_session: bool = False


class AliceResponse(BaseModel):
    response: Response
    session_state: dict[str, Any] = Field(default_factory=dict)
    user_state_update: dict[str, Any] | None = None
    version: str = "1.0"
