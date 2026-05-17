import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    YANDEX_SKILL_ID: str = ""
    WEBHOOK_PATH_SECRET: str = "dev"
    LLM_BASE_URL: str = "https://router.example/v1"
    LLM_API_KEY: str = "sk-dev"
    LLM_MODEL: str = "yandexgpt-lite/latest"
    DB_URL: str = "sqlite:///./elk.db"
    LOG_LEVEL: str = "INFO"
    CONFIG_TOML_PATH: str = "config.toml"

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_config() -> dict[str, Any]:
    path = Path(get_settings().CONFIG_TOML_PATH)
    return tomllib.loads(path.read_text(encoding="utf-8"))


def reset_caches() -> None:
    for fn in (get_settings, get_config):
        if hasattr(fn, "cache_clear"):
            fn.cache_clear()
