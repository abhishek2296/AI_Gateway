from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_DIR.parent


class Settings(BaseSettings):
    # Application
    APP_NAME: str = "AI Gateway"
    APP_VERSION: str = "0.1.0"
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Ollama
    OLLAMA_HOST: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen3:8b"

    # Logging
    LOG_LEVEL: str = "INFO"

    TIMEOUT: int = 60  # Timeout for requests in seconds

    # Database
    DATABASE_URL: str
    TEST_DATABASE_URL: str | None = None
    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 10
    DB_ECHO: bool = False

    model_config = SettingsConfigDict(
        # Root settings are shared by Docker and the application. The backend file
        # remains available for backend-specific overrides such as Ollama settings.
        env_file=(PROJECT_ROOT / ".env", BACKEND_DIR / ".env"),
        extra="ignore"
    )

@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings()
