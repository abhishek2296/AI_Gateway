from pydantic import BaseModel
from src.core.enums import Provider
from src.core.config import settings

class ChatRequest(BaseModel):
    message: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "message": "Explain FastAPI in one sentence."
            }
        }
    }


class ChatResponse(BaseModel):
    response: str
    model: str
    provider: Provider

class ChatResponse(BaseModel):

    response: str
    model: str
    provider: Provider

    model_config = {
        "json_schema_extra": {
            "example": {
                "response": "FastAPI is a modern Python framework for building APIs.",
                "model": "qwen3:8b",
                "provider": "ollama"
            }
        }
    }