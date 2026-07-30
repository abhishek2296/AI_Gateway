"""Unit tests for provider/ORM to registry mappers."""

from __future__ import annotations

from src.models.ai_model import AIModel
from src.models.provider import Provider
from src.providers.base import ModelInfo as ProviderModelInfo
from src.registry.mappers import map_orm_model_to_registry, map_provider_model_to_registry
from src.registry.models import ModelCapability, ProviderType


def test_map_provider_model_to_registry() -> None:
    mapped = map_provider_model_to_registry(
        ProviderType.OPENAI,
        ProviderModelInfo(
            id="gpt-4o",
            name="gpt-4o",
            supports_streaming=True,
            supports_tools=True,
            supports_vision=True,
        ),
    )
    assert mapped.provider is ProviderType.OPENAI
    assert ModelCapability.STREAMING in mapped.capabilities
    assert ModelCapability.FUNCTION_CALLING in mapped.capabilities


def test_map_orm_model_to_registry() -> None:
    provider = Provider(
        id=1,
        name="ollama",
        display_name="Ollama",
        provider_type="ollama",
    )
    ai_model = AIModel(
        provider_id=1,
        model_name="qwen3:8b",
        display_name="Qwen3 8B",
        supports_streaming=True,
        is_default=True,
        is_active=True,
    )
    mapped = map_orm_model_to_registry(provider, ai_model)
    assert mapped.metadata.get("is_default") is True
    assert ModelCapability.STREAMING in mapped.capabilities
