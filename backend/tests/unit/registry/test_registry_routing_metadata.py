"""Additional tests for ModelInfo routing metadata and supports()."""

from __future__ import annotations

import pytest

from src.core.enums import ProviderType
from src.registry.exceptions import InvalidModelMetadataError
from src.registry.models import ModelCapability, ModelInfo


def test_model_supports_streaming_capability() -> None:
    model = ModelInfo(
        name="stream-model",
        provider=ProviderType.OLLAMA,
        display_name="Stream",
        description="",
        capabilities=frozenset({ModelCapability.CHAT, ModelCapability.STREAMING}),
    )

    assert model.supports(ModelCapability.STREAMING) is True
    assert model.supports(ModelCapability.VISION) is False


def test_model_info_accepts_optional_routing_metadata() -> None:
    model = ModelInfo(
        name="routed-model",
        provider=ProviderType.OPENAI,
        display_name="Routed",
        description="",
        capabilities=frozenset({ModelCapability.CHAT}),
        priority=10,
        cost_per_input_token=0.000003,
        cost_per_output_token=0.000015,
        latency_score=8,
        quality_score=9,
        tags=frozenset({"fast", "premium"}),
    )

    assert model.priority == 10
    assert model.latency_score == 8
    assert model.tags == frozenset({"fast", "premium"})


def test_model_info_works_without_routing_metadata() -> None:
    model = ModelInfo(
        name="basic-model",
        provider=ProviderType.OLLAMA,
        display_name="Basic",
        description="",
        capabilities=frozenset({ModelCapability.CHAT}),
    )

    assert model.priority is None
    assert model.cost_per_input_token is None
    assert model.tags == frozenset()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("priority", -1),
        ("cost_per_input_token", -0.01),
        ("latency_score", 0),
        ("quality_score", 11),
    ],
)
def test_model_info_rejects_invalid_routing_metadata(field: str, value: object) -> None:
    kwargs = {
        "name": "bad-model",
        "provider": ProviderType.OPENAI,
        "display_name": "Bad",
        "description": "",
        "capabilities": frozenset({ModelCapability.CHAT}),
        field: value,
    }
    with pytest.raises(InvalidModelMetadataError):
        ModelInfo(**kwargs)
