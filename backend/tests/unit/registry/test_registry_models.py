"""
Unit tests for registry model metadata.

Each test targets one specific rule enforced by ``ModelInfo.__post_init__``
(see ``src/registry/models.py``). Keeping one behavior per test makes it easy
to tell exactly which validation rule broke if a test fails.
"""

from __future__ import annotations

import pytest

from src.core.enums import ProviderType
from src.registry.exceptions import InvalidModelMetadataError
from src.registry.models import ModelCapability, ModelInfo


def test_model_info_accepts_valid_metadata() -> None:
    """
    A fully-populated, well-formed ``ModelInfo`` should construct without error.

    This is the "happy path" test: every field is given a valid value
    (including the optional ones like ``context_window`` and ``metadata``),
    and we confirm both that construction succeeds and that ``supports()``
    correctly reports capabilities that were/weren't included.
    """
    model = ModelInfo(
        name="gpt-4o",
        provider=ProviderType.OPENAI,
        display_name="GPT-4o",
        description="General-purpose multimodal model.",
        capabilities=frozenset(
            {
                ModelCapability.CHAT,
                ModelCapability.VISION,
                ModelCapability.STREAMING,
                ModelCapability.FUNCTION_CALLING,
            },
        ),
        context_window=128_000,
        max_output_tokens=16_384,
        enabled=True,
        metadata={"family": "gpt-4"},
    )

    assert model.name == "gpt-4o"
    assert model.provider is ProviderType.OPENAI
    # VISION was included in `capabilities` above, so supports() must be True.
    assert model.supports(ModelCapability.VISION) is True
    # EMBEDDING was NOT included, so supports() must be False.
    assert model.supports(ModelCapability.EMBEDDING) is False


def test_model_info_strips_name_and_display_name() -> None:
    """
    Leading/trailing whitespace in ``name``/``display_name`` should be trimmed.

    Example: passing ``"  qwen3:8b  "`` should result in the stored value
    being the clean string ``"qwen3:8b"``, so downstream code (e.g. provider
    API calls that use ``model.name`` as the literal model identifier) never
    has to worry about stray whitespace breaking a lookup.
    """
    model = ModelInfo(
        name="  qwen3:8b  ",
        provider=ProviderType.OLLAMA,
        display_name="  Qwen3 8B  ",
        description="Local chat model.",
        capabilities=frozenset({ModelCapability.CHAT, ModelCapability.STREAMING}),
    )

    assert model.name == "qwen3:8b"
    assert model.display_name == "Qwen3 8B"


@pytest.mark.parametrize(
    ("kwargs", "field"),
    [
        # name: empty string and whitespace-only string are both invalid.
        ({"name": ""}, "name"),
        ({"name": "   "}, "name"),
        # display_name: same "must be non-empty" rule as name.
        ({"display_name": ""}, "display_name"),
        # context_window: must be strictly positive (zero and negative are invalid).
        ({"context_window": 0}, "context_window"),
        ({"context_window": -1}, "context_window"),
        # max_output_tokens: same "must be > 0" rule as context_window.
        ({"max_output_tokens": 0}, "max_output_tokens"),
    ],
)
def test_model_info_rejects_invalid_values(kwargs: dict, field: str) -> None:
    """
    Each invalid value listed above should raise ``InvalidModelMetadataError``
    with ``.field`` pointing at exactly the field that was invalid.

    This is a parametrized test: it runs once per ``(kwargs, field)`` pair in
    the list above. ``kwargs`` overrides one field of an otherwise-valid base
    payload, and ``field`` is the name we expect to see on the raised
    exception's ``.field`` attribute — confirming the error correctly
    identifies *which* field was the problem, not just that something failed.
    """
    base = {
        "name": "test-model",
        "provider": ProviderType.OLLAMA,
        "display_name": "Test Model",
        "description": "Test description.",
        "capabilities": frozenset({ModelCapability.CHAT}),
    }
    base.update(kwargs)

    with pytest.raises(InvalidModelMetadataError) as exc_info:
        ModelInfo(**base)

    assert exc_info.value.field == field


def test_model_info_rejects_invalid_capabilities_type() -> None:
    """
    ``capabilities`` must be a ``frozenset``, not a regular mutable ``set``.

    ``ModelInfo`` is meant to be an immutable value object. Allowing a plain
    ``set`` would let external code mutate ``capabilities`` after the fact
    (e.g. ``model.capabilities.add(...)``), silently breaking the "frozen"
    guarantee. This test confirms passing a plain ``set`` is rejected with a
    message that mentions "frozenset" so the fix is obvious to the caller.
    """
    with pytest.raises(InvalidModelMetadataError, match="frozenset"):
        ModelInfo(
            name="test-model",
            provider=ProviderType.OPENAI,
            display_name="Test Model",
            description="Test description.",
            capabilities={ModelCapability.CHAT},  # type: ignore[arg-type]
        )


def test_model_info_allows_empty_description_and_capabilities() -> None:
    """
    An empty ``description`` and an empty ``capabilities`` set are both valid.

    These fields are allowed to be "empty" (as opposed to ``name``/
    ``display_name``, which must be non-empty) because a model may be added
    to the catalog as a placeholder before its full metadata is known, or
    simply may not have capability flags populated yet. ``enabled=False`` is
    also exercised here to confirm the boolean flag round-trips correctly.
    """
    model = ModelInfo(
        name="placeholder",
        provider=ProviderType.GEMINI,
        display_name="Placeholder",
        description="",
        capabilities=frozenset(),
        enabled=False,
    )

    assert model.description == ""
    assert model.capabilities == frozenset()
    assert model.enabled is False
