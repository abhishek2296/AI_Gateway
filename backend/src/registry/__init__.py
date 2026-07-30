"""
Model registry metadata layer — catalog types, validation, and storage contract.

Typical import::

    from src.registry import BaseModelRegistry, MemoryModelRegistry, ModelInfo, ProviderType
"""

from src.registry.base import BaseModelRegistry
from src.registry.exceptions import (
    AmbiguousModelError,
    InvalidModelMetadataError,
    ModelAlreadyRegisteredError,
    ModelDisabledError,
    ModelNotFoundError,
    RegistryError,
)
from src.registry.filters import ModelListFilters
from src.registry.memory import MemoryModelRegistry
from src.registry.models import ModelCapability, ModelInfo, ProviderType

__all__ = [
    "AmbiguousModelError",
    "BaseModelRegistry",
    "InvalidModelMetadataError",
    "MemoryModelRegistry",
    "ModelAlreadyRegisteredError",
    "ModelCapability",
    "ModelDisabledError",
    "ModelInfo",
    "ModelListFilters",
    "ModelNotFoundError",
    "ProviderType",
    "RegistryError",
]
