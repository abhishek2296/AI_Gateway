"""
Model registry metadata layer — catalog types, validation, and storage contract.

Typical import::

    from src.registry import BaseModelRegistry, MemoryModelRegistry, ModelInfo, ProviderType
"""

from src.core.enums import ProviderType
from src.registry.base import BaseModelRegistry, CatalogModelRegistry
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
from src.registry.metrics import RegistryMetrics
from src.registry.models import ModelCapability, ModelInfo
from src.registry.read_only import ReadOnlyModelRegistry, ReadOnlyModelRegistryView

__all__ = [
    "AmbiguousModelError",
    "BaseModelRegistry",
    "CatalogModelRegistry",
    "InvalidModelMetadataError",
    "MemoryModelRegistry",
    "ModelAlreadyRegisteredError",
    "ModelCapability",
    "ModelDisabledError",
    "ModelInfo",
    "ModelListFilters",
    "ModelNotFoundError",
    "ProviderType",
    "ReadOnlyModelRegistry",
    "ReadOnlyModelRegistryView",
    "RegistryError",
    "RegistryMetrics",
]
