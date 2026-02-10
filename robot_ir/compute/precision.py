"""Precision policy for neural network computation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


class Precision(Enum):
    """Numerical precision levels."""

    FP32 = auto()  # Full precision
    FP16 = auto()  # Half precision
    BF16 = auto()  # Brain floating point
    FP8_E4M3 = auto()  # 8-bit floating point (e4m3)
    FP8_E5M2 = auto()  # 8-bit floating point (e5m2)
    INT8 = auto()  # 8-bit integer
    INT4 = auto()  # 4-bit integer (weights only)
    MIXED = auto()  # Mixed precision


@dataclass
class PrecisionPolicy:
    """
    Precision policy for model execution.

    Allows fine-grained control over which layers use which precision,
    enabling aggressive quantization while preserving accuracy where needed.

    Attributes:
        preferred: Default precision for the model.
        compute_precision: Precision for compute operations.
        storage_precision: Precision for weight storage.
        critical_layers: Layers that must use higher precision.
        quantization_config: Additional quantization parameters.

    Example:
        >>> policy = PrecisionPolicy(
        ...     preferred=Precision.FP8_E4M3,
        ...     critical_layers=["action_head", "final_norm"],
        ... )
    """

    preferred: Precision = Precision.FP16
    compute_precision: Precision | None = None
    storage_precision: Precision | None = None
    critical_layers: list[str] = field(default_factory=list)
    quantization_config: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Set defaults."""
        if self.compute_precision is None:
            self.compute_precision = self.preferred
        if self.storage_precision is None:
            self.storage_precision = self.preferred

    def is_quantized(self) -> bool:
        """Whether model uses quantization."""
        return self.preferred in (
            Precision.INT8,
            Precision.INT4,
            Precision.FP8_E4M3,
            Precision.FP8_E5M2,
        )

    def get_layer_precision(self, layer_name: str) -> Precision:
        """Get precision for a specific layer."""
        if layer_name in self.critical_layers:
            return Precision.FP16  # Keep critical layers in higher precision
        return self.preferred

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "preferred": self.preferred.name,
            "compute_precision": self.compute_precision.name if self.compute_precision else None,
            "storage_precision": self.storage_precision.name if self.storage_precision else None,
            "critical_layers": self.critical_layers,
            "quantization_config": self.quantization_config,
        }
