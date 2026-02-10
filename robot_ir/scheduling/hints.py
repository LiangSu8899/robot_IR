"""Scheduling hints for hardware optimization.

These hints guide the IR lowering process for specific hardware targets,
particularly Jetson Thor optimization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


class StreamPartition(Enum):
    """CUDA stream partitioning strategy."""

    SINGLE = auto()  # Single stream
    PER_STAGE = auto()  # One stream per pipeline stage
    PER_MODALITY = auto()  # One stream per modality


@dataclass
class MemoryHint:
    """
    Memory placement and optimization hints.

    Attributes:
        prefer_sram: Whether to prefer SRAM placement.
        streaming_layout: Use streaming memory layout.
        pin_memory: Pin memory for faster transfer.
        placement: Memory placement ("gpu", "cpu", "unified").
        cache_priority: Cache priority (0-100).
    """

    prefer_sram: bool = False
    streaming_layout: bool = False
    pin_memory: bool = True
    placement: str = "gpu"
    cache_priority: int = 50

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "prefer_sram": self.prefer_sram,
            "streaming_layout": self.streaming_layout,
            "pin_memory": self.pin_memory,
            "placement": self.placement,
            "cache_priority": self.cache_priority,
        }


@dataclass
class SchedulingHints:
    """
    Scheduling hints for hardware-aware optimization.

    These hints directly map to optimization passes that can be applied
    during IR lowering. Particularly relevant for Jetson Thor.

    Attributes:
        memory_bound_priority: Prioritize memory-bound optimizations.
        kv_reuse_threshold: Threshold for KV cache reuse (0.0-1.0).
        sensor_fusion_batching: Enable sensor fusion batching.
        kernel_fusion_allowed: Allow kernel fusion.
        stream_partition: CUDA stream partitioning strategy.
        max_concurrent_streams: Maximum concurrent CUDA streams.
        enable_cuda_graphs: Enable CUDA graph capture.
        tensorrt_enabled: Enable TensorRT acceleration.
        memory_hints: Per-component memory hints.

    Example:
        >>> hints = SchedulingHints(
        ...     memory_bound_priority=True,
        ...     kv_reuse_threshold=0.8,
        ...     enable_cuda_graphs=True,
        ...     tensorrt_enabled=True,
        ... )
    """

    memory_bound_priority: bool = True
    kv_reuse_threshold: float = 0.5
    sensor_fusion_batching: bool = True
    kernel_fusion_allowed: bool = True
    stream_partition: StreamPartition = StreamPartition.PER_STAGE
    max_concurrent_streams: int = 4
    enable_cuda_graphs: bool = True
    tensorrt_enabled: bool = True
    memory_hints: dict[str, MemoryHint] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate hints."""
        if not 0.0 <= self.kv_reuse_threshold <= 1.0:
            raise ValueError(f"KV reuse threshold must be in [0, 1], got {self.kv_reuse_threshold}")

    def get_memory_hint(self, component: str) -> MemoryHint:
        """Get memory hint for a component, with default fallback."""
        return self.memory_hints.get(component, MemoryHint())

    def set_memory_hint(self, component: str, hint: MemoryHint) -> None:
        """Set memory hint for a component."""
        self.memory_hints[component] = hint

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "memory_bound_priority": self.memory_bound_priority,
            "kv_reuse_threshold": self.kv_reuse_threshold,
            "sensor_fusion_batching": self.sensor_fusion_batching,
            "kernel_fusion_allowed": self.kernel_fusion_allowed,
            "stream_partition": self.stream_partition.name,
            "max_concurrent_streams": self.max_concurrent_streams,
            "enable_cuda_graphs": self.enable_cuda_graphs,
            "tensorrt_enabled": self.tensorrt_enabled,
            "memory_hints": {k: v.to_dict() for k, v in self.memory_hints.items()},
        }


# Thor-specific presets
THOR_DEFAULT_HINTS = SchedulingHints(
    memory_bound_priority=True,
    kv_reuse_threshold=0.8,
    sensor_fusion_batching=True,
    kernel_fusion_allowed=True,
    stream_partition=StreamPartition.PER_STAGE,
    max_concurrent_streams=4,
    enable_cuda_graphs=True,
    tensorrt_enabled=True,
)

THOR_LOW_LATENCY_HINTS = SchedulingHints(
    memory_bound_priority=True,
    kv_reuse_threshold=0.9,
    sensor_fusion_batching=False,  # Disable for lower latency
    kernel_fusion_allowed=True,
    stream_partition=StreamPartition.PER_MODALITY,
    max_concurrent_streams=8,
    enable_cuda_graphs=True,
    tensorrt_enabled=True,
)
