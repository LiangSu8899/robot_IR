"""State buffer definitions - Core innovation of Robot IR.

This module defines how stateful memory (KV cache, world model latent,
belief state, proprio history) is managed across execution steps.
This is the key differentiator from standard ML IRs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from robot_ir.core.types import TensorShape


class Persistence(Enum):
    """State persistence scope."""

    STEP = auto()  # Reset every step
    EPISODE = auto()  # Persist across episode
    GLOBAL = auto()  # Persist across episodes (e.g., learned priors)


class ReuseStrategy(Enum):
    """State reuse strategy for optimization."""

    KV_CACHE = auto()  # Transformer KV cache reuse
    TEMPORAL_STATE = auto()  # RNN/LSTM hidden state
    WORLD_MODEL_MEMORY = auto()  # World model latent memory
    ACTION_BUFFER = auto()  # Action chunking buffer
    BELIEF_STATE = auto()  # Probabilistic belief state
    PROPRIO_HISTORY = auto()  # Proprioceptive history window


class EvictionPolicy(Enum):
    """Cache eviction policy."""

    LRU = auto()  # Least recently used
    TIME_DECAY = auto()  # Time-based decay
    CONFIDENCE_DECAY = auto()  # Confidence-weighted decay
    FIFO = auto()  # First in, first out


@dataclass
class StateBuffer:
    """
    State buffer for managing persistent memory across execution.

    This is the core abstraction that enables efficient memory reuse
    in real-time robot policy execution.

    Attributes:
        name: Unique buffer identifier.
        shape: Buffer tensor shape.
        persistence: How long state persists.
        reuse_strategy: Strategy for reusing state.
        eviction_policy: How to evict old entries.
        max_size: Maximum buffer size (for variable-length buffers).
        prefetch: Whether to prefetch this buffer.
        placement: Memory placement hint ("gpu", "cpu", "unified").

    Example:
        >>> kv_cache = StateBuffer(
        ...     name="transformer_kv",
        ...     shape=TensorShape((32, 2, 512, 64)),  # layers, kv, seq, head_dim
        ...     persistence=Persistence.STEP,
        ...     reuse_strategy=ReuseStrategy.KV_CACHE,
        ... )
    """

    name: str
    shape: TensorShape
    persistence: Persistence = Persistence.STEP
    reuse_strategy: ReuseStrategy = ReuseStrategy.KV_CACHE
    eviction_policy: EvictionPolicy = EvictionPolicy.LRU
    max_size: int | None = None
    prefetch: bool = False
    placement: str = "gpu"

    def __post_init__(self) -> None:
        """Validate buffer configuration."""
        if not self.name:
            raise ValueError("Buffer name is required")
        if self.placement not in ("gpu", "cpu", "unified"):
            raise ValueError(f"Invalid placement: {self.placement}")

    @property
    def is_kv_cache(self) -> bool:
        """Whether this is a KV cache buffer."""
        return self.reuse_strategy == ReuseStrategy.KV_CACHE

    @property
    def is_temporal(self) -> bool:
        """Whether this buffer holds temporal state."""
        return self.reuse_strategy in (
            ReuseStrategy.TEMPORAL_STATE,
            ReuseStrategy.PROPRIO_HISTORY,
        )

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "shape": {
                "dims": self.shape.dims,
                "dtype": self.shape.dtype.name,
            },
            "persistence": self.persistence.name,
            "reuse_strategy": self.reuse_strategy.name,
            "eviction_policy": self.eviction_policy.name,
            "max_size": self.max_size,
            "prefetch": self.prefetch,
            "placement": self.placement,
        }


@dataclass
class KVCacheSpec(StateBuffer):
    """
    Specialized KV cache specification for transformer models.

    Attributes:
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        head_dim: Dimension per head.
        max_seq_len: Maximum sequence length.
        sliding_window: Sliding window size (None for full attention).
    """

    num_layers: int = 24
    num_heads: int = 16
    head_dim: int = 64
    max_seq_len: int = 2048
    sliding_window: int | None = None

    def __post_init__(self) -> None:
        """Set up KV cache shape."""
        super().__post_init__()
        self.reuse_strategy = ReuseStrategy.KV_CACHE
        # Shape: (layers, 2, seq, heads, head_dim)
        # 2 for key and value


@dataclass
class WorldModelMemory(StateBuffer):
    """
    World model latent memory specification.

    Attributes:
        latent_dim: Dimension of world model latent.
        compression_ratio: Compression ratio from observation.
        rollout_steps: Number of rollout steps supported.
    """

    latent_dim: int = 512
    compression_ratio: float = 1.0
    rollout_steps: int = 1

    def __post_init__(self) -> None:
        """Set up world model memory."""
        super().__post_init__()
        self.reuse_strategy = ReuseStrategy.WORLD_MODEL_MEMORY
        self.persistence = Persistence.EPISODE
