"""Sensor node definitions for multi-modal input pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from robot_ir.core.types import TensorShape
    from robot_ir.compute.nodes import ComputeNode


class Modality(Enum):
    """Sensor modality types."""

    VISION = auto()  # Camera images
    AUDIO = auto()  # Microphone input
    IMU = auto()  # Inertial measurement unit
    TACTILE = auto()  # Touch/force sensors
    LIDAR = auto()  # LiDAR point clouds
    PROPRIO = auto()  # Proprioceptive state (joint positions, velocities)
    DEPTH = auto()  # Depth camera
    THERMAL = auto()  # Thermal camera


class BufferType(Enum):
    """Buffer strategy for sensor data."""

    STREAM = auto()  # Continuous streaming (latest frame)
    SLIDING_WINDOW = auto()  # Keep N most recent frames
    EVENT = auto()  # Event-triggered buffering


@dataclass
class BufferPolicy:
    """
    Buffer policy for sensor data management.

    Attributes:
        type: Buffer strategy type.
        window_size: Number of frames to keep (for SLIDING_WINDOW).
        reuse_allowed: Whether buffer can be reused across steps.
        drop_policy: What to do when buffer is full.
    """

    type: BufferType = BufferType.STREAM
    window_size: int = 1
    reuse_allowed: bool = True
    drop_policy: str = "oldest"  # "oldest" or "newest"

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "type": self.type.name,
            "window_size": self.window_size,
            "reuse_allowed": self.reuse_allowed,
            "drop_policy": self.drop_policy,
        }


@dataclass
class SensorNode:
    """
    Sensor input node representing a single sensor stream.

    This node represents the entry point for sensor data into the
    robot policy execution graph.

    Attributes:
        name: Unique sensor identifier.
        modality: Type of sensor (vision, audio, IMU, etc.).
        shape: Output tensor shape.
        rate_hz: Sensor sampling rate in Hz.
        preprocessing: List of preprocessing operations.
        buffering: Buffer policy for managing sensor data.
        latency_ms: Expected sensor latency in milliseconds.

    Example:
        >>> camera = SensorNode(
        ...     name="camera_front",
        ...     modality=Modality.VISION,
        ...     shape=TensorShape((3, 224, 224)),
        ...     rate_hz=30.0,
        ... )
    """

    name: str
    modality: Modality
    shape: TensorShape
    rate_hz: float = 30.0
    preprocessing: list[ComputeNode] = field(default_factory=list)
    buffering: BufferPolicy = field(default_factory=BufferPolicy)
    latency_ms: float = 0.0

    def __post_init__(self) -> None:
        """Validate sensor configuration."""
        if self.rate_hz <= 0:
            raise ValueError(f"Sensor rate must be positive, got {self.rate_hz}")

    @property
    def period_ms(self) -> float:
        """Sensor period in milliseconds."""
        return 1000.0 / self.rate_hz

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "modality": self.modality.name,
            "shape": {
                "dims": self.shape.dims,
                "dtype": self.shape.dtype.name,
                "layout": self.shape.layout,
            },
            "rate_hz": self.rate_hz,
            "buffering": self.buffering.to_dict(),
            "latency_ms": self.latency_ms,
        }
