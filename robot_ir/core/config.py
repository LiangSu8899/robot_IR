"""Deployment configuration for Robot IR."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


class TargetPlatform(Enum):
    """Supported deployment targets."""

    THOR = auto()  # Jetson Thor
    ORIN = auto()  # Jetson Orin
    X86 = auto()  # x86 workstation
    SIM = auto()  # Simulation environment


@dataclass
class DeploymentConfig:
    """
    Deployment configuration for the robot module.

    Attributes:
        target: Target hardware platform.
        realtime: Whether real-time guarantees are required.
        multi_gpu: Whether to enable multi-GPU execution.
        ros_bridge: Whether to enable ROS2 bridge.
        max_memory_mb: Maximum GPU memory budget in MB.
        enable_profiling: Whether to enable performance profiling.
    """

    target: TargetPlatform = TargetPlatform.THOR
    realtime: bool = True
    multi_gpu: bool = False
    ros_bridge: bool = False
    max_memory_mb: int | None = None
    enable_profiling: bool = False

    def is_jetson(self) -> bool:
        """Whether target is a Jetson device."""
        return self.target in (TargetPlatform.THOR, TargetPlatform.ORIN)

    def is_edge(self) -> bool:
        """Whether target is an edge device."""
        return self.is_jetson()
