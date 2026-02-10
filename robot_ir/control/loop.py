"""Control loop specification for real-time execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


class PipelineMode(Enum):
    """Execution pipeline mode."""

    SYNC = auto()  # Synchronous execution
    ASYNC = auto()  # Asynchronous execution
    STAGGERED = auto()  # Staggered multi-rate execution


@dataclass
class RateSpec:
    """
    Rate specification for a component.

    Attributes:
        name: Component name.
        frequency_hz: Execution frequency in Hz.
        priority: Execution priority (higher = more important).
    """

    name: str
    frequency_hz: float
    priority: int = 0


@dataclass
class ControlLoop:
    """
    Control loop configuration for real-time execution.

    This is a robot-specific abstraction that defines timing constraints
    and execution policies for the policy loop.

    Attributes:
        frequency_hz: Main loop frequency in Hz.
        latency_budget_ms: Maximum allowed latency in milliseconds.
        pipeline_mode: How to pipeline execution.
        jitter_tolerance_ms: Acceptable timing jitter in milliseconds.
        multi_rate: Multi-rate execution specifications.
        deadline_policy: What to do when deadline is missed.

    Example:
        >>> control = ControlLoop(
        ...     frequency_hz=20.0,
        ...     latency_budget_ms=40.0,
        ...     pipeline_mode=PipelineMode.STAGGERED,
        ...     multi_rate=[
        ...         RateSpec("vision", 10.0),
        ...         RateSpec("policy", 20.0),
        ...         RateSpec("control", 100.0),
        ...     ],
        ... )
    """

    frequency_hz: float = 20.0
    latency_budget_ms: float = 50.0
    pipeline_mode: PipelineMode = PipelineMode.SYNC
    jitter_tolerance_ms: float = 5.0
    multi_rate: list[RateSpec] = field(default_factory=list)
    deadline_policy: str = "drop"  # "drop", "block", "warn"

    def __post_init__(self) -> None:
        """Validate control loop configuration."""
        if self.frequency_hz <= 0:
            raise ValueError(f"Frequency must be positive, got {self.frequency_hz}")
        if self.latency_budget_ms <= 0:
            raise ValueError(f"Latency budget must be positive, got {self.latency_budget_ms}")

    @property
    def period_ms(self) -> float:
        """Control loop period in milliseconds."""
        return 1000.0 / self.frequency_hz

    def is_multi_rate(self) -> bool:
        """Whether using multi-rate execution."""
        return len(self.multi_rate) > 0

    def get_component_rate(self, name: str) -> float | None:
        """Get rate for a specific component."""
        for spec in self.multi_rate:
            if spec.name == name:
                return spec.frequency_hz
        return None

    def validate_latency(self) -> bool:
        """Check if latency budget is achievable."""
        return self.latency_budget_ms < self.period_ms

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "frequency_hz": self.frequency_hz,
            "latency_budget_ms": self.latency_budget_ms,
            "pipeline_mode": self.pipeline_mode.name,
            "jitter_tolerance_ms": self.jitter_tolerance_ms,
            "multi_rate": [
                {"name": r.name, "frequency_hz": r.frequency_hz, "priority": r.priority}
                for r in self.multi_rate
            ],
            "deadline_policy": self.deadline_policy,
        }
