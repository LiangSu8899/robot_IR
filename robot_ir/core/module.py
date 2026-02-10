"""RobotModule - Top-level IR container."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from robot_ir.sensors.node import SensorNode
    from robot_ir.memory.state import StateBuffer
    from robot_ir.compute.graph import ComputeGraph
    from robot_ir.control.loop import ControlLoop
    from robot_ir.scheduling.hints import SchedulingHints
    from robot_ir.core.config import DeploymentConfig


@dataclass
class RobotModule:
    """
    Top-level Robot IR container.

    This is the root node of a Robot IR graph, containing all components
    needed to represent a complete robot policy execution pipeline.

    Attributes:
        name: Module identifier.
        version: IR version string.
        sensors: List of sensor input nodes.
        states: List of state buffers (KV cache, world model memory, etc.).
        graph: The compute graph containing all neural/logic nodes.
        control: Control loop configuration.
        schedule: Scheduling hints for optimization.
        deploy: Deployment configuration.

    Example:
        >>> module = RobotModule(
        ...     name="pi0_policy",
        ...     sensors=[camera_sensor, proprio_sensor],
        ...     states=[kv_cache, world_latent],
        ...     graph=compute_graph,
        ...     control=ControlLoop(frequency_hz=20.0),
        ... )
    """

    name: str
    version: str = "0.1.0"

    # Input specification
    sensors: list[SensorNode] = field(default_factory=list)

    # State management (key differentiator from ML IRs)
    states: list[StateBuffer] = field(default_factory=list)

    # Compute graph
    graph: ComputeGraph | None = None

    # Control loop specification
    control: ControlLoop | None = None

    # Optimization hints
    schedule: SchedulingHints | None = None

    # Deployment configuration
    deploy: DeploymentConfig | None = None

    def __post_init__(self) -> None:
        """Validate module configuration."""
        if not self.name:
            raise ValueError("Module name is required")

    def add_sensor(self, sensor: SensorNode) -> None:
        """Add a sensor node to the module."""
        self.sensors.append(sensor)

    def add_state(self, state: StateBuffer) -> None:
        """Add a state buffer to the module."""
        self.states.append(state)

    def get_sensor(self, name: str) -> SensorNode | None:
        """Get sensor by name."""
        for sensor in self.sensors:
            if sensor.name == name:
                return sensor
        return None

    def get_state(self, name: str) -> StateBuffer | None:
        """Get state buffer by name."""
        for state in self.states:
            if state.name == name:
                return state
        return None

    def validate(self) -> list[str]:
        """
        Validate the module configuration.

        Returns:
            List of validation error messages. Empty if valid.
        """
        errors = []

        if not self.sensors:
            errors.append("Module must have at least one sensor")

        if self.graph is None:
            errors.append("Module must have a compute graph")

        if self.control is None:
            errors.append("Module must have a control loop configuration")

        return errors

    def to_dict(self) -> dict:
        """Serialize module to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "sensors": [s.to_dict() for s in self.sensors],
            "states": [s.to_dict() for s in self.states],
            "graph": self.graph.to_dict() if self.graph else None,
            "control": self.control.to_dict() if self.control else None,
            "schedule": self.schedule.to_dict() if self.schedule else None,
            "deploy": self.deploy.__dict__ if self.deploy else None,
        }
