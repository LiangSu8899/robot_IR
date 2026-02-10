"""Multi-rate execution infrastructure for robot policies.

Handles different execution frequencies for vision, policy, and control loops.
Provides rate conversion buffers, async sensor handling, and pipeline scheduling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Callable
import math

if TYPE_CHECKING:
    from robot_ir.core.module import RobotModule
    from robot_ir.core.types import TensorShape


class InterpolationMode(Enum):
    """Interpolation mode for rate conversion."""

    ZERO_ORDER_HOLD = auto()  # Hold last value (most common for actions)
    LINEAR = auto()  # Linear interpolation
    CUBIC = auto()  # Cubic spline interpolation
    SLERP = auto()  # Spherical linear interpolation (for quaternions)


class ExtrapolationMode(Enum):
    """Extrapolation mode when data is not available."""

    HOLD = auto()  # Hold last value
    ZERO = auto()  # Output zeros
    PREDICT = auto()  # Use prediction model
    FAIL = auto()  # Raise error


class SensorSyncMode(Enum):
    """Synchronization mode for sensors."""

    LATEST = auto()  # Always use latest available
    SYNC = auto()  # Wait for all sensors
    TIMESTAMP = auto()  # Sync by timestamp


@dataclass
class RateConversionBuffer:
    """
    Buffer for rate conversion between components.

    Handles upsampling and downsampling between different execution rates.

    Attributes:
        name: Buffer identifier.
        source_rate_hz: Source update frequency.
        target_rate_hz: Target consumption frequency.
        buffer_size: Number of samples to buffer.
        interpolation: Interpolation mode for upsampling.
        extrapolation: Extrapolation mode for missing data.
        max_staleness_ms: Maximum allowed data staleness.

    Example:
        >>> # Vision at 10Hz feeding policy at 20Hz (upsampling)
        >>> buffer = RateConversionBuffer(
        ...     name="vision_to_policy",
        ...     source_rate_hz=10.0,
        ...     target_rate_hz=20.0,
        ...     interpolation=InterpolationMode.ZERO_ORDER_HOLD,
        ... )
    """

    name: str
    source_rate_hz: float
    target_rate_hz: float
    buffer_size: int = 4
    interpolation: InterpolationMode = InterpolationMode.ZERO_ORDER_HOLD
    extrapolation: ExtrapolationMode = ExtrapolationMode.HOLD
    max_staleness_ms: float = 100.0

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.source_rate_hz <= 0:
            raise ValueError(f"Source rate must be positive: {self.source_rate_hz}")
        if self.target_rate_hz <= 0:
            raise ValueError(f"Target rate must be positive: {self.target_rate_hz}")

    @property
    def is_upsampling(self) -> bool:
        """Whether this buffer performs upsampling."""
        return self.target_rate_hz > self.source_rate_hz

    @property
    def is_downsampling(self) -> bool:
        """Whether this buffer performs downsampling."""
        return self.target_rate_hz < self.source_rate_hz

    @property
    def rate_ratio(self) -> float:
        """Ratio of target to source rate."""
        return self.target_rate_hz / self.source_rate_hz

    @property
    def source_period_ms(self) -> float:
        """Source update period in milliseconds."""
        return 1000.0 / self.source_rate_hz

    @property
    def target_period_ms(self) -> float:
        """Target consumption period in milliseconds."""
        return 1000.0 / self.target_rate_hz

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "source_rate_hz": self.source_rate_hz,
            "target_rate_hz": self.target_rate_hz,
            "buffer_size": self.buffer_size,
            "interpolation": self.interpolation.name,
            "extrapolation": self.extrapolation.name,
            "max_staleness_ms": self.max_staleness_ms,
            "is_upsampling": self.is_upsampling,
            "rate_ratio": self.rate_ratio,
        }


@dataclass
class AsyncSensorConfig:
    """
    Configuration for asynchronous sensor handling.

    Attributes:
        name: Sensor identifier.
        rate_hz: Expected sensor update rate.
        sync_mode: Synchronization mode.
        queue_size: Size of the async queue.
        drop_policy: Policy for queue overflow ("oldest", "newest").
        timeout_ms: Timeout for sync mode.
        required: Whether this sensor is required for policy execution.
    """

    name: str
    rate_hz: float
    sync_mode: SensorSyncMode = SensorSyncMode.LATEST
    queue_size: int = 2
    drop_policy: str = "oldest"
    timeout_ms: float = 50.0
    required: bool = True

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "rate_hz": self.rate_hz,
            "sync_mode": self.sync_mode.name,
            "queue_size": self.queue_size,
            "drop_policy": self.drop_policy,
            "timeout_ms": self.timeout_ms,
            "required": self.required,
        }


@dataclass
class PipelineStage:
    """
    A stage in the multi-rate execution pipeline.

    Attributes:
        name: Stage identifier.
        components: Components executed in this stage.
        rate_hz: Execution frequency.
        priority: Scheduling priority (higher = more important).
        max_latency_ms: Maximum allowed latency for this stage.
        can_skip: Whether this stage can be skipped under load.
        dependencies: Stages this stage depends on.
    """

    name: str
    components: list[str] = field(default_factory=list)
    rate_hz: float = 20.0
    priority: int = 0
    max_latency_ms: float = 50.0
    can_skip: bool = False
    dependencies: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "components": self.components,
            "rate_hz": self.rate_hz,
            "priority": self.priority,
            "max_latency_ms": self.max_latency_ms,
            "can_skip": self.can_skip,
            "dependencies": self.dependencies,
        }


@dataclass
class MultiRateSchedule:
    """
    Complete multi-rate execution schedule.

    Defines how different components execute at different rates,
    including rate conversion and synchronization.

    Attributes:
        stages: Ordered list of pipeline stages.
        rate_buffers: Rate conversion buffers between components.
        sensor_configs: Async sensor configurations.
        base_rate_hz: Base scheduling rate (LCM of all rates).
        deadline_enforcement: Enable strict deadline enforcement.
    """

    stages: list[PipelineStage] = field(default_factory=list)
    rate_buffers: list[RateConversionBuffer] = field(default_factory=list)
    sensor_configs: list[AsyncSensorConfig] = field(default_factory=list)
    base_rate_hz: float = 100.0
    deadline_enforcement: bool = True

    def add_stage(self, stage: PipelineStage) -> None:
        """Add a pipeline stage."""
        self.stages.append(stage)

    def add_rate_buffer(self, buffer: RateConversionBuffer) -> None:
        """Add a rate conversion buffer."""
        self.rate_buffers.append(buffer)

    def add_sensor_config(self, config: AsyncSensorConfig) -> None:
        """Add an async sensor configuration."""
        self.sensor_configs.append(config)

    def get_stage(self, name: str) -> PipelineStage | None:
        """Get stage by name."""
        for stage in self.stages:
            if stage.name == name:
                return stage
        return None

    def get_rate_buffer(self, name: str) -> RateConversionBuffer | None:
        """Get rate buffer by name."""
        for buffer in self.rate_buffers:
            if buffer.name == name:
                return buffer
        return None

    def validate(self) -> list[str]:
        """Validate schedule configuration."""
        errors = []

        # Check for missing dependencies
        stage_names = {s.name for s in self.stages}
        for stage in self.stages:
            for dep in stage.dependencies:
                if dep not in stage_names:
                    errors.append(f"Stage '{stage.name}' depends on unknown stage '{dep}'")

        # Check for circular dependencies
        if self._has_circular_dependency():
            errors.append("Circular dependency detected in pipeline stages")

        # Check rate buffer connectivity
        for buffer in self.rate_buffers:
            if buffer.max_staleness_ms < buffer.source_period_ms:
                errors.append(
                    f"Rate buffer '{buffer.name}' staleness ({buffer.max_staleness_ms}ms) "
                    f"< source period ({buffer.source_period_ms}ms)"
                )

        return errors

    def _has_circular_dependency(self) -> bool:
        """Check for circular dependencies using DFS."""
        visited = set()
        rec_stack = set()

        def dfs(stage_name: str) -> bool:
            visited.add(stage_name)
            rec_stack.add(stage_name)

            stage = self.get_stage(stage_name)
            if stage:
                for dep in stage.dependencies:
                    if dep not in visited:
                        if dfs(dep):
                            return True
                    elif dep in rec_stack:
                        return True

            rec_stack.remove(stage_name)
            return False

        for stage in self.stages:
            if stage.name not in visited:
                if dfs(stage.name):
                    return True
        return False

    def get_execution_order(self) -> list[str]:
        """Get topological order of stages."""
        in_degree = {s.name: 0 for s in self.stages}
        for stage in self.stages:
            for dep in stage.dependencies:
                if dep in in_degree:
                    in_degree[stage.name] += 1

        queue = [name for name, degree in in_degree.items() if degree == 0]
        order = []

        while queue:
            current = queue.pop(0)
            order.append(current)

            for stage in self.stages:
                if current in stage.dependencies:
                    in_degree[stage.name] -= 1
                    if in_degree[stage.name] == 0:
                        queue.append(stage.name)

        return order

    def compute_hyperperiod(self) -> float:
        """Compute hyperperiod (LCM of all periods) in milliseconds."""
        if not self.stages:
            return 1000.0 / self.base_rate_hz

        periods = [1000.0 / s.rate_hz for s in self.stages]

        def lcm(a: float, b: float) -> float:
            return abs(a * b) / math.gcd(int(a * 1000), int(b * 1000)) * 1000

        result = periods[0]
        for period in periods[1:]:
            result = lcm(result, period)

        return result

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "stages": [s.to_dict() for s in self.stages],
            "rate_buffers": [b.to_dict() for b in self.rate_buffers],
            "sensor_configs": [c.to_dict() for c in self.sensor_configs],
            "base_rate_hz": self.base_rate_hz,
            "deadline_enforcement": self.deadline_enforcement,
            "execution_order": self.get_execution_order(),
            "hyperperiod_ms": self.compute_hyperperiod(),
        }


class MultiRateScheduler:
    """
    Scheduler for multi-rate policy execution.

    Coordinates execution of components at different rates,
    manages rate conversion buffers, and handles sensor synchronization.
    """

    def __init__(self, schedule: MultiRateSchedule) -> None:
        """Initialize scheduler."""
        self.schedule = schedule
        self._execution_order = schedule.get_execution_order()
        self._tick_count = 0
        self._hyperperiod = schedule.compute_hyperperiod()

    def get_stages_for_tick(self, tick_ms: float) -> list[str]:
        """
        Get which stages should execute at a given tick.

        Args:
            tick_ms: Current tick time in milliseconds.

        Returns:
            List of stage names that should execute.
        """
        stages_to_run = []

        for stage in self.schedule.stages:
            period_ms = 1000.0 / stage.rate_hz
            # Check if this tick is aligned with stage period
            if tick_ms % period_ms < (1000.0 / self.schedule.base_rate_hz):
                stages_to_run.append(stage.name)

        # Return in execution order
        return [s for s in self._execution_order if s in stages_to_run]

    def create_execution_plan(self, duration_ms: float) -> list[dict]:
        """
        Create an execution plan for a given duration.

        Args:
            duration_ms: Duration to plan in milliseconds.

        Returns:
            List of execution events with timestamps.
        """
        plan = []
        base_period = 1000.0 / self.schedule.base_rate_hz

        tick = 0.0
        while tick < duration_ms:
            stages = self.get_stages_for_tick(tick)
            if stages:
                plan.append({
                    "tick_ms": tick,
                    "stages": stages,
                })
            tick += base_period

        return plan

    def estimate_cpu_utilization(self) -> dict[str, float]:
        """
        Estimate CPU utilization per stage.

        Returns:
            Dictionary of stage name to utilization percentage.
        """
        utilization = {}

        for stage in self.schedule.stages:
            # Utilization = (execution time / period) * 100
            # Assume max latency is worst-case execution time
            period_ms = 1000.0 / stage.rate_hz
            util = (stage.max_latency_ms / period_ms) * 100.0
            utilization[stage.name] = min(util, 100.0)

        return utilization

    def check_schedulability(self) -> tuple[bool, list[str]]:
        """
        Check if schedule is feasible.

        Uses rate-monotonic analysis for single-core.

        Returns:
            Tuple of (is_schedulable, list of warnings).
        """
        warnings = []

        # Calculate total utilization
        total_util = sum(self.estimate_cpu_utilization().values())

        # Liu & Layland bound for n tasks
        n = len(self.schedule.stages)
        if n > 0:
            bound = n * (2 ** (1.0 / n) - 1) * 100.0

            if total_util > bound:
                warnings.append(
                    f"Total utilization ({total_util:.1f}%) exceeds "
                    f"rate-monotonic bound ({bound:.1f}%)"
                )
            elif total_util > 100.0:
                warnings.append(
                    f"Total utilization ({total_util:.1f}%) exceeds 100%"
                )

        # Check individual deadline feasibility
        for stage in self.schedule.stages:
            period_ms = 1000.0 / stage.rate_hz
            if stage.max_latency_ms > period_ms:
                warnings.append(
                    f"Stage '{stage.name}' latency ({stage.max_latency_ms}ms) "
                    f"> period ({period_ms:.1f}ms)"
                )

        is_schedulable = len(warnings) == 0 or total_util <= 100.0
        return is_schedulable, warnings


def create_schedule_from_module(module: RobotModule) -> MultiRateSchedule:
    """
    Create a multi-rate schedule from a Robot IR module.

    Analyzes the module's control loop and sensors to create
    an appropriate multi-rate execution schedule.

    Args:
        module: Robot IR module.

    Returns:
        MultiRateSchedule for the module.
    """
    schedule = MultiRateSchedule()

    if not module.control:
        return schedule

    # Create stages from multi-rate specs
    if module.control.multi_rate:
        for rate_spec in module.control.multi_rate:
            stage = PipelineStage(
                name=rate_spec.name,
                rate_hz=rate_spec.frequency_hz,
                priority=rate_spec.priority,
                max_latency_ms=module.control.latency_budget_ms / 2,
            )
            schedule.add_stage(stage)
    else:
        # Single-rate fallback
        schedule.add_stage(PipelineStage(
            name="main",
            rate_hz=module.control.frequency_hz,
            max_latency_ms=module.control.latency_budget_ms,
        ))

    # Add sensor configs
    for sensor in module.sensors:
        config = AsyncSensorConfig(
            name=sensor.name,
            rate_hz=sensor.rate_hz,
            timeout_ms=sensor.latency_ms + 10.0 if sensor.latency_ms > 0 else 50.0,
        )
        schedule.add_sensor_config(config)

    # Create rate buffers between stages
    stage_rates = {s.name: s.rate_hz for s in schedule.stages}
    for i, stage in enumerate(schedule.stages[:-1]):
        next_stage = schedule.stages[i + 1]
        if stage.rate_hz != next_stage.rate_hz:
            buffer = RateConversionBuffer(
                name=f"{stage.name}_to_{next_stage.name}",
                source_rate_hz=stage.rate_hz,
                target_rate_hz=next_stage.rate_hz,
            )
            schedule.add_rate_buffer(buffer)
            next_stage.dependencies.append(stage.name)

    # Set base rate
    all_rates = [s.rate_hz for s in schedule.stages] + [c.rate_hz for c in schedule.sensor_configs]
    if all_rates:
        schedule.base_rate_hz = max(all_rates)

    return schedule


# Preset schedules for common configurations


def create_vla_schedule(
    vision_hz: float = 10.0,
    policy_hz: float = 20.0,
    control_hz: float = 100.0,
) -> MultiRateSchedule:
    """
    Create a standard VLA (Vision-Language-Action) schedule.

    Typical rates:
    - Vision: 10 Hz (heavy computation)
    - Policy: 20 Hz (action generation)
    - Control: 100 Hz (low-level control)

    Args:
        vision_hz: Vision processing rate.
        policy_hz: Policy execution rate.
        control_hz: Control output rate.

    Returns:
        MultiRateSchedule for VLA execution.
    """
    schedule = MultiRateSchedule(base_rate_hz=control_hz)

    # Vision stage (slowest)
    schedule.add_stage(PipelineStage(
        name="vision",
        components=["vision_encoder", "vision_projector"],
        rate_hz=vision_hz,
        priority=1,
        max_latency_ms=80.0,
        can_skip=True,
    ))

    # Policy stage
    schedule.add_stage(PipelineStage(
        name="policy",
        components=["policy_backbone", "action_head"],
        rate_hz=policy_hz,
        priority=2,
        max_latency_ms=40.0,
        can_skip=False,
        dependencies=["vision"],
    ))

    # Control stage (fastest)
    schedule.add_stage(PipelineStage(
        name="control",
        components=["action_interpolator", "safety_filter"],
        rate_hz=control_hz,
        priority=3,
        max_latency_ms=5.0,
        can_skip=False,
        dependencies=["policy"],
    ))

    # Rate conversion buffers
    schedule.add_rate_buffer(RateConversionBuffer(
        name="vision_to_policy",
        source_rate_hz=vision_hz,
        target_rate_hz=policy_hz,
        interpolation=InterpolationMode.ZERO_ORDER_HOLD,
    ))

    schedule.add_rate_buffer(RateConversionBuffer(
        name="policy_to_control",
        source_rate_hz=policy_hz,
        target_rate_hz=control_hz,
        interpolation=InterpolationMode.LINEAR,
    ))

    return schedule


def create_diffusion_schedule(
    vision_hz: float = 30.0,
    diffusion_hz: float = 10.0,
    action_hz: float = 10.0,
) -> MultiRateSchedule:
    """
    Create a schedule for diffusion policy execution.

    Diffusion policies have unique characteristics:
    - Multi-step inference (DDIM)
    - Action trajectory output
    - Lower frequency due to iterative denoising

    Args:
        vision_hz: Vision processing rate.
        diffusion_hz: Diffusion inference rate.
        action_hz: Action output rate.

    Returns:
        MultiRateSchedule for diffusion policy.
    """
    schedule = MultiRateSchedule(base_rate_hz=vision_hz)

    schedule.add_stage(PipelineStage(
        name="vision",
        components=["vision_encoder", "obs_encoder"],
        rate_hz=vision_hz,
        priority=1,
        max_latency_ms=20.0,
    ))

    schedule.add_stage(PipelineStage(
        name="diffusion",
        components=["denoiser", "diffusion_loop"],
        rate_hz=diffusion_hz,
        priority=2,
        max_latency_ms=80.0,  # Iterative denoising takes time
        can_skip=False,
        dependencies=["vision"],
    ))

    schedule.add_stage(PipelineStage(
        name="action",
        components=["action_selector"],
        rate_hz=action_hz,
        priority=3,
        max_latency_ms=5.0,
        dependencies=["diffusion"],
    ))

    schedule.add_rate_buffer(RateConversionBuffer(
        name="vision_to_diffusion",
        source_rate_hz=vision_hz,
        target_rate_hz=diffusion_hz,
        interpolation=InterpolationMode.ZERO_ORDER_HOLD,
    ))

    return schedule


def create_act_schedule(
    vision_hz: float = 30.0,
    policy_hz: float = 50.0,
) -> MultiRateSchedule:
    """
    Create a schedule for ACT policy execution.

    ACT uses temporal ensemble which requires careful scheduling.

    Args:
        vision_hz: Vision processing rate.
        policy_hz: Policy (and action output) rate.

    Returns:
        MultiRateSchedule for ACT policy.
    """
    schedule = MultiRateSchedule(base_rate_hz=policy_hz)

    schedule.add_stage(PipelineStage(
        name="vision",
        components=["vision_backbone"],
        rate_hz=vision_hz,
        priority=1,
        max_latency_ms=15.0,
    ))

    schedule.add_stage(PipelineStage(
        name="encoder",
        components=["proprio_encoder", "transformer_encoder"],
        rate_hz=policy_hz,
        priority=2,
        max_latency_ms=10.0,
        dependencies=["vision"],
    ))

    schedule.add_stage(PipelineStage(
        name="decoder",
        components=["cvae_sampler", "transformer_decoder", "action_head"],
        rate_hz=policy_hz,
        priority=3,
        max_latency_ms=10.0,
        dependencies=["encoder"],
    ))

    schedule.add_stage(PipelineStage(
        name="ensemble",
        components=["temporal_ensemble"],
        rate_hz=policy_hz,
        priority=4,
        max_latency_ms=2.0,
        dependencies=["decoder"],
    ))

    schedule.add_rate_buffer(RateConversionBuffer(
        name="vision_to_encoder",
        source_rate_hz=vision_hz,
        target_rate_hz=policy_hz,
        interpolation=InterpolationMode.ZERO_ORDER_HOLD,
    ))

    return schedule
