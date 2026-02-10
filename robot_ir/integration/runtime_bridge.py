"""Bridge between Robot IR and robot_runtime.

This module provides the interface for converting Robot IR modules
to robot_runtime adapters and vice versa.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from robot_ir.core.module import RobotModule
    from robot_ir.core.config import DeploymentConfig


@dataclass
class RuntimeBridge:
    """
    Bridge between Robot IR and robot_runtime.

    Converts Robot IR modules to robot_runtime-compatible adapters
    and manages the execution lifecycle.

    Example:
        >>> bridge = RuntimeBridge()
        >>> adapter = bridge.to_runtime_adapter(ir_module)
        >>> adapter.warmup(sample_obs)
        >>> action, state, stats = adapter.step(obs, state)
    """

    device: str = "cuda:0"
    enable_profiling: bool = False

    def to_runtime_adapter(self, module: RobotModule) -> Any:
        """
        Convert Robot IR module to robot_runtime adapter.

        Args:
            module: Robot IR module.

        Returns:
            robot_runtime adapter instance.
        """
        # TODO: Implement conversion
        # 1. Map sensors to observation spec
        # 2. Map compute graph to model
        # 3. Map control loop to runtime config
        # 4. Create appropriate adapter type
        raise NotImplementedError("Runtime adapter conversion not yet implemented")

    def from_runtime_adapter(self, adapter: Any) -> RobotModule:
        """
        Create Robot IR module from robot_runtime adapter.

        Args:
            adapter: robot_runtime adapter instance.

        Returns:
            Robot IR module.
        """
        # TODO: Implement reverse conversion
        # 1. Extract model structure
        # 2. Infer sensor configuration
        # 3. Extract runtime parameters
        raise NotImplementedError("Reverse conversion not yet implemented")

    def create_runtime_config(self, module: RobotModule) -> dict:
        """
        Create robot_runtime configuration from IR module.

        Args:
            module: Robot IR module.

        Returns:
            Configuration dictionary for RuntimeConfig.
        """
        config = {}

        if module.control:
            config["deadline_ms"] = module.control.latency_budget_ms

        if module.schedule:
            config["enable_cuda_graph"] = module.schedule.enable_cuda_graphs

        if module.deploy:
            config["device"] = self.device

        return config


class IRCompiler:
    """
    Compile Robot IR to optimized execution form.

    Takes a Robot IR module through lowering passes and
    produces an optimized executable.
    """

    def __init__(self, target: DeploymentConfig | None = None) -> None:
        """Initialize compiler."""
        self.target = target
        self._passes: list = []

    def add_default_passes(self) -> None:
        """Add default optimization passes."""
        from robot_ir.lowering.passes import (
            KVReusePlanningPass,
            LatencySchedulerPass,
            PrecisionPlannerPass,
            MemoryLayoutPass,
        )

        self._passes = [
            KVReusePlanningPass(),
            PrecisionPlannerPass(),
            MemoryLayoutPass(),
            LatencySchedulerPass(),
        ]

    def compile(self, module: RobotModule) -> Any:
        """
        Compile IR module to executable form.

        Args:
            module: Robot IR module.

        Returns:
            Compiled executable.
        """
        from robot_ir.lowering.base import LoweringContext, PassPipeline

        ctx = LoweringContext(target=self.target)
        pipeline = PassPipeline(self._passes)

        optimized = pipeline.run(module, ctx)

        if ctx.has_errors():
            raise RuntimeError(f"Compilation failed: {ctx.errors}")

        # TODO: Generate executable from optimized IR
        return optimized
