"""Bridge between Robot IR and robot_runtime.

This module provides the interface for converting Robot IR modules
to robot_runtime adapters and vice versa.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
        >>> from robot_ir.integration import Pi0Mapping, RuntimeBridge
        >>>
        >>> # Create IR module
        >>> mapping = Pi0Mapping()
        >>> ir_module = mapping.create_module("my_pi0")
        >>>
        >>> # Create runtime adapter
        >>> bridge = RuntimeBridge()
        >>> runtime_config, adapter_config = bridge.create_runtime_configs(ir_module)
        >>>
        >>> # Use with robot_runtime
        >>> from robot_runtime import RuntimeConfig, PI0Adapter, PI0Config
        >>> config = RuntimeConfig(**runtime_config)
        >>> pi0_config = PI0Config(**adapter_config)
        >>> adapter = PI0Adapter(policy, config, pi0_config)
    """

    device: str = "cuda:0"
    enable_profiling: bool = False

    def create_runtime_configs(self, module: RobotModule) -> tuple[dict, dict]:
        """
        Create robot_runtime configuration dicts from IR module.

        Args:
            module: Robot IR module.

        Returns:
            Tuple of (RuntimeConfig dict, adapter-specific config dict).
        """
        runtime_config = self._create_runtime_config(module)
        adapter_config = self._create_adapter_config(module)
        return runtime_config, adapter_config

    def _create_runtime_config(self, module: RobotModule) -> dict:
        """
        Create RuntimeConfig dictionary from IR module.

        Args:
            module: Robot IR module.

        Returns:
            Dictionary compatible with robot_runtime.RuntimeConfig.
        """
        config = {
            "device": self.device,
            "enable_profiling": self.enable_profiling,
        }

        # Map control loop to deadline
        if module.control:
            config["deadline_ms"] = module.control.latency_budget_ms
            # Map deadline policy
            if module.control.deadline_policy == "drop":
                config["timeout_policy"] = "RETURN_LAST"
            elif module.control.deadline_policy == "block":
                config["timeout_policy"] = "RAISE"

        # Map scheduling hints to backend selection
        if module.schedule:
            if module.schedule.enable_cuda_graphs:
                config["backend"] = "cuda_graph"
            elif module.schedule.tensorrt_enabled:
                config["backend"] = "tensorrt"
            else:
                config["backend"] = "eager"

            config["preallocate"] = True
            config["warmup_iterations"] = 3

        return config

    def _create_adapter_config(self, module: RobotModule) -> dict:
        """
        Create adapter-specific configuration from IR module.

        Args:
            module: Robot IR module.

        Returns:
            Dictionary for adapter configuration (PI0Config, etc.).
        """
        config = {}

        # Extract action chunk configuration from state buffers
        for state in module.states:
            if state.name == "action_chunk":
                chunk_shape = state.shape.dims
                if len(chunk_shape) >= 2:
                    config["chunk_size"] = chunk_shape[0]

        # Extract temporal settings from compute graph
        if module.graph:
            chunk_selector = module.graph.get_node("action_chunk_selector")
            if chunk_selector:
                node_config = getattr(chunk_selector, "config", {})
                if "temporal_steps" in node_config:
                    config["temporal_action_steps"] = node_config["temporal_steps"]
                if "chunk_size" in node_config:
                    config["chunk_size"] = node_config["chunk_size"]

        # Default values
        config.setdefault("chunk_size", 50)
        config.setdefault("temporal_action_steps", 1)
        config.setdefault("use_action_cache", True)
        config.setdefault("normalize_observations", True)
        config.setdefault("denormalize_actions", True)

        return config

    def extract_observation_spec(self, module: RobotModule) -> dict:
        """
        Extract observation specification from IR module.

        Returns dict mapping sensor names to their specifications,
        compatible with robot_runtime's Observation class.

        Args:
            module: Robot IR module.

        Returns:
            Dictionary with observation specifications.
        """
        spec = {
            "images": {},
            "state_dim": None,
            "n_obs_steps": 1,
        }

        for sensor in module.sensors:
            if sensor.modality.name == "VISION":
                spec["images"][sensor.name] = {
                    "shape": sensor.shape.dims,
                    "dtype": sensor.shape.dtype.name,
                }
            elif sensor.modality.name == "PROPRIO":
                spec["state_dim"] = sensor.shape.dims[-1]
                spec["n_obs_steps"] = sensor.buffering.window_size

        return spec

    def extract_action_spec(self, module: RobotModule) -> dict:
        """
        Extract action specification from IR module.

        Args:
            module: Robot IR module.

        Returns:
            Dictionary with action specifications.
        """
        spec = {
            "action_dim": None,
            "chunk_size": None,
        }

        for state in module.states:
            if state.name == "action_chunk":
                dims = state.shape.dims
                if len(dims) >= 2:
                    spec["chunk_size"] = dims[0]
                    spec["action_dim"] = dims[1]

        return spec

    def get_adapter_type(self, module: RobotModule) -> str:
        """
        Determine the appropriate adapter type for the IR module.

        Args:
            module: Robot IR module.

        Returns:
            Adapter type string ("pi0", "pi0.5", "act", etc.).
        """
        name = module.name.lower()

        # Check module name for hints
        if "pi0" in name and ("5" in name or "05" in name):
            return "pi0.5"
        elif "pi0" in name or "pi_0" in name:
            return "pi0"
        elif "act" in name:
            return "act"
        elif "diffusion" in name:
            return "diffusion"

        # Check compute graph for model references
        if module.graph:
            for node in module.graph.nodes:
                if hasattr(node, "model_ref"):
                    ref = node.model_ref.lower()
                    if "siglip" in ref:
                        return "pi0"
                    elif "paligemma" in ref:
                        return "pi0.5"

        return "unknown"

    def validate_for_runtime(self, module: RobotModule) -> list[str]:
        """
        Validate IR module for robot_runtime compatibility.

        Args:
            module: Robot IR module.

        Returns:
            List of validation error messages. Empty if valid.
        """
        errors = []

        # Check basic module validity
        errors.extend(module.validate())

        # Check for required sensors
        has_vision = any(s.modality.name == "VISION" for s in module.sensors)
        has_proprio = any(s.modality.name == "PROPRIO" for s in module.sensors)

        if not has_vision:
            errors.append("Module must have at least one VISION sensor")
        if not has_proprio:
            errors.append("Module must have a PROPRIO sensor")

        # Check for action output
        has_action_buffer = any(s.name == "action_chunk" for s in module.states)
        if not has_action_buffer:
            errors.append("Module must have an action_chunk state buffer")

        # Check control loop
        if module.control:
            if module.control.latency_budget_ms <= 0:
                errors.append("Latency budget must be positive")
            if module.control.frequency_hz <= 0:
                errors.append("Control frequency must be positive")

        return errors


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
            MultiRateLoopPass,
            KernelFusionPass,
        )

        self._passes = [
            KVReusePlanningPass(),
            PrecisionPlannerPass(),
            MemoryLayoutPass(),
            LatencySchedulerPass(),
            MultiRateLoopPass(),
            KernelFusionPass(),
        ]

    def compile(self, module: RobotModule) -> RobotModule:
        """
        Compile IR module through optimization passes.

        Args:
            module: Robot IR module.

        Returns:
            Optimized module.
        """
        from robot_ir.lowering.base import LoweringContext, PassPipeline

        ctx = LoweringContext(target=self.target)
        pipeline = PassPipeline(self._passes)

        optimized = pipeline.run(module, ctx)

        if ctx.has_errors():
            raise RuntimeError(f"Compilation failed: {ctx.errors}")

        return optimized

    def get_optimization_report(self, module: RobotModule) -> dict:
        """
        Get a report of optimizations that would be applied.

        Args:
            module: Robot IR module.

        Returns:
            Dictionary with optimization analysis.
        """
        report = {
            "module_name": module.name,
            "num_sensors": len(module.sensors),
            "num_states": len(module.states),
            "num_compute_nodes": len(module.graph.nodes) if module.graph else 0,
            "optimizations": [],
        }

        # Analyze potential optimizations
        if module.schedule:
            if module.schedule.enable_cuda_graphs:
                report["optimizations"].append("CUDA graph capture")
            if module.schedule.tensorrt_enabled:
                report["optimizations"].append("TensorRT compilation")
            if module.schedule.kernel_fusion_allowed:
                report["optimizations"].append("Kernel fusion")
            if module.schedule.kv_reuse_threshold > 0:
                report["optimizations"].append(
                    f"KV cache reuse (threshold={module.schedule.kv_reuse_threshold})"
                )

        # Memory analysis
        for state in module.states:
            if state.reuse_strategy.name == "KV_CACHE":
                report["optimizations"].append(f"KV cache: {state.name}")

        return report
