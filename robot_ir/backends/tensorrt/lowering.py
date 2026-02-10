"""TensorRT lowering pass for Robot IR.

Converts neural blocks to TensorRT engines and integrates
them into the execution graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from robot_ir.lowering.base import LoweringPass, LoweringContext
from robot_ir.backends.tensorrt.engine import TRTConfig, TRTEngineBuilder, TRTEngineSpec

if TYPE_CHECKING:
    from robot_ir.core.module import RobotModule
    from robot_ir.compute.nodes import NeuralBlock


@dataclass
class TRTLoweringConfig:
    """Configuration for TensorRT lowering."""

    cache_dir: Path = field(default_factory=lambda: Path(".trt_cache"))
    workspace_size_mb: int = 2048
    max_batch_size: int = 1
    enable_cuda_graph: bool = True
    prefer_fp8: bool = True
    fallback_to_eager: bool = True


class TensorRTLoweringPass(LoweringPass):
    """
    Lower neural blocks to TensorRT engines.

    This pass:
    1. Identifies neural blocks that can be compiled to TRT
    2. Creates engine specifications with precision settings
    3. Builds or loads cached TRT engines
    4. Updates the IR with engine references
    """

    name = "tensorrt_lowering"

    def __init__(self, config: TRTLoweringConfig | None = None) -> None:
        """Initialize pass."""
        self.config = config or TRTLoweringConfig()
        self._builder: TRTEngineBuilder | None = None
        self._engine_specs: dict[str, TRTEngineSpec] = {}

    def run(self, module: RobotModule, ctx: LoweringContext) -> RobotModule:
        """
        Execute TensorRT lowering.

        Args:
            module: Robot IR module.
            ctx: Lowering context.

        Returns:
            Module with TRT engine references.
        """
        if not self._should_run(module, ctx):
            return module

        # Initialize builder
        self._builder = TRTEngineBuilder(TRTConfig(
            workspace_size_mb=self.config.workspace_size_mb,
            max_batch_size=self.config.max_batch_size,
            enable_cuda_graph=self.config.enable_cuda_graph,
            cache_dir=self.config.cache_dir,
        ))

        # Process each neural block
        if module.graph:
            for node in module.graph.nodes:
                if self._is_neural_block(node):
                    self._process_block(node, module, ctx)

        # Store engine specs in context for later use
        ctx.set_metadata("trt_engine_specs", self._engine_specs)

        return module

    def _should_run(self, module: RobotModule, ctx: LoweringContext) -> bool:
        """Check if TRT lowering should run."""
        if module.schedule is None:
            return False
        return module.schedule.tensorrt_enabled

    def _is_neural_block(self, node) -> bool:
        """Check if node is a neural block."""
        from robot_ir.compute.nodes import NeuralBlock
        return isinstance(node, NeuralBlock)

    def _process_block(
        self,
        block: NeuralBlock,
        module: RobotModule,
        ctx: LoweringContext,
    ) -> None:
        """
        Process a single neural block.

        Args:
            block: Neural block to process.
            module: Parent module.
            ctx: Lowering context.
        """
        # Infer input shapes from sensors and upstream nodes
        input_shapes = self._infer_input_shapes(block, module)

        # Create engine spec
        spec = self._builder.create_spec_from_block(block, input_shapes)
        self._engine_specs[block.name] = spec

        # Build engine (or load from cache)
        engine_path = self._builder.build(spec)

        if engine_path:
            # Store engine path in block metadata
            block.metadata["trt_engine_path"] = str(engine_path)
            block.metadata["trt_precision"] = spec.config.precision.name
            ctx.log(f"Built TRT engine for {block.name}: {engine_path}")
        elif self.config.fallback_to_eager:
            block.metadata["trt_fallback"] = True
            ctx.log(f"TRT build failed for {block.name}, using eager fallback")
        else:
            ctx.add_error(f"TRT build failed for {block.name}")

    def _infer_input_shapes(
        self,
        block: NeuralBlock,
        module: RobotModule,
    ) -> dict[str, tuple]:
        """
        Infer input shapes for a neural block.

        Args:
            block: Neural block.
            module: Parent module.

        Returns:
            Dictionary of input name to shape.
        """
        input_shapes = {}

        # Check if this is an entry point (connected to sensor)
        if module.graph:
            predecessors = module.graph.get_predecessors(block.name)

            if not predecessors:
                # Entry point - get shapes from sensors
                if "vision" in block.name.lower():
                    for sensor in module.sensors:
                        if sensor.modality.name == "VISION":
                            input_shapes[sensor.name] = (1,) + sensor.shape.dims
                            break
                elif "proprio" in block.name.lower():
                    for sensor in module.sensors:
                        if sensor.modality.name == "PROPRIO":
                            input_shapes[sensor.name] = (1,) + sensor.shape.dims
                            break
            else:
                # Get shapes from predecessor outputs
                for pred_name in predecessors:
                    pred_node = module.graph.get_node(pred_name)
                    if pred_node and hasattr(pred_node, "output_shape"):
                        input_shapes[pred_name] = pred_node.output_shape

        # Default shape if not inferred
        if not input_shapes:
            input_shapes["input"] = (1, 3, 224, 224)  # Default vision shape

        return input_shapes

    def get_engine_specs(self) -> dict[str, TRTEngineSpec]:
        """Get all created engine specs."""
        return self._engine_specs


class TRTExecutionPlanner:
    """
    Plan execution order for TRT engines.

    Handles:
    - Engine loading order
    - Memory allocation
    - Stream assignment
    - CUDA graph capture
    """

    def __init__(self) -> None:
        """Initialize planner."""
        self._execution_order: list[str] = []
        self._stream_assignment: dict[str, int] = {}
        self._memory_plan: dict[str, dict] = {}

    def plan(self, module: RobotModule, engine_specs: dict[str, TRTEngineSpec]) -> dict:
        """
        Create execution plan for TRT engines.

        Args:
            module: Robot IR module.
            engine_specs: Engine specifications.

        Returns:
            Execution plan dictionary.
        """
        # Determine execution order from graph
        if module.graph:
            self._execution_order = module.graph.topological_sort()

        # Assign streams based on dependencies
        self._assign_streams(module, engine_specs)

        # Plan memory allocation
        self._plan_memory(engine_specs)

        return {
            "execution_order": self._execution_order,
            "stream_assignment": self._stream_assignment,
            "memory_plan": self._memory_plan,
            "enable_cuda_graph": self._can_use_cuda_graph(engine_specs),
        }

    def _assign_streams(
        self,
        module: RobotModule,
        engine_specs: dict[str, TRTEngineSpec],
    ) -> None:
        """Assign CUDA streams to engines."""
        max_streams = 4
        if module.schedule:
            max_streams = module.schedule.max_concurrent_streams

        # Simple round-robin assignment for now
        stream_id = 0
        for name in self._execution_order:
            if name in engine_specs:
                self._stream_assignment[name] = stream_id
                stream_id = (stream_id + 1) % max_streams

    def _plan_memory(self, engine_specs: dict[str, TRTEngineSpec]) -> None:
        """Plan memory allocation for engines."""
        for name, spec in engine_specs.items():
            # Estimate memory requirements
            input_size = sum(
                self._tensor_size(s.get("shape", ()), s.get("dtype", "float16"))
                for s in spec.input_specs
            )
            output_size = sum(
                self._tensor_size(s.get("shape", ()), s.get("dtype", "float16"))
                for s in spec.output_specs
            )

            self._memory_plan[name] = {
                "input_size_bytes": input_size,
                "output_size_bytes": output_size,
                "workspace_size_bytes": spec.config.workspace_size_mb * 1024 * 1024,
            }

    def _tensor_size(self, shape: tuple, dtype: str) -> int:
        """Calculate tensor size in bytes."""
        if not shape:
            return 0

        num_elements = 1
        for dim in shape:
            num_elements *= dim

        dtype_sizes = {
            "float32": 4,
            "float16": 2,
            "bfloat16": 2,
            "int8": 1,
            "int4": 0.5,
        }

        return int(num_elements * dtype_sizes.get(dtype, 2))

    def _can_use_cuda_graph(self, engine_specs: dict[str, TRTEngineSpec]) -> bool:
        """Check if CUDA graph can be used."""
        # CUDA graph requires static shapes
        for spec in engine_specs.values():
            if spec.config.dynamic_shapes:
                return False
        return True
