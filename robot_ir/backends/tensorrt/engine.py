"""TensorRT engine builder and runner.

Provides compilation of Robot IR neural blocks to TensorRT engines
with support for various precision modes and optimization strategies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from robot_ir.compute.nodes import NeuralBlock
    from robot_ir.compute.precision import PrecisionPolicy


class TRTPrecision(Enum):
    """TensorRT precision modes."""

    FP32 = auto()
    FP16 = auto()
    BF16 = auto()
    FP8 = auto()
    INT8 = auto()
    INT4 = auto()  # Weight-only quantization


class TRTOptLevel(Enum):
    """TensorRT optimization level."""

    FAST_BUILD = 0  # Fast build, less optimization
    BALANCED = 3  # Balanced build time and performance
    MAX_PERF = 5  # Maximum optimization (slow build)


@dataclass
class TRTConfig:
    """
    TensorRT engine build configuration.

    Attributes:
        precision: Target precision mode.
        opt_level: Optimization level.
        workspace_size_mb: Maximum workspace size in MB.
        max_batch_size: Maximum batch size.
        dynamic_shapes: Enable dynamic shape support.
        enable_cuda_graph: Enable CUDA graph capture.
        cache_dir: Directory for engine caching.
        calibration_data: Calibration data for INT8/FP8.
        layer_precision_overrides: Per-layer precision overrides.
    """

    precision: TRTPrecision = TRTPrecision.FP16
    opt_level: TRTOptLevel = TRTOptLevel.BALANCED
    workspace_size_mb: int = 2048
    max_batch_size: int = 1
    dynamic_shapes: bool = False
    enable_cuda_graph: bool = True
    cache_dir: Path | None = None
    calibration_data: Any | None = None
    layer_precision_overrides: dict[str, TRTPrecision] = field(default_factory=dict)

    def to_builder_flags(self) -> dict:
        """Convert to TensorRT builder configuration."""
        flags = {
            "workspace_size": self.workspace_size_mb * 1024 * 1024,
            "max_batch_size": self.max_batch_size,
            "builder_optimization_level": self.opt_level.value,
        }

        # Precision flags
        if self.precision == TRTPrecision.FP16:
            flags["fp16_mode"] = True
        elif self.precision == TRTPrecision.BF16:
            flags["bf16_mode"] = True
        elif self.precision == TRTPrecision.FP8:
            flags["fp8_mode"] = True
        elif self.precision == TRTPrecision.INT8:
            flags["int8_mode"] = True

        return flags


@dataclass
class TRTEngineSpec:
    """
    Specification for a TensorRT engine.

    Attributes:
        name: Engine identifier.
        input_specs: Input tensor specifications.
        output_specs: Output tensor specifications.
        config: Build configuration.
        source_block: Source neural block from IR.
    """

    name: str
    input_specs: list[dict] = field(default_factory=list)
    output_specs: list[dict] = field(default_factory=list)
    config: TRTConfig = field(default_factory=TRTConfig)
    source_block: NeuralBlock | None = None

    def get_cache_key(self) -> str:
        """Generate cache key for engine lookup."""
        import hashlib
        key_data = f"{self.name}_{self.config.precision.name}_{self.config.max_batch_size}"
        for spec in self.input_specs:
            key_data += f"_{spec.get('shape', '')}_{spec.get('dtype', '')}"
        return hashlib.md5(key_data.encode()).hexdigest()[:16]


class TRTEngineBuilder:
    """
    Build TensorRT engines from Robot IR neural blocks.

    Example:
        >>> builder = TRTEngineBuilder(config)
        >>> spec = builder.create_spec(neural_block)
        >>> engine_path = builder.build(spec)
    """

    def __init__(self, config: TRTConfig | None = None) -> None:
        """Initialize builder."""
        self.config = config or TRTConfig()
        self._trt_available = self._check_tensorrt()

    def _check_tensorrt(self) -> bool:
        """Check if TensorRT is available."""
        try:
            import tensorrt
            return True
        except ImportError:
            return False

    def create_spec_from_block(
        self,
        block: NeuralBlock,
        input_shapes: dict[str, tuple],
    ) -> TRTEngineSpec:
        """
        Create engine spec from a neural block.

        Args:
            block: Neural block from IR.
            input_shapes: Input tensor shapes.

        Returns:
            TRTEngineSpec for building.
        """
        # Map IR precision to TRT precision
        trt_precision = self._map_precision(block.precision_policy)

        config = TRTConfig(
            precision=trt_precision,
            opt_level=self.config.opt_level,
            workspace_size_mb=self.config.workspace_size_mb,
            max_batch_size=self.config.max_batch_size,
            enable_cuda_graph=self.config.enable_cuda_graph,
            cache_dir=self.config.cache_dir,
        )

        # Apply layer precision overrides for critical layers
        if block.precision_policy and block.precision_policy.critical_layers:
            for layer in block.precision_policy.critical_layers:
                config.layer_precision_overrides[layer] = TRTPrecision.FP16

        input_specs = []
        for name, shape in input_shapes.items():
            input_specs.append({
                "name": name,
                "shape": shape,
                "dtype": "float16" if trt_precision == TRTPrecision.FP16 else "float32",
            })

        return TRTEngineSpec(
            name=block.name,
            input_specs=input_specs,
            config=config,
            source_block=block,
        )

    def _map_precision(self, policy: PrecisionPolicy | None) -> TRTPrecision:
        """Map IR precision policy to TRT precision."""
        if policy is None:
            return TRTPrecision.FP16

        from robot_ir.compute.precision import Precision

        mapping = {
            Precision.FP32: TRTPrecision.FP32,
            Precision.FP16: TRTPrecision.FP16,
            Precision.BF16: TRTPrecision.BF16,
            Precision.FP8_E4M3: TRTPrecision.FP8,
            Precision.FP8_E5M2: TRTPrecision.FP8,
            Precision.INT8: TRTPrecision.INT8,
            Precision.INT4: TRTPrecision.INT4,
        }

        return mapping.get(policy.preferred, TRTPrecision.FP16)

    def build(self, spec: TRTEngineSpec) -> Path | None:
        """
        Build TensorRT engine from spec.

        Args:
            spec: Engine specification.

        Returns:
            Path to built engine file, or None if TRT unavailable.
        """
        if not self._trt_available:
            return None

        # Check cache first
        if spec.config.cache_dir:
            cache_path = spec.config.cache_dir / f"{spec.get_cache_key()}.engine"
            if cache_path.exists():
                return cache_path

        # Build engine (placeholder - actual TRT build logic)
        engine_path = self._do_build(spec)

        return engine_path

    def _do_build(self, spec: TRTEngineSpec) -> Path | None:
        """
        Execute TensorRT engine build.

        This is a placeholder for actual TensorRT build logic.
        In production, this would:
        1. Load the model from checkpoint
        2. Create TensorRT network
        3. Configure builder with precision settings
        4. Build and serialize engine
        """
        # Placeholder implementation
        # Actual implementation requires tensorrt package
        return None


class TRTEngineRunner:
    """
    Run inference with TensorRT engines.

    Supports CUDA graph capture for reduced kernel launch overhead.
    """

    def __init__(self, engine_path: Path, enable_cuda_graph: bool = True) -> None:
        """Initialize runner."""
        self.engine_path = engine_path
        self.enable_cuda_graph = enable_cuda_graph
        self._engine = None
        self._context = None
        self._cuda_graph = None
        self._is_warmed_up = False

    def load(self) -> bool:
        """
        Load TensorRT engine.

        Returns:
            True if loaded successfully.
        """
        if not self.engine_path.exists():
            return False

        try:
            import tensorrt as trt
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            with open(self.engine_path, "rb") as f:
                self._engine = runtime.deserialize_cuda_engine(f.read())
            self._context = self._engine.create_execution_context()
            return True
        except Exception:
            return False

    def warmup(self, sample_inputs: dict[str, Any], iterations: int = 3) -> None:
        """
        Warmup engine and optionally capture CUDA graph.

        Args:
            sample_inputs: Sample input tensors.
            iterations: Number of warmup iterations.
        """
        if self._context is None:
            raise RuntimeError("Engine not loaded")

        # Run warmup iterations
        for _ in range(iterations):
            self._run_inference(sample_inputs)

        # Capture CUDA graph if enabled
        if self.enable_cuda_graph:
            self._capture_cuda_graph(sample_inputs)

        self._is_warmed_up = True

    def _capture_cuda_graph(self, sample_inputs: dict[str, Any]) -> None:
        """Capture CUDA graph for inference."""
        try:
            import torch
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                # Warmup in capturing mode
                self._run_inference(sample_inputs)
                stream.synchronize()

                # Capture graph
                self._cuda_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self._cuda_graph, stream=stream):
                    self._run_inference(sample_inputs)
        except Exception:
            self._cuda_graph = None

    def _run_inference(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Run single inference pass."""
        # Placeholder - actual inference logic
        return {}

    def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Run inference.

        Args:
            inputs: Input tensors.

        Returns:
            Output tensors.
        """
        if not self._is_warmed_up:
            raise RuntimeError("Engine not warmed up")

        if self._cuda_graph is not None:
            # Replay CUDA graph
            self._cuda_graph.replay()
            return self._get_outputs()
        else:
            return self._run_inference(inputs)

    def _get_outputs(self) -> dict[str, Any]:
        """Get outputs after CUDA graph replay."""
        return {}
