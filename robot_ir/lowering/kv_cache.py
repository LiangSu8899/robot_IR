"""KV Cache optimization pass.

Analyzes and optimizes KV cache usage in transformer-based policies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from robot_ir.lowering.base import LoweringPass, LoweringContext
from robot_ir.memory.state import ReuseStrategy, EvictionPolicy

if TYPE_CHECKING:
    from robot_ir.core.module import RobotModule
    from robot_ir.memory.state import StateBuffer


@dataclass
class KVCacheAnalysis:
    """Analysis results for a KV cache buffer."""

    buffer_name: str
    num_layers: int = 0
    max_seq_len: int = 0
    head_dim: int = 0
    num_heads: int = 0
    estimated_size_mb: float = 0.0
    reuse_ratio: float = 0.0  # Fraction of cache reused across steps
    optimal_window: int | None = None  # Optimal sliding window size
    placement: str = "gpu"


@dataclass
class KVCacheOptPlan:
    """Optimization plan for KV cache."""

    enable_sliding_window: bool = False
    window_size: int = 512
    enable_paged_attention: bool = False
    page_size: int = 16
    enable_prefill_chunking: bool = False
    chunk_size: int = 256
    memory_layout: str = "BNSH"  # Batch, Num_heads, Seq, Head_dim
    enable_flash_attention: bool = True
    recompute_threshold: float = 0.3  # Recompute if reuse ratio below this


class KVCacheAnalyzer:
    """
    Analyze KV cache usage patterns.

    Determines optimal cache configuration based on:
    - Model architecture
    - Memory constraints
    - Access patterns
    """

    def __init__(self) -> None:
        """Initialize analyzer."""
        self._analyses: dict[str, KVCacheAnalysis] = {}

    def analyze(self, module: RobotModule) -> dict[str, KVCacheAnalysis]:
        """
        Analyze KV cache buffers in module.

        Args:
            module: Robot IR module.

        Returns:
            Dictionary of buffer name to analysis.
        """
        for state in module.states:
            if state.reuse_strategy == ReuseStrategy.KV_CACHE:
                analysis = self._analyze_buffer(state, module)
                self._analyses[state.name] = analysis

        return self._analyses

    def _analyze_buffer(
        self,
        buffer: StateBuffer,
        module: RobotModule,
    ) -> KVCacheAnalysis:
        """Analyze a single KV cache buffer."""
        shape = buffer.shape.dims

        # Parse shape based on common layouts
        # Typical: (num_layers, 2, seq_len, num_heads, head_dim)
        # or: (num_layers, 2, num_heads, seq_len, head_dim)
        analysis = KVCacheAnalysis(buffer_name=buffer.name)

        if len(shape) >= 4:
            analysis.num_layers = shape[0]
            # shape[1] is typically 2 for K and V
            analysis.max_seq_len = shape[2] if len(shape) == 4 else shape[3]
            analysis.head_dim = shape[-1]
            if len(shape) >= 5:
                analysis.num_heads = shape[3] if shape[2] == 2 else shape[2]

        # Calculate size in MB
        dtype_size = 2  # FP16
        if buffer.shape.dtype.name in ("FLOAT32",):
            dtype_size = 4
        elif buffer.shape.dtype.name in ("INT8", "FLOAT8_E4M3", "FLOAT8_E5M2"):
            dtype_size = 1

        num_elements = 1
        for dim in shape:
            num_elements *= dim

        analysis.estimated_size_mb = (num_elements * dtype_size) / (1024 * 1024)
        analysis.placement = buffer.placement

        # Estimate reuse ratio based on control loop
        if module.control:
            # Higher frequency = more reuse opportunity
            freq = module.control.frequency_hz
            if freq >= 20:
                analysis.reuse_ratio = 0.9
            elif freq >= 10:
                analysis.reuse_ratio = 0.7
            else:
                analysis.reuse_ratio = 0.5

        # Calculate optimal sliding window
        if analysis.max_seq_len > 512 and analysis.reuse_ratio > 0.7:
            analysis.optimal_window = min(512, analysis.max_seq_len // 2)

        return analysis


class KVCacheOptimizer:
    """
    Optimize KV cache configuration.

    Generates optimization plan based on analysis results.
    """

    def __init__(self, memory_budget_mb: float = 1024) -> None:
        """Initialize optimizer."""
        self.memory_budget_mb = memory_budget_mb

    def optimize(
        self,
        analyses: dict[str, KVCacheAnalysis],
        module: RobotModule,
    ) -> dict[str, KVCacheOptPlan]:
        """
        Generate optimization plans for KV caches.

        Args:
            analyses: Analysis results.
            module: Robot IR module.

        Returns:
            Dictionary of buffer name to optimization plan.
        """
        plans = {}

        # Calculate total cache size
        total_size = sum(a.estimated_size_mb for a in analyses.values())

        for name, analysis in analyses.items():
            plan = self._create_plan(analysis, total_size, module)
            plans[name] = plan

        return plans

    def _create_plan(
        self,
        analysis: KVCacheAnalysis,
        total_size: float,
        module: RobotModule,
    ) -> KVCacheOptPlan:
        """Create optimization plan for a single cache."""
        plan = KVCacheOptPlan()

        # Enable sliding window if sequence is long and reuse is high
        if analysis.optimal_window is not None:
            plan.enable_sliding_window = True
            plan.window_size = analysis.optimal_window

        # Enable paged attention for large caches
        if analysis.estimated_size_mb > 256:
            plan.enable_paged_attention = True
            plan.page_size = 16

        # Enable prefill chunking for long sequences
        if analysis.max_seq_len > 1024:
            plan.enable_prefill_chunking = True
            plan.chunk_size = 256

        # Enable flash attention if available
        plan.enable_flash_attention = True

        # Determine if recomputation is beneficial
        if analysis.reuse_ratio < 0.3:
            # Low reuse - might be better to recompute
            plan.enable_sliding_window = False

        # Choose memory layout based on hardware
        if module.deploy and module.deploy.target.name == "THOR":
            # Thor prefers BNSH layout
            plan.memory_layout = "BNSH"
        else:
            plan.memory_layout = "BSNH"

        return plan


class KVCacheOptimizationPass(LoweringPass):
    """
    Lowering pass for KV cache optimization.

    Analyzes cache usage and applies optimization strategies.
    """

    name = "kv_cache_optimization"

    def __init__(self, memory_budget_mb: float = 1024) -> None:
        """Initialize pass."""
        self.analyzer = KVCacheAnalyzer()
        self.optimizer = KVCacheOptimizer(memory_budget_mb)

    def run(self, module: RobotModule, ctx: LoweringContext) -> RobotModule:
        """
        Execute KV cache optimization.

        Args:
            module: Robot IR module.
            ctx: Lowering context.

        Returns:
            Optimized module.
        """
        # Analyze KV caches
        analyses = self.analyzer.analyze(module)

        if not analyses:
            ctx.log("No KV cache buffers found")
            return module

        # Generate optimization plans
        plans = self.optimizer.optimize(analyses, module)

        # Apply optimizations
        for name, plan in plans.items():
            self._apply_plan(name, plan, module, ctx)

        # Store analysis and plans in context
        ctx.set_metadata("kv_cache_analyses", analyses)
        ctx.set_metadata("kv_cache_plans", plans)

        return module

    def _apply_plan(
        self,
        buffer_name: str,
        plan: KVCacheOptPlan,
        module: RobotModule,
        ctx: LoweringContext,
    ) -> None:
        """Apply optimization plan to buffer."""
        buffer = module.get_state(buffer_name)
        if buffer is None:
            return

        # Update buffer metadata with optimization settings
        buffer.metadata = getattr(buffer, "metadata", {})
        buffer.metadata["kv_opt_plan"] = {
            "sliding_window": plan.enable_sliding_window,
            "window_size": plan.window_size,
            "paged_attention": plan.enable_paged_attention,
            "page_size": plan.page_size,
            "flash_attention": plan.enable_flash_attention,
            "memory_layout": plan.memory_layout,
        }

        # Update eviction policy based on optimization
        if plan.enable_sliding_window:
            buffer.eviction_policy = EvictionPolicy.FIFO
            buffer.max_size = plan.window_size

        ctx.log(f"Applied KV optimization to {buffer_name}: "
                f"sliding_window={plan.enable_sliding_window}, "
                f"flash_attention={plan.enable_flash_attention}")

    def should_run(self, module: RobotModule, ctx: LoweringContext) -> bool:
        """Check if pass should run."""
        if module.schedule is None:
            return True
        return module.schedule.kv_reuse_threshold > 0
