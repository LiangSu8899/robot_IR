"""IR lowering infrastructure.

Lowering path:
    Robot IR -> Execution IR -> Hardware Graph

Execution IR can be mapped to:
- TensorRT graph
- CUDA stream DAG
- ROS2 executor graph
- Triton kernels
"""

from robot_ir.lowering.base import LoweringPass, LoweringContext, PassPipeline
from robot_ir.lowering.passes import (
    KVReusePlanningPass,
    LatencySchedulerPass,
    PrecisionPlannerPass,
    MemoryLayoutPass,
    MultiRateLoopPass,
    KernelFusionPass,
)
from robot_ir.lowering.kv_cache import (
    KVCacheAnalyzer,
    KVCacheOptimizer,
    KVCacheOptimizationPass,
    KVCacheAnalysis,
    KVCacheOptPlan,
)

__all__ = [
    # Base
    "LoweringPass",
    "LoweringContext",
    "PassPipeline",
    # Standard passes
    "KVReusePlanningPass",
    "LatencySchedulerPass",
    "PrecisionPlannerPass",
    "MemoryLayoutPass",
    "MultiRateLoopPass",
    "KernelFusionPass",
    # KV cache optimization
    "KVCacheAnalyzer",
    "KVCacheOptimizer",
    "KVCacheOptimizationPass",
    "KVCacheAnalysis",
    "KVCacheOptPlan",
]
