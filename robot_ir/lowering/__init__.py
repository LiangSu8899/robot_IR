"""IR lowering infrastructure.

Lowering path:
    Robot IR -> Execution IR -> Hardware Graph

Execution IR can be mapped to:
- TensorRT graph
- CUDA stream DAG
- ROS2 executor graph
- Triton kernels
"""

from robot_ir.lowering.base import LoweringPass, LoweringContext
from robot_ir.lowering.passes import (
    KVReusePlanningPass,
    LatencySchedulerPass,
    PrecisionPlannerPass,
    MemoryLayoutPass,
    MultiRateLoopPass,
)

__all__ = [
    "LoweringPass",
    "LoweringContext",
    "KVReusePlanningPass",
    "LatencySchedulerPass",
    "PrecisionPlannerPass",
    "MemoryLayoutPass",
    "MultiRateLoopPass",
]
