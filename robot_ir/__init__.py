"""
Robot IR - Intermediate Representation for Real-time Robot Policy Execution

A unified intermediate layer for expressing real-time robot policy execution graphs:
Sensor -> World Model -> Policy -> Action -> Control Loop

Core features:
- Time-Aware Graph: Support control frequency and multi-rate execution
- Memory-Aware Execution: First-class support for KV cache, state reuse, streaming buffers
- Multi-Agent/Multi-Model Native: Natural support for multi-model coordination
- Hardware-Lowering Friendly: Easy lowering to TensorRT, Triton, CUDA, ROS2
"""

from robot_ir.core.module import RobotModule
from robot_ir.core.types import TensorShape, DataType
from robot_ir.sensors.node import SensorNode, Modality, BufferPolicy, BufferType
from robot_ir.memory.state import StateBuffer, Persistence, ReuseStrategy, EvictionPolicy
from robot_ir.compute.graph import ComputeGraph, DataEdge
from robot_ir.compute.nodes import (
    ComputeNode,
    NeuralBlock,
    FusionBlock,
    ControlLogicNode,
    WorldModelBlock,
)
from robot_ir.compute.precision import PrecisionPolicy, Precision
from robot_ir.control.loop import ControlLoop, PipelineMode
from robot_ir.scheduling.hints import SchedulingHints, MemoryHint
from robot_ir.core.config import DeploymentConfig, TargetPlatform

__version__ = "0.1.0"
__all__ = [
    # Core
    "RobotModule",
    "TensorShape",
    "DataType",
    # Sensors
    "SensorNode",
    "Modality",
    "BufferPolicy",
    "BufferType",
    # Memory
    "StateBuffer",
    "Persistence",
    "ReuseStrategy",
    "EvictionPolicy",
    # Compute
    "ComputeGraph",
    "DataEdge",
    "ComputeNode",
    "NeuralBlock",
    "FusionBlock",
    "ControlLogicNode",
    "WorldModelBlock",
    "PrecisionPolicy",
    "Precision",
    # Control
    "ControlLoop",
    "PipelineMode",
    # Scheduling
    "SchedulingHints",
    "MemoryHint",
    # Config
    "DeploymentConfig",
    "TargetPlatform",
]
