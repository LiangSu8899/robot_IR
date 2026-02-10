"""Serialization utilities for Robot IR.

Provides JSON serialization and deserialization for Robot IR modules.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from robot_ir.core.types import TensorShape, DataType
from robot_ir.core.module import RobotModule
from robot_ir.core.config import DeploymentConfig, TargetPlatform
from robot_ir.sensors.node import SensorNode, Modality, BufferPolicy, BufferType
from robot_ir.memory.state import StateBuffer, Persistence, ReuseStrategy, EvictionPolicy
from robot_ir.compute.graph import ComputeGraph, DataEdge
from robot_ir.compute.nodes import NeuralBlock, FusionBlock, ControlLogicNode, WorldModelBlock, ControlLogicType
from robot_ir.compute.precision import PrecisionPolicy, Precision
from robot_ir.control.loop import ControlLoop, PipelineMode, RateSpec
from robot_ir.scheduling.hints import SchedulingHints, MemoryHint, StreamPartition


class IRSerializer:
    """
    Serialize Robot IR modules to JSON.

    Example:
        >>> serializer = IRSerializer()
        >>> json_str = serializer.to_json(module)
        >>> with open("module.json", "w") as f:
        ...     f.write(json_str)
    """

    def to_json(self, module: RobotModule, indent: int = 2) -> str:
        """
        Serialize module to JSON string.

        Args:
            module: Robot IR module.
            indent: JSON indentation level.

        Returns:
            JSON string representation.
        """
        data = self.to_dict(module)
        return json.dumps(data, indent=indent, default=str)

    def to_dict(self, module: RobotModule) -> dict:
        """
        Convert module to dictionary.

        Args:
            module: Robot IR module.

        Returns:
            Dictionary representation.
        """
        return {
            "version": "0.1.0",
            "module": module.to_dict(),
        }

    def save(self, module: RobotModule, path: str | Path) -> None:
        """
        Save module to JSON file.

        Args:
            module: Robot IR module.
            path: Output file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.to_json(module))


class IRDeserializer:
    """
    Deserialize Robot IR modules from JSON.

    Example:
        >>> deserializer = IRDeserializer()
        >>> module = deserializer.from_json(json_str)
    """

    def from_json(self, json_str: str) -> RobotModule:
        """
        Deserialize module from JSON string.

        Args:
            json_str: JSON string.

        Returns:
            Robot IR module.
        """
        data = json.loads(json_str)
        return self.from_dict(data)

    def from_dict(self, data: dict) -> RobotModule:
        """
        Create module from dictionary.

        Args:
            data: Dictionary representation.

        Returns:
            Robot IR module.
        """
        module_data = data.get("module", data)
        return self._parse_module(module_data)

    def load(self, path: str | Path) -> RobotModule:
        """
        Load module from JSON file.

        Args:
            path: Input file path.

        Returns:
            Robot IR module.
        """
        with open(path) as f:
            return self.from_json(f.read())

    def _parse_module(self, data: dict) -> RobotModule:
        """Parse module from dictionary."""
        module = RobotModule(
            name=data["name"],
            version=data.get("version", "0.1.0"),
        )

        # Parse sensors
        for sensor_data in data.get("sensors", []):
            module.add_sensor(self._parse_sensor(sensor_data))

        # Parse states
        for state_data in data.get("states", []):
            module.add_state(self._parse_state(state_data))

        # Parse compute graph
        if data.get("graph"):
            module.graph = self._parse_graph(data["graph"])

        # Parse control loop
        if data.get("control"):
            module.control = self._parse_control(data["control"])

        # Parse scheduling hints
        if data.get("schedule"):
            module.schedule = self._parse_schedule(data["schedule"])

        # Parse deployment config
        if data.get("deploy"):
            module.deploy = self._parse_deploy(data["deploy"])

        return module

    def _parse_sensor(self, data: dict) -> SensorNode:
        """Parse sensor node from dictionary."""
        shape_data = data["shape"]
        shape = TensorShape(
            dims=tuple(shape_data["dims"]),
            dtype=DataType[shape_data.get("dtype", "FLOAT32")],
            layout=shape_data.get("layout", "NCHW"),
        )

        buffering_data = data.get("buffering", {})
        buffering = BufferPolicy(
            type=BufferType[buffering_data.get("type", "STREAM")],
            window_size=buffering_data.get("window_size", 1),
            reuse_allowed=buffering_data.get("reuse_allowed", True),
        )

        return SensorNode(
            name=data["name"],
            modality=Modality[data["modality"]],
            shape=shape,
            rate_hz=data.get("rate_hz", 30.0),
            buffering=buffering,
            latency_ms=data.get("latency_ms", 0.0),
        )

    def _parse_state(self, data: dict) -> StateBuffer:
        """Parse state buffer from dictionary."""
        shape_data = data["shape"]
        shape = TensorShape(
            dims=tuple(shape_data["dims"]),
            dtype=DataType[shape_data.get("dtype", "FLOAT32")],
        )

        return StateBuffer(
            name=data["name"],
            shape=shape,
            persistence=Persistence[data.get("persistence", "STEP")],
            reuse_strategy=ReuseStrategy[data.get("reuse_strategy", "KV_CACHE")],
            eviction_policy=EvictionPolicy[data.get("eviction_policy", "LRU")],
            max_size=data.get("max_size"),
            prefetch=data.get("prefetch", False),
            placement=data.get("placement", "gpu"),
        )

    def _parse_graph(self, data: dict) -> ComputeGraph:
        """Parse compute graph from dictionary."""
        graph = ComputeGraph()

        # Parse nodes
        for node_data in data.get("nodes", []):
            node = self._parse_node(node_data)
            graph.nodes.append(node)

        # Parse edges
        for edge_data in data.get("edges", []):
            edge = DataEdge(
                source=edge_data["source"],
                target=edge_data["target"],
                source_output=edge_data.get("source_output", 0),
                target_input=edge_data.get("target_input", 0),
                tensor_name=edge_data.get("tensor_name", ""),
            )
            graph.edges.append(edge)

        graph.entry_points = data.get("entry_points", [])
        graph.exit_points = data.get("exit_points", [])

        return graph

    def _parse_node(self, data: dict) -> Any:
        """Parse compute node from dictionary."""
        node_type = data.get("type", "NeuralBlock")

        if node_type == "NeuralBlock":
            precision_data = data.get("precision_policy")
            precision_policy = None
            if precision_data:
                precision_policy = PrecisionPolicy(
                    preferred=Precision[precision_data.get("preferred", "FP16")],
                    critical_layers=precision_data.get("critical_layers", []),
                )

            return NeuralBlock(
                name=data["name"],
                model_ref=data.get("model_ref", ""),
                precision_policy=precision_policy,
                is_trainable=data.get("is_trainable", False),
                checkpoint_path=data.get("checkpoint_path"),
            )

        elif node_type == "FusionBlock":
            return FusionBlock(
                name=data["name"],
                fusion_type=data.get("fusion_type", "concat"),
                input_modalities=data.get("input_modalities", []),
                output_dim=data.get("output_dim", 512),
                num_experts=data.get("num_experts"),
                top_k=data.get("top_k"),
            )

        elif node_type == "ControlLogicNode":
            return ControlLogicNode(
                name=data["name"],
                logic_type=ControlLogicType[data.get("logic_type", "SELECTOR")],
                config=data.get("config", {}),
            )

        elif node_type == "WorldModelBlock":
            return WorldModelBlock(
                name=data["name"],
                latent_dim=data.get("latent_dim", 512),
                rollout_steps=data.get("rollout_steps", 1),
                compression_ratio=data.get("compression_ratio", 1.0),
                action_conditioned=data.get("action_conditioned", True),
                stochastic=data.get("stochastic", False),
            )

        raise ValueError(f"Unknown node type: {node_type}")

    def _parse_control(self, data: dict) -> ControlLoop:
        """Parse control loop from dictionary."""
        multi_rate = []
        for rate_data in data.get("multi_rate", []):
            multi_rate.append(RateSpec(
                name=rate_data["name"],
                frequency_hz=rate_data["frequency_hz"],
                priority=rate_data.get("priority", 0),
            ))

        return ControlLoop(
            frequency_hz=data.get("frequency_hz", 20.0),
            latency_budget_ms=data.get("latency_budget_ms", 50.0),
            pipeline_mode=PipelineMode[data.get("pipeline_mode", "SYNC")],
            jitter_tolerance_ms=data.get("jitter_tolerance_ms", 5.0),
            multi_rate=multi_rate,
            deadline_policy=data.get("deadline_policy", "drop"),
        )

    def _parse_schedule(self, data: dict) -> SchedulingHints:
        """Parse scheduling hints from dictionary."""
        memory_hints = {}
        for name, hint_data in data.get("memory_hints", {}).items():
            memory_hints[name] = MemoryHint(
                prefer_sram=hint_data.get("prefer_sram", False),
                streaming_layout=hint_data.get("streaming_layout", False),
                pin_memory=hint_data.get("pin_memory", True),
                placement=hint_data.get("placement", "gpu"),
                cache_priority=hint_data.get("cache_priority", 50),
            )

        return SchedulingHints(
            memory_bound_priority=data.get("memory_bound_priority", True),
            kv_reuse_threshold=data.get("kv_reuse_threshold", 0.5),
            sensor_fusion_batching=data.get("sensor_fusion_batching", True),
            kernel_fusion_allowed=data.get("kernel_fusion_allowed", True),
            stream_partition=StreamPartition[data.get("stream_partition", "PER_STAGE")],
            max_concurrent_streams=data.get("max_concurrent_streams", 4),
            enable_cuda_graphs=data.get("enable_cuda_graphs", True),
            tensorrt_enabled=data.get("tensorrt_enabled", True),
            memory_hints=memory_hints,
        )

    def _parse_deploy(self, data: dict) -> DeploymentConfig:
        """Parse deployment config from dictionary."""
        return DeploymentConfig(
            target=TargetPlatform[data.get("target", "THOR")],
            realtime=data.get("realtime", True),
            multi_gpu=data.get("multi_gpu", False),
            ros_bridge=data.get("ros_bridge", False),
            max_memory_mb=data.get("max_memory_mb"),
            enable_profiling=data.get("enable_profiling", False),
        )


# Convenience functions
def save_module(module: RobotModule, path: str | Path) -> None:
    """Save module to JSON file."""
    IRSerializer().save(module, path)


def load_module(path: str | Path) -> RobotModule:
    """Load module from JSON file."""
    return IRDeserializer().load(path)


def to_json(module: RobotModule) -> str:
    """Convert module to JSON string."""
    return IRSerializer().to_json(module)


def from_json(json_str: str) -> RobotModule:
    """Create module from JSON string."""
    return IRDeserializer().from_json(json_str)
