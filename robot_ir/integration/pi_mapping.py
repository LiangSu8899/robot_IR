"""Mapping specifications for Pi-series models.

Defines how Pi0, Pi0.5, and related VLA models map to Robot IR.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from robot_ir.core.types import TensorShape, DataType
from robot_ir.core.module import RobotModule
from robot_ir.sensors.node import SensorNode, Modality, BufferPolicy, BufferType
from robot_ir.memory.state import StateBuffer, Persistence, ReuseStrategy
from robot_ir.compute.graph import ComputeGraph, DataEdge
from robot_ir.compute.nodes import NeuralBlock, FusionBlock
from robot_ir.compute.precision import PrecisionPolicy, Precision
from robot_ir.control.loop import ControlLoop, PipelineMode, RateSpec
from robot_ir.scheduling.hints import SchedulingHints, MemoryHint


@dataclass
class Pi0Mapping:
    """
    Robot IR mapping for Pi0 model architecture.

    Pi0 architecture:
    - Vision encoder: SigLIP ViT-So400M
    - Language encoder: Gemma 2B (optional)
    - Action expert: Transformer decoder with flow matching

    Attributes:
        image_size: Input image size.
        action_dim: Action dimension.
        action_horizon: Action chunk size.
        num_cameras: Number of camera inputs.
        use_language: Whether to use language conditioning.
    """

    image_size: int = 224
    action_dim: int = 7
    action_horizon: int = 50
    num_cameras: int = 2
    use_language: bool = True
    proprio_dim: int = 14

    def create_module(self, name: str = "pi0_policy") -> RobotModule:
        """
        Create Robot IR module for Pi0.

        Returns:
            RobotModule configured for Pi0 architecture.
        """
        module = RobotModule(name=name)

        # Sensors
        for i in range(self.num_cameras):
            camera = SensorNode(
                name=f"camera_{i}",
                modality=Modality.VISION,
                shape=TensorShape((3, self.image_size, self.image_size), DataType.FLOAT16),
                rate_hz=30.0,
                buffering=BufferPolicy(type=BufferType.STREAM, reuse_allowed=True),
            )
            module.add_sensor(camera)

        proprio = SensorNode(
            name="proprio",
            modality=Modality.PROPRIO,
            shape=TensorShape((self.proprio_dim,), DataType.FLOAT32),
            rate_hz=100.0,
        )
        module.add_sensor(proprio)

        # State buffers
        kv_cache = StateBuffer(
            name="transformer_kv",
            shape=TensorShape((24, 2, 256, 64), DataType.FLOAT16),  # layers, kv, seq, head_dim
            persistence=Persistence.STEP,
            reuse_strategy=ReuseStrategy.KV_CACHE,
            placement="gpu",
        )
        module.add_state(kv_cache)

        action_buffer = StateBuffer(
            name="action_chunk",
            shape=TensorShape((self.action_horizon, self.action_dim), DataType.FLOAT32),
            persistence=Persistence.STEP,
            reuse_strategy=ReuseStrategy.ACTION_BUFFER,
        )
        module.add_state(action_buffer)

        # Compute graph
        graph = self._create_compute_graph()
        module.graph = graph

        # Control loop
        module.control = ControlLoop(
            frequency_hz=20.0,
            latency_budget_ms=40.0,
            pipeline_mode=PipelineMode.SYNC,
            multi_rate=[
                RateSpec("vision", 10.0, priority=1),
                RateSpec("policy", 20.0, priority=2),
            ],
        )

        # Scheduling hints
        module.schedule = SchedulingHints(
            memory_bound_priority=True,
            kv_reuse_threshold=0.8,
            enable_cuda_graphs=True,
            tensorrt_enabled=True,
        )

        return module

    def _create_compute_graph(self) -> ComputeGraph:
        """Create compute graph for Pi0."""
        graph = ComputeGraph()

        # Vision encoder
        vision_encoder = NeuralBlock(
            name="vision_encoder",
            model_ref="siglip_so400m_patch14_224",
            precision_policy=PrecisionPolicy(
                preferred=Precision.FP8_E4M3,
                critical_layers=["patch_embed", "norm"],
            ),
        )
        graph.add_node(vision_encoder)

        # Fusion block
        fusion = FusionBlock(
            name="multimodal_fusion",
            fusion_type="attention",
            input_modalities=["vision", "proprio", "language"],
            output_dim=1024,
        )
        graph.add_node(fusion)

        # Policy decoder
        policy_decoder = NeuralBlock(
            name="policy_decoder",
            model_ref="pi0_action_expert",
            precision_policy=PrecisionPolicy(
                preferred=Precision.FP16,
                critical_layers=["flow_head"],
            ),
        )
        graph.add_node(policy_decoder)

        # Action head
        action_head = NeuralBlock(
            name="action_head",
            model_ref="flow_matching_head",
            precision_policy=PrecisionPolicy(preferred=Precision.FP32),  # Keep action head in FP32
        )
        graph.add_node(action_head)

        # Edges
        graph.add_edge(DataEdge("vision_encoder", "multimodal_fusion"))
        graph.add_edge(DataEdge("multimodal_fusion", "policy_decoder"))
        graph.add_edge(DataEdge("policy_decoder", "action_head"))

        graph.entry_points = ["vision_encoder"]
        graph.exit_points = ["action_head"]

        return graph


@dataclass
class Pi05Mapping:
    """
    Robot IR mapping for Pi0.5 model architecture.

    Pi0.5 extends Pi0 with:
    - Full VLM backbone (PaLiGemma 3B)
    - Language understanding
    - Larger action expert
    """

    image_size: int = 224
    action_dim: int = 7
    action_horizon: int = 50
    num_cameras: int = 2
    proprio_dim: int = 14

    def create_module(self, name: str = "pi05_policy") -> RobotModule:
        """
        Create Robot IR module for Pi0.5.

        Returns:
            RobotModule configured for Pi0.5 architecture.
        """
        # Start with Pi0 base
        pi0 = Pi0Mapping(
            image_size=self.image_size,
            action_dim=self.action_dim,
            action_horizon=self.action_horizon,
            num_cameras=self.num_cameras,
            use_language=True,
            proprio_dim=self.proprio_dim,
        )
        module = pi0.create_module(name)

        # Update for Pi0.5 specifics
        # Larger KV cache for VLM
        for state in module.states:
            if state.name == "transformer_kv":
                state.shape = TensorShape((28, 2, 512, 128), DataType.FLOAT16)

        # Update control loop for larger model
        module.control = ControlLoop(
            frequency_hz=10.0,  # Lower freq for larger model
            latency_budget_ms=80.0,
            pipeline_mode=PipelineMode.STAGGERED,
        )

        return module


# Convenience function to get mapping by name
def get_policy_mapping(policy_name: str) -> Pi0Mapping | Pi05Mapping:
    """
    Get policy mapping by name.

    Args:
        policy_name: Name of the policy ("pi0", "pi0.5", etc.)

    Returns:
        Appropriate mapping instance.
    """
    mappings = {
        "pi0": Pi0Mapping,
        "pi0.5": Pi05Mapping,
        "pi05": Pi05Mapping,
    }

    if policy_name.lower() not in mappings:
        raise ValueError(f"Unknown policy: {policy_name}. Available: {list(mappings.keys())}")

    return mappings[policy_name.lower()]()
