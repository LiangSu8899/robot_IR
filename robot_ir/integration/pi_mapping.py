"""Mapping specifications for Pi-series models.

Defines how Pi0, Pi0.5, and related VLA models map to Robot IR.
These mappings ensure the IR accurately reflects the actual model structure
used in robot_runtime adapters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from robot_ir.core.types import TensorShape, DataType
from robot_ir.core.module import RobotModule
from robot_ir.sensors.node import SensorNode, Modality, BufferPolicy, BufferType
from robot_ir.memory.state import StateBuffer, Persistence, ReuseStrategy
from robot_ir.compute.graph import ComputeGraph, DataEdge
from robot_ir.compute.nodes import NeuralBlock, FusionBlock, ControlLogicNode, ControlLogicType
from robot_ir.compute.precision import PrecisionPolicy, Precision
from robot_ir.control.loop import ControlLoop, PipelineMode, RateSpec
from robot_ir.scheduling.hints import SchedulingHints, MemoryHint, THOR_DEFAULT_HINTS


@dataclass
class Pi0Config:
    """
    Configuration for Pi0 IR mapping.

    Maps to robot_runtime's PI0Config structure.
    """

    # Model architecture
    image_size: int = 224
    action_dim: int = 7
    state_dim: int = 14
    chunk_size: int = 50  # Action horizon
    n_obs_steps: int = 1

    # Camera configuration
    num_cameras: int = 2
    camera_names: list[str] = field(default_factory=lambda: ["cam_high", "cam_low"])

    # Runtime behavior
    use_action_cache: bool = True
    temporal_action_steps: int = 1

    # Precision
    vision_precision: Precision = Precision.FP8_E4M3
    policy_precision: Precision = Precision.FP16
    action_precision: Precision = Precision.FP32

    # Control frequency
    control_frequency_hz: float = 20.0
    vision_frequency_hz: float = 10.0
    latency_budget_ms: float = 40.0


@dataclass
class Pi0Mapping:
    """
    Robot IR mapping for Pi0 model architecture.

    Pi0 architecture:
    - Vision encoder: SigLIP ViT-So400M (400M params)
    - Language encoder: Gemma 2B (optional)
    - Action expert: Transformer decoder with flow matching
    - Output: Action chunks (chunk_size x action_dim)

    This mapping produces a RobotModule that accurately represents
    the Pi0 policy structure for lowering to robot_runtime.
    """

    config: Pi0Config = field(default_factory=Pi0Config)

    def create_module(self, name: str = "pi0_policy") -> RobotModule:
        """
        Create Robot IR module for Pi0.

        Returns:
            RobotModule configured for Pi0 architecture.
        """
        module = RobotModule(name=name)

        # Add sensors
        self._add_sensors(module)

        # Add state buffers
        self._add_state_buffers(module)

        # Create compute graph
        module.graph = self._create_compute_graph()

        # Configure control loop
        module.control = self._create_control_loop()

        # Add scheduling hints for Thor
        module.schedule = self._create_scheduling_hints()

        return module

    def _add_sensors(self, module: RobotModule) -> None:
        """Add sensor nodes to module."""
        cfg = self.config

        # Camera inputs
        for i, cam_name in enumerate(cfg.camera_names[:cfg.num_cameras]):
            camera = SensorNode(
                name=cam_name,
                modality=Modality.VISION,
                shape=TensorShape(
                    (3, cfg.image_size, cfg.image_size),
                    DataType.FLOAT16,
                    layout="NCHW",
                ),
                rate_hz=30.0,
                buffering=BufferPolicy(
                    type=BufferType.SLIDING_WINDOW,
                    window_size=cfg.n_obs_steps,
                    reuse_allowed=True,
                ),
                latency_ms=5.0,
            )
            module.add_sensor(camera)

        # Proprioceptive state
        proprio = SensorNode(
            name="observation.state",
            modality=Modality.PROPRIO,
            shape=TensorShape((cfg.state_dim,), DataType.FLOAT32),
            rate_hz=100.0,
            buffering=BufferPolicy(
                type=BufferType.SLIDING_WINDOW,
                window_size=cfg.n_obs_steps,
            ),
        )
        module.add_sensor(proprio)

    def _add_state_buffers(self, module: RobotModule) -> None:
        """Add state buffers for memory management."""
        cfg = self.config

        # Action chunk buffer (for temporal action caching)
        action_buffer = StateBuffer(
            name="action_chunk",
            shape=TensorShape(
                (cfg.chunk_size, cfg.action_dim),
                DataType.FLOAT32,
            ),
            persistence=Persistence.STEP,
            reuse_strategy=ReuseStrategy.ACTION_BUFFER,
            placement="gpu",
        )
        module.add_state(action_buffer)

        # Chunk index state
        chunk_state = StateBuffer(
            name="chunk_index",
            shape=TensorShape((1,), DataType.INT32),
            persistence=Persistence.STEP,
            reuse_strategy=ReuseStrategy.TEMPORAL_STATE,
        )
        module.add_state(chunk_state)

        # Vision feature cache (for multi-step reuse)
        vision_cache = StateBuffer(
            name="vision_features",
            shape=TensorShape(
                (cfg.num_cameras, 256, 1152),  # patches x hidden_dim
                DataType.FLOAT16,
            ),
            persistence=Persistence.STEP,
            reuse_strategy=ReuseStrategy.KV_CACHE,
            placement="gpu",
            prefetch=True,
        )
        module.add_state(vision_cache)

    def _create_compute_graph(self) -> ComputeGraph:
        """Create compute graph for Pi0."""
        cfg = self.config
        graph = ComputeGraph()

        # Vision encoder (SigLIP)
        vision_encoder = NeuralBlock(
            name="vision_encoder",
            model_ref="google/siglip-so400m-patch14-224",
            precision_policy=PrecisionPolicy(
                preferred=cfg.vision_precision,
                critical_layers=["patch_embed", "norm", "head"],
            ),
            memory_hint=MemoryHint(
                prefer_sram=True,
                streaming_layout=True,
                cache_priority=80,
            ),
        )
        graph.add_node(vision_encoder)

        # Proprio encoder (MLP)
        proprio_encoder = NeuralBlock(
            name="proprio_encoder",
            model_ref="pi0_proprio_mlp",
            precision_policy=PrecisionPolicy(preferred=cfg.policy_precision),
        )
        graph.add_node(proprio_encoder)

        # Multi-modal fusion (cross-attention)
        fusion = FusionBlock(
            name="multimodal_fusion",
            fusion_type="cross_attention",
            input_modalities=["vision", "proprio"],
            output_dim=1024,
        )
        graph.add_node(fusion)

        # Action expert (transformer decoder)
        action_expert = NeuralBlock(
            name="action_expert",
            model_ref="pi0_action_transformer",
            precision_policy=PrecisionPolicy(
                preferred=cfg.policy_precision,
                critical_layers=["final_layer_norm"],
            ),
        )
        graph.add_node(action_expert)

        # Flow matching head
        flow_head = NeuralBlock(
            name="flow_matching_head",
            model_ref="pi0_flow_head",
            precision_policy=PrecisionPolicy(
                preferred=cfg.action_precision,  # Keep action head in FP32
            ),
        )
        graph.add_node(flow_head)

        # Action chunk selector (for temporal caching)
        chunk_selector = ControlLogicNode(
            name="action_chunk_selector",
            logic_type=ControlLogicType.SELECTOR,
            config={
                "mode": "temporal_cache",
                "chunk_size": cfg.chunk_size,
                "temporal_steps": cfg.temporal_action_steps,
            },
        )
        graph.add_node(chunk_selector)

        # Connect nodes
        graph.add_edge(DataEdge("vision_encoder", "multimodal_fusion", tensor_name="vision_features"))
        graph.add_edge(DataEdge("proprio_encoder", "multimodal_fusion", tensor_name="proprio_features"))
        graph.add_edge(DataEdge("multimodal_fusion", "action_expert", tensor_name="fused_features"))
        graph.add_edge(DataEdge("action_expert", "flow_matching_head", tensor_name="action_logits"))
        graph.add_edge(DataEdge("flow_matching_head", "action_chunk_selector", tensor_name="action_chunk"))

        graph.entry_points = ["vision_encoder", "proprio_encoder"]
        graph.exit_points = ["action_chunk_selector"]

        return graph

    def _create_control_loop(self) -> ControlLoop:
        """Create control loop configuration."""
        cfg = self.config
        return ControlLoop(
            frequency_hz=cfg.control_frequency_hz,
            latency_budget_ms=cfg.latency_budget_ms,
            pipeline_mode=PipelineMode.STAGGERED,
            jitter_tolerance_ms=5.0,
            multi_rate=[
                RateSpec("vision", cfg.vision_frequency_hz, priority=1),
                RateSpec("policy", cfg.control_frequency_hz, priority=2),
            ],
            deadline_policy="drop",
        )

    def _create_scheduling_hints(self) -> SchedulingHints:
        """Create Thor-optimized scheduling hints."""
        hints = SchedulingHints(
            memory_bound_priority=True,
            kv_reuse_threshold=0.8,
            sensor_fusion_batching=True,
            kernel_fusion_allowed=True,
            enable_cuda_graphs=True,
            tensorrt_enabled=True,
        )

        # Add component-specific memory hints
        hints.set_memory_hint("vision_encoder", MemoryHint(
            prefer_sram=True,
            streaming_layout=True,
            cache_priority=90,
        ))
        hints.set_memory_hint("action_chunk", MemoryHint(
            pin_memory=True,
            placement="gpu",
        ))

        return hints

    def to_runtime_config(self) -> dict:
        """
        Convert to robot_runtime configuration dict.

        Returns:
            Dictionary compatible with RuntimeConfig.
        """
        cfg = self.config
        return {
            "deadline_ms": cfg.latency_budget_ms,
            "timeout_policy": "RETURN_LAST",
            "backend": "cuda_graph",
            "device": "cuda:0",
            "preallocate": True,
            "warmup_iterations": 3,
        }

    def to_pi0_config(self) -> dict:
        """
        Convert to robot_runtime PI0Config dict.

        Returns:
            Dictionary compatible with PI0Config.
        """
        cfg = self.config
        return {
            "chunk_size": cfg.chunk_size,
            "temporal_action_steps": cfg.temporal_action_steps,
            "use_action_cache": cfg.use_action_cache,
            "normalize_observations": True,
            "denormalize_actions": True,
        }


@dataclass
class Pi05Config(Pi0Config):
    """Configuration for Pi0.5 IR mapping."""

    # Pi0.5 uses larger VLM backbone
    vlm_model: str = "google/paligemma-3b-pt-224"
    use_language: bool = True
    max_language_tokens: int = 256

    # Larger model = lower frequency
    control_frequency_hz: float = 10.0
    latency_budget_ms: float = 80.0


@dataclass
class Pi05Mapping:
    """
    Robot IR mapping for Pi0.5 model architecture.

    Pi0.5 extends Pi0 with:
    - Full VLM backbone (PaLiGemma 3B)
    - Language understanding
    - Larger action expert
    """

    config: Pi05Config = field(default_factory=Pi05Config)

    def create_module(self, name: str = "pi05_policy") -> RobotModule:
        """Create Robot IR module for Pi0.5."""
        # Start with Pi0 base
        pi0_mapping = Pi0Mapping(config=Pi0Config(
            image_size=self.config.image_size,
            action_dim=self.config.action_dim,
            state_dim=self.config.state_dim,
            chunk_size=self.config.chunk_size,
            num_cameras=self.config.num_cameras,
            control_frequency_hz=self.config.control_frequency_hz,
            latency_budget_ms=self.config.latency_budget_ms,
        ))
        module = pi0_mapping.create_module(name)

        # Add language sensor
        if self.config.use_language:
            language_sensor = SensorNode(
                name="language_tokens",
                modality=Modality.AUDIO,  # Using AUDIO as placeholder for language
                shape=TensorShape(
                    (self.config.max_language_tokens,),
                    DataType.INT32,
                ),
                rate_hz=1.0,  # Language updates infrequently
                buffering=BufferPolicy(type=BufferType.STREAM),
            )
            module.add_sensor(language_sensor)

            # Add language KV cache
            language_kv = StateBuffer(
                name="language_kv_cache",
                shape=TensorShape(
                    (28, 2, self.config.max_language_tokens, 128),
                    DataType.FLOAT16,
                ),
                persistence=Persistence.EPISODE,  # Persist across steps
                reuse_strategy=ReuseStrategy.KV_CACHE,
                placement="gpu",
            )
            module.add_state(language_kv)

        # Update vision encoder to VLM
        if module.graph:
            vision_node = module.graph.get_node("vision_encoder")
            if vision_node and isinstance(vision_node, NeuralBlock):
                vision_node.model_ref = self.config.vlm_model

        return module


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

    policy_key = policy_name.lower().replace("-", "").replace("_", "")
    if policy_key not in mappings:
        raise ValueError(f"Unknown policy: {policy_name}. Available: {list(mappings.keys())}")

    return mappings[policy_key]()
