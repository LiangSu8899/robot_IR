"""Mapping specifications for ACT, Diffusion Policy, and OpenVLA models.

Defines how these popular robot learning policies map to Robot IR.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from robot_ir.core.types import TensorShape, DataType
from robot_ir.core.module import RobotModule
from robot_ir.sensors.node import SensorNode, Modality, BufferPolicy, BufferType
from robot_ir.memory.state import StateBuffer, Persistence, ReuseStrategy
from robot_ir.compute.graph import ComputeGraph, DataEdge
from robot_ir.compute.nodes import NeuralBlock, FusionBlock, ControlLogicNode, ControlLogicType
from robot_ir.compute.precision import PrecisionPolicy, Precision
from robot_ir.control.loop import ControlLoop, PipelineMode, RateSpec
from robot_ir.scheduling.hints import SchedulingHints, MemoryHint


class DiffusionScheduler(Enum):
    """Diffusion noise scheduler types."""
    DDPM = "ddpm"
    DDIM = "ddim"
    DPM_PP = "dpm_pp"


# =============================================================================
# ACT (Action Chunking with Transformers)
# =============================================================================


@dataclass
class ACTConfig:
    """
    Configuration for ACT IR mapping.

    ACT: Action Chunking with Transformers
    - Transformer encoder-decoder architecture
    - CVAE for action generation
    - Temporal ensemble for smooth actions
    """

    # Model architecture
    image_size: int = 224
    action_dim: int = 7
    state_dim: int = 14
    chunk_size: int = 100  # ACT typically uses 100

    # Encoder config
    hidden_dim: int = 512
    num_encoder_layers: int = 4
    num_decoder_layers: int = 7
    num_heads: int = 8
    feedforward_dim: int = 3200

    # Camera configuration
    num_cameras: int = 2
    camera_names: list[str] = field(default_factory=lambda: ["cam_high", "cam_wrist"])

    # CVAE settings
    latent_dim: int = 32
    kl_weight: float = 10.0

    # Temporal ensemble
    use_temporal_ensemble: bool = True
    temporal_weight: float = 0.01

    # Precision
    encoder_precision: Precision = Precision.FP16
    decoder_precision: Precision = Precision.FP16
    action_precision: Precision = Precision.FP32

    # Control frequency
    control_frequency_hz: float = 50.0
    vision_frequency_hz: float = 30.0
    latency_budget_ms: float = 20.0


@dataclass
class ACTMapping:
    """
    Robot IR mapping for ACT model architecture.

    ACT architecture:
    - Vision backbone: ResNet18 (pretrained)
    - Transformer encoder for observations
    - CVAE latent sampling
    - Transformer decoder for action generation
    - Temporal ensemble for smoothing

    Reference: https://github.com/tonyzhaozh/act
    """

    config: ACTConfig = field(default_factory=ACTConfig)

    def create_module(self, name: str = "act_policy") -> RobotModule:
        """Create Robot IR module for ACT."""
        module = RobotModule(name=name)

        self._add_sensors(module)
        self._add_state_buffers(module)
        module.graph = self._create_compute_graph()
        module.control = self._create_control_loop()
        module.schedule = self._create_scheduling_hints()

        return module

    def _add_sensors(self, module: RobotModule) -> None:
        """Add sensor nodes to module."""
        cfg = self.config

        # Camera inputs
        for cam_name in cfg.camera_names[:cfg.num_cameras]:
            camera = SensorNode(
                name=cam_name,
                modality=Modality.VISION,
                shape=TensorShape(
                    (3, cfg.image_size, cfg.image_size),
                    DataType.FLOAT16,
                    layout="NCHW",
                ),
                rate_hz=cfg.vision_frequency_hz,
                buffering=BufferPolicy(
                    type=BufferType.SLIDING_WINDOW,
                    window_size=1,
                    reuse_allowed=True,
                ),
                latency_ms=3.0,
            )
            module.add_sensor(camera)

        # Proprioceptive state (joint positions)
        proprio = SensorNode(
            name="qpos",
            modality=Modality.PROPRIO,
            shape=TensorShape((cfg.state_dim,), DataType.FLOAT32),
            rate_hz=100.0,
            buffering=BufferPolicy(type=BufferType.STREAM),
        )
        module.add_sensor(proprio)

    def _add_state_buffers(self, module: RobotModule) -> None:
        """Add state buffers for memory management."""
        cfg = self.config

        # Action chunk buffer
        action_buffer = StateBuffer(
            name="action_chunk",
            shape=TensorShape((cfg.chunk_size, cfg.action_dim), DataType.FLOAT32),
            persistence=Persistence.STEP,
            reuse_strategy=ReuseStrategy.ACTION_BUFFER,
            placement="gpu",
        )
        module.add_state(action_buffer)

        # Temporal ensemble buffer
        if cfg.use_temporal_ensemble:
            ensemble_buffer = StateBuffer(
                name="temporal_ensemble",
                shape=TensorShape((cfg.chunk_size, cfg.action_dim), DataType.FLOAT32),
                persistence=Persistence.STEP,
                reuse_strategy=ReuseStrategy.TEMPORAL_STATE,
                placement="gpu",
            )
            module.add_state(ensemble_buffer)

        # CVAE latent buffer
        latent_buffer = StateBuffer(
            name="cvae_latent",
            shape=TensorShape((cfg.latent_dim,), DataType.FLOAT32),
            persistence=Persistence.STEP,
            reuse_strategy=ReuseStrategy.TEMPORAL_STATE,
            placement="gpu",
        )
        module.add_state(latent_buffer)

    def _create_compute_graph(self) -> ComputeGraph:
        """Create compute graph for ACT."""
        cfg = self.config
        graph = ComputeGraph()

        # Vision backbone (ResNet18)
        vision_backbone = NeuralBlock(
            name="vision_backbone",
            model_ref="resnet18",
            precision_policy=PrecisionPolicy(preferred=cfg.encoder_precision),
            memory_hint=MemoryHint(prefer_sram=True, cache_priority=80),
        )
        graph.add_node(vision_backbone)

        # Proprio encoder (MLP)
        proprio_encoder = NeuralBlock(
            name="proprio_encoder",
            model_ref="act_proprio_mlp",
            precision_policy=PrecisionPolicy(preferred=cfg.encoder_precision),
        )
        graph.add_node(proprio_encoder)

        # Transformer encoder
        transformer_encoder = NeuralBlock(
            name="transformer_encoder",
            model_ref="act_encoder",
            precision_policy=PrecisionPolicy(preferred=cfg.encoder_precision),
            metadata={
                "num_layers": cfg.num_encoder_layers,
                "hidden_dim": cfg.hidden_dim,
                "num_heads": cfg.num_heads,
            },
        )
        graph.add_node(transformer_encoder)

        # CVAE latent sampling
        cvae_sampler = NeuralBlock(
            name="cvae_sampler",
            model_ref="act_cvae",
            precision_policy=PrecisionPolicy(preferred=Precision.FP32),
            metadata={"latent_dim": cfg.latent_dim},
        )
        graph.add_node(cvae_sampler)

        # Transformer decoder
        transformer_decoder = NeuralBlock(
            name="transformer_decoder",
            model_ref="act_decoder",
            precision_policy=PrecisionPolicy(preferred=cfg.decoder_precision),
            metadata={
                "num_layers": cfg.num_decoder_layers,
                "hidden_dim": cfg.hidden_dim,
                "num_heads": cfg.num_heads,
            },
        )
        graph.add_node(transformer_decoder)

        # Action head
        action_head = NeuralBlock(
            name="action_head",
            model_ref="act_action_mlp",
            precision_policy=PrecisionPolicy(preferred=cfg.action_precision),
        )
        graph.add_node(action_head)

        # Temporal ensemble (if enabled)
        if cfg.use_temporal_ensemble:
            ensemble_node = ControlLogicNode(
                name="temporal_ensemble",
                logic_type=ControlLogicType.AGGREGATOR,
                config={
                    "mode": "exponential_weighted_mean",
                    "weight": cfg.temporal_weight,
                    "chunk_size": cfg.chunk_size,
                },
            )
            graph.add_node(ensemble_node)

        # Connect nodes
        graph.add_edge(DataEdge("vision_backbone", "transformer_encoder", tensor_name="vision_tokens"))
        graph.add_edge(DataEdge("proprio_encoder", "transformer_encoder", tensor_name="proprio_embed"))
        graph.add_edge(DataEdge("transformer_encoder", "cvae_sampler", tensor_name="encoder_output"))
        graph.add_edge(DataEdge("cvae_sampler", "transformer_decoder", tensor_name="latent_z"))
        graph.add_edge(DataEdge("transformer_encoder", "transformer_decoder", tensor_name="memory"))
        graph.add_edge(DataEdge("transformer_decoder", "action_head", tensor_name="decoder_output"))

        if cfg.use_temporal_ensemble:
            graph.add_edge(DataEdge("action_head", "temporal_ensemble", tensor_name="raw_actions"))
            graph.exit_points = ["temporal_ensemble"]
        else:
            graph.exit_points = ["action_head"]

        graph.entry_points = ["vision_backbone", "proprio_encoder"]

        return graph

    def _create_control_loop(self) -> ControlLoop:
        """Create control loop configuration."""
        cfg = self.config
        return ControlLoop(
            frequency_hz=cfg.control_frequency_hz,
            latency_budget_ms=cfg.latency_budget_ms,
            pipeline_mode=PipelineMode.STREAMING,
            jitter_tolerance_ms=2.0,
            multi_rate=[
                RateSpec("vision", cfg.vision_frequency_hz, priority=1),
                RateSpec("policy", cfg.control_frequency_hz, priority=2),
            ],
            deadline_policy="drop",
        )

    def _create_scheduling_hints(self) -> SchedulingHints:
        """Create scheduling hints."""
        hints = SchedulingHints(
            memory_bound_priority=True,
            sensor_fusion_batching=True,
            kernel_fusion_allowed=True,
            enable_cuda_graphs=True,
            tensorrt_enabled=True,
        )

        hints.set_memory_hint("vision_backbone", MemoryHint(
            prefer_sram=True,
            streaming_layout=True,
            cache_priority=90,
        ))

        return hints

    def to_runtime_config(self) -> dict:
        """Convert to robot_runtime configuration dict."""
        cfg = self.config
        return {
            "deadline_ms": cfg.latency_budget_ms,
            "timeout_policy": "RETURN_LAST",
            "backend": "cuda_graph",
            "device": "cuda:0",
            "preallocate": True,
            "warmup_iterations": 3,
        }


# =============================================================================
# Diffusion Policy
# =============================================================================


@dataclass
class DiffusionPolicyConfig:
    """
    Configuration for Diffusion Policy IR mapping.

    Diffusion Policy: Visuomotor Policy Learning via Action Diffusion
    - Conditional denoising diffusion for action generation
    - U-Net or Transformer-based denoiser
    """

    # Model architecture
    image_size: int = 96  # Default for diffusion policy
    action_dim: int = 2  # Default for push-t
    state_dim: int = 5
    action_horizon: int = 16
    observation_horizon: int = 2
    prediction_horizon: int = 16

    # Diffusion settings
    num_diffusion_steps: int = 100
    num_inference_steps: int = 10  # DDIM steps
    scheduler: DiffusionScheduler = DiffusionScheduler.DDIM

    # Architecture type
    use_transformer: bool = False  # False = U-Net, True = Transformer
    hidden_dim: int = 256

    # Camera configuration
    num_cameras: int = 1
    camera_names: list[str] = field(default_factory=lambda: ["image"])

    # Precision
    encoder_precision: Precision = Precision.FP16
    diffusion_precision: Precision = Precision.FP16
    action_precision: Precision = Precision.FP32

    # Control frequency
    control_frequency_hz: float = 10.0
    latency_budget_ms: float = 100.0  # Diffusion is slower


@dataclass
class DiffusionPolicyMapping:
    """
    Robot IR mapping for Diffusion Policy architecture.

    Diffusion Policy architecture:
    - Vision encoder (ResNet)
    - Observation encoder
    - U-Net or Transformer denoiser
    - DDIM sampling for fast inference

    Reference: https://github.com/real-stanford/diffusion_policy
    """

    config: DiffusionPolicyConfig = field(default_factory=DiffusionPolicyConfig)

    def create_module(self, name: str = "diffusion_policy") -> RobotModule:
        """Create Robot IR module for Diffusion Policy."""
        module = RobotModule(name=name)

        self._add_sensors(module)
        self._add_state_buffers(module)
        module.graph = self._create_compute_graph()
        module.control = self._create_control_loop()
        module.schedule = self._create_scheduling_hints()

        return module

    def _add_sensors(self, module: RobotModule) -> None:
        """Add sensor nodes to module."""
        cfg = self.config

        # Camera inputs
        for cam_name in cfg.camera_names[:cfg.num_cameras]:
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
                    window_size=cfg.observation_horizon,
                    reuse_allowed=True,
                ),
            )
            module.add_sensor(camera)

        # State input
        state = SensorNode(
            name="agent_pos",
            modality=Modality.PROPRIO,
            shape=TensorShape((cfg.state_dim,), DataType.FLOAT32),
            rate_hz=100.0,
            buffering=BufferPolicy(
                type=BufferType.SLIDING_WINDOW,
                window_size=cfg.observation_horizon,
            ),
        )
        module.add_sensor(state)

    def _add_state_buffers(self, module: RobotModule) -> None:
        """Add state buffers for memory management."""
        cfg = self.config

        # Action buffer
        action_buffer = StateBuffer(
            name="action_trajectory",
            shape=TensorShape(
                (cfg.prediction_horizon, cfg.action_dim),
                DataType.FLOAT32,
            ),
            persistence=Persistence.STEP,
            reuse_strategy=ReuseStrategy.ACTION_BUFFER,
            placement="gpu",
        )
        module.add_state(action_buffer)

        # Noise trajectory (for diffusion)
        noise_buffer = StateBuffer(
            name="noise_trajectory",
            shape=TensorShape(
                (cfg.prediction_horizon, cfg.action_dim),
                DataType.FLOAT32,
            ),
            persistence=Persistence.STEP,
            reuse_strategy=ReuseStrategy.TEMPORAL_STATE,
            placement="gpu",
        )
        module.add_state(noise_buffer)

        # Observation embedding cache
        obs_cache = StateBuffer(
            name="obs_embedding_cache",
            shape=TensorShape((cfg.hidden_dim,), DataType.FLOAT16),
            persistence=Persistence.STEP,
            reuse_strategy=ReuseStrategy.KV_CACHE,
            placement="gpu",
        )
        module.add_state(obs_cache)

    def _create_compute_graph(self) -> ComputeGraph:
        """Create compute graph for Diffusion Policy."""
        cfg = self.config
        graph = ComputeGraph()

        # Vision encoder
        vision_encoder = NeuralBlock(
            name="vision_encoder",
            model_ref="resnet18_diffusion",
            precision_policy=PrecisionPolicy(preferred=cfg.encoder_precision),
            memory_hint=MemoryHint(prefer_sram=True, cache_priority=80),
        )
        graph.add_node(vision_encoder)

        # Observation encoder (combines vision + state history)
        obs_encoder = FusionBlock(
            name="obs_encoder",
            fusion_type="concat_linear",
            input_modalities=["vision", "state"],
            output_dim=cfg.hidden_dim,
        )
        graph.add_node(obs_encoder)

        # Noise sampler (initial random noise)
        noise_sampler = ControlLogicNode(
            name="noise_sampler",
            logic_type=ControlLogicType.GENERATOR,
            config={
                "mode": "gaussian_noise",
                "shape": (cfg.prediction_horizon, cfg.action_dim),
            },
        )
        graph.add_node(noise_sampler)

        # Denoiser (U-Net or Transformer)
        if cfg.use_transformer:
            denoiser = NeuralBlock(
                name="denoiser",
                model_ref="diffusion_transformer",
                precision_policy=PrecisionPolicy(preferred=cfg.diffusion_precision),
                metadata={
                    "num_inference_steps": cfg.num_inference_steps,
                    "scheduler": cfg.scheduler.value,
                },
            )
        else:
            denoiser = NeuralBlock(
                name="denoiser",
                model_ref="diffusion_unet",
                precision_policy=PrecisionPolicy(preferred=cfg.diffusion_precision),
                metadata={
                    "num_inference_steps": cfg.num_inference_steps,
                    "scheduler": cfg.scheduler.value,
                    "hidden_dim": cfg.hidden_dim,
                },
            )
        graph.add_node(denoiser)

        # Diffusion loop controller
        diffusion_loop = ControlLogicNode(
            name="diffusion_loop",
            logic_type=ControlLogicType.LOOP,
            config={
                "num_iterations": cfg.num_inference_steps,
                "mode": "ddim_sampling",
            },
        )
        graph.add_node(diffusion_loop)

        # Action selector (select action_horizon from prediction_horizon)
        action_selector = ControlLogicNode(
            name="action_selector",
            logic_type=ControlLogicType.SELECTOR,
            config={
                "mode": "slice",
                "start": 0,
                "end": cfg.action_horizon,
            },
        )
        graph.add_node(action_selector)

        # Connect nodes
        graph.add_edge(DataEdge("vision_encoder", "obs_encoder", tensor_name="vision_features"))
        graph.add_edge(DataEdge("obs_encoder", "denoiser", tensor_name="obs_embedding"))
        graph.add_edge(DataEdge("noise_sampler", "diffusion_loop", tensor_name="initial_noise"))
        graph.add_edge(DataEdge("denoiser", "diffusion_loop", tensor_name="denoised"))
        graph.add_edge(DataEdge("diffusion_loop", "action_selector", tensor_name="action_trajectory"))

        graph.entry_points = ["vision_encoder", "noise_sampler"]
        graph.exit_points = ["action_selector"]

        return graph

    def _create_control_loop(self) -> ControlLoop:
        """Create control loop configuration."""
        cfg = self.config
        return ControlLoop(
            frequency_hz=cfg.control_frequency_hz,
            latency_budget_ms=cfg.latency_budget_ms,
            pipeline_mode=PipelineMode.STANDARD,
            jitter_tolerance_ms=10.0,
            deadline_policy="extend",  # Diffusion may need extra time
        )

    def _create_scheduling_hints(self) -> SchedulingHints:
        """Create scheduling hints."""
        return SchedulingHints(
            memory_bound_priority=True,
            kernel_fusion_allowed=True,
            enable_cuda_graphs=False,  # Diffusion loop is dynamic
            tensorrt_enabled=True,
        )

    def to_runtime_config(self) -> dict:
        """Convert to robot_runtime configuration dict."""
        cfg = self.config
        return {
            "deadline_ms": cfg.latency_budget_ms,
            "timeout_policy": "EXTEND",
            "backend": "eager",  # Diffusion needs dynamic control flow
            "device": "cuda:0",
            "preallocate": True,
            "warmup_iterations": 5,
        }


# =============================================================================
# OpenVLA
# =============================================================================


@dataclass
class OpenVLAConfig:
    """
    Configuration for OpenVLA IR mapping.

    OpenVLA: Open Vision-Language-Action Model
    - Vision encoder + LLM backbone
    - Language-conditioned robot control
    - Action tokenization
    """

    # Model architecture
    image_size: int = 224
    action_dim: int = 7
    action_vocab_size: int = 256

    # VLM settings
    vlm_backbone: str = "Qwen/Qwen2-VL-2B-Instruct"
    max_new_tokens: int = 256
    use_bf16: bool = True

    # Camera configuration
    num_cameras: int = 1
    camera_names: list[str] = field(default_factory=lambda: ["image"])

    # Language settings
    max_language_tokens: int = 512
    use_language_cache: bool = True

    # Action tokenization
    action_bins: int = 256
    use_action_cache: bool = True

    # Precision
    vision_precision: Precision = Precision.FP16
    llm_precision: Precision = Precision.FP16
    action_precision: Precision = Precision.FP32

    # Control frequency (OpenVLA is slower due to LLM)
    control_frequency_hz: float = 5.0
    latency_budget_ms: float = 200.0


@dataclass
class OpenVLAMapping:
    """
    Robot IR mapping for OpenVLA architecture.

    OpenVLA architecture:
    - Vision encoder (SigLIP or similar)
    - LLM backbone (Qwen2-VL, Llama, etc.)
    - Action tokenizer/detokenizer
    - Autoregressive action generation

    Reference: https://github.com/openvla/openvla
    """

    config: OpenVLAConfig = field(default_factory=OpenVLAConfig)

    def create_module(self, name: str = "openvla_policy") -> RobotModule:
        """Create Robot IR module for OpenVLA."""
        module = RobotModule(name=name)

        self._add_sensors(module)
        self._add_state_buffers(module)
        module.graph = self._create_compute_graph()
        module.control = self._create_control_loop()
        module.schedule = self._create_scheduling_hints()

        return module

    def _add_sensors(self, module: RobotModule) -> None:
        """Add sensor nodes to module."""
        cfg = self.config

        # Camera input
        for cam_name in cfg.camera_names[:cfg.num_cameras]:
            camera = SensorNode(
                name=cam_name,
                modality=Modality.VISION,
                shape=TensorShape(
                    (3, cfg.image_size, cfg.image_size),
                    DataType.FLOAT16,
                    layout="NCHW",
                ),
                rate_hz=10.0,
                buffering=BufferPolicy(
                    type=BufferType.STREAM,
                    reuse_allowed=True,
                ),
            )
            module.add_sensor(camera)

        # Language instruction
        language = SensorNode(
            name="language_instruction",
            modality=Modality.AUDIO,  # Using AUDIO as language placeholder
            shape=TensorShape((cfg.max_language_tokens,), DataType.INT32),
            rate_hz=0.1,  # Language rarely changes
            buffering=BufferPolicy(type=BufferType.STREAM),
        )
        module.add_sensor(language)

    def _add_state_buffers(self, module: RobotModule) -> None:
        """Add state buffers for memory management."""
        cfg = self.config

        # LLM KV cache
        kv_cache = StateBuffer(
            name="llm_kv_cache",
            shape=TensorShape(
                (32, 2, cfg.max_language_tokens + 256, 128),  # layers, k/v, seq, head_dim
                DataType.FLOAT16,
            ),
            persistence=Persistence.EPISODE,
            reuse_strategy=ReuseStrategy.KV_CACHE,
            placement="gpu",
        )
        module.add_state(kv_cache)

        # Action token buffer
        action_buffer = StateBuffer(
            name="action_tokens",
            shape=TensorShape((cfg.action_dim,), DataType.INT32),
            persistence=Persistence.STEP,
            reuse_strategy=ReuseStrategy.ACTION_BUFFER,
            placement="gpu",
        )
        module.add_state(action_buffer)

        # Vision embedding cache (for multi-step reuse)
        if cfg.use_language_cache:
            vision_cache = StateBuffer(
                name="vision_embedding_cache",
                shape=TensorShape((256, 4096), DataType.FLOAT16),
                persistence=Persistence.STEP,
                reuse_strategy=ReuseStrategy.KV_CACHE,
                placement="gpu",
                prefetch=True,
            )
            module.add_state(vision_cache)

    def _create_compute_graph(self) -> ComputeGraph:
        """Create compute graph for OpenVLA."""
        cfg = self.config
        graph = ComputeGraph()

        # Vision encoder
        vision_encoder = NeuralBlock(
            name="vision_encoder",
            model_ref="siglip_vit",
            precision_policy=PrecisionPolicy(preferred=cfg.vision_precision),
            memory_hint=MemoryHint(prefer_sram=True, cache_priority=90),
        )
        graph.add_node(vision_encoder)

        # Vision projector (align vision to LLM dim)
        vision_projector = NeuralBlock(
            name="vision_projector",
            model_ref="openvla_projector",
            precision_policy=PrecisionPolicy(preferred=cfg.vision_precision),
        )
        graph.add_node(vision_projector)

        # Language tokenizer embedding
        language_embed = NeuralBlock(
            name="language_embedding",
            model_ref="qwen2_embed",
            precision_policy=PrecisionPolicy(preferred=cfg.llm_precision),
        )
        graph.add_node(language_embed)

        # Multimodal fusion (concat vision + language tokens)
        fusion = FusionBlock(
            name="multimodal_fusion",
            fusion_type="concat",
            input_modalities=["vision", "language"],
            output_dim=4096,
        )
        graph.add_node(fusion)

        # LLM backbone
        llm_backbone = NeuralBlock(
            name="llm_backbone",
            model_ref=cfg.vlm_backbone,
            precision_policy=PrecisionPolicy(
                preferred=cfg.llm_precision,
                critical_layers=["lm_head"],
            ),
            memory_hint=MemoryHint(
                prefer_sram=False,
                streaming_layout=True,
                cache_priority=50,
            ),
        )
        graph.add_node(llm_backbone)

        # Action token generator (autoregressive)
        action_generator = ControlLogicNode(
            name="action_generator",
            logic_type=ControlLogicType.LOOP,
            config={
                "mode": "autoregressive",
                "num_tokens": cfg.action_dim,
                "vocab_size": cfg.action_vocab_size,
            },
        )
        graph.add_node(action_generator)

        # Action detokenizer
        action_detokenizer = NeuralBlock(
            name="action_detokenizer",
            model_ref="openvla_action_decoder",
            precision_policy=PrecisionPolicy(preferred=cfg.action_precision),
            metadata={
                "bins": cfg.action_bins,
                "action_dim": cfg.action_dim,
            },
        )
        graph.add_node(action_detokenizer)

        # Connect nodes
        graph.add_edge(DataEdge("vision_encoder", "vision_projector", tensor_name="vision_features"))
        graph.add_edge(DataEdge("vision_projector", "multimodal_fusion", tensor_name="vision_tokens"))
        graph.add_edge(DataEdge("language_embedding", "multimodal_fusion", tensor_name="language_tokens"))
        graph.add_edge(DataEdge("multimodal_fusion", "llm_backbone", tensor_name="input_embeds"))
        graph.add_edge(DataEdge("llm_backbone", "action_generator", tensor_name="logits"))
        graph.add_edge(DataEdge("action_generator", "action_detokenizer", tensor_name="action_tokens"))

        graph.entry_points = ["vision_encoder", "language_embedding"]
        graph.exit_points = ["action_detokenizer"]

        return graph

    def _create_control_loop(self) -> ControlLoop:
        """Create control loop configuration."""
        cfg = self.config
        return ControlLoop(
            frequency_hz=cfg.control_frequency_hz,
            latency_budget_ms=cfg.latency_budget_ms,
            pipeline_mode=PipelineMode.STANDARD,
            jitter_tolerance_ms=20.0,
            deadline_policy="extend",  # LLM inference may need extra time
        )

    def _create_scheduling_hints(self) -> SchedulingHints:
        """Create scheduling hints."""
        hints = SchedulingHints(
            memory_bound_priority=True,
            kv_reuse_threshold=0.9,  # High reuse for LLM KV cache
            kernel_fusion_allowed=True,
            enable_cuda_graphs=False,  # Autoregressive is dynamic
            tensorrt_enabled=True,
        )

        hints.set_memory_hint("llm_backbone", MemoryHint(
            prefer_sram=False,
            streaming_layout=True,
            cache_priority=80,
        ))

        return hints

    def to_runtime_config(self) -> dict:
        """Convert to robot_runtime configuration dict."""
        cfg = self.config
        return {
            "deadline_ms": cfg.latency_budget_ms,
            "timeout_policy": "EXTEND",
            "backend": "eager",  # Autoregressive needs dynamic control
            "device": "cuda:0",
            "preallocate": True,
            "warmup_iterations": 3,
        }


# =============================================================================
# Policy Registry
# =============================================================================


def get_policy_mapping(policy_name: str):
    """
    Get policy mapping by name.

    Args:
        policy_name: Name of the policy ("act", "diffusion", "openvla", etc.)

    Returns:
        Appropriate mapping instance.
    """
    from robot_ir.integration.pi_mapping import Pi0Mapping, Pi05Mapping

    mappings = {
        "pi0": Pi0Mapping,
        "pi0.5": Pi05Mapping,
        "pi05": Pi05Mapping,
        "act": ACTMapping,
        "diffusion": DiffusionPolicyMapping,
        "diffusion_policy": DiffusionPolicyMapping,
        "openvla": OpenVLAMapping,
        "open_vla": OpenVLAMapping,
    }

    policy_key = policy_name.lower().replace("-", "").replace("_", "")

    # Handle diffusion_policy -> diffusionpolicy
    if policy_key == "diffusionpolicy":
        policy_key = "diffusion"

    if policy_key not in mappings:
        available = ["pi0", "pi0.5", "act", "diffusion", "openvla"]
        raise ValueError(f"Unknown policy: {policy_name}. Available: {available}")

    return mappings[policy_key]()


def list_supported_policies() -> list[str]:
    """List all supported policy architectures."""
    return [
        "pi0",
        "pi0.5",
        "act",
        "diffusion_policy",
        "openvla",
    ]
