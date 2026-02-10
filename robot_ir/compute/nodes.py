"""Compute node definitions for the IR graph."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from robot_ir.core.types import TensorShape, TensorSpec
    from robot_ir.compute.precision import PrecisionPolicy
    from robot_ir.scheduling.hints import MemoryHint


class NodeType(Enum):
    """Type of compute node."""

    NEURAL = auto()  # Neural network block
    FUSION = auto()  # Multi-modal fusion
    CONTROL = auto()  # Control logic
    WORLD_MODEL = auto()  # World model
    PREPROCESSOR = auto()  # Preprocessing
    POSTPROCESSOR = auto()  # Postprocessing


class ControlLogicType(Enum):
    """Control logic node types."""

    SELECTOR = auto()  # Select between inputs
    GATING = auto()  # Gating mechanism
    SAFETY_FILTER = auto()  # Safety constraint filter
    ROUTER = auto()  # Route to different paths


@dataclass
class ComputeNode(ABC):
    """
    Base class for all compute nodes in the IR graph.

    Attributes:
        name: Unique node identifier.
        inputs: List of input tensor specifications.
        outputs: List of output tensor specifications.
        node_type: Type of compute node.
    """

    name: str
    inputs: list[TensorSpec] = field(default_factory=list)
    outputs: list[TensorSpec] = field(default_factory=list)
    node_type: NodeType = NodeType.NEURAL

    @abstractmethod
    def to_dict(self) -> dict:
        """Serialize node to dictionary."""
        pass


@dataclass
class NeuralBlock(ComputeNode):
    """
    Neural network compute block.

    Represents a neural network module (encoder, decoder, head, etc.).

    Attributes:
        model_ref: Reference to the model (path or registry key).
        precision_policy: Precision/quantization policy.
        memory_hint: Memory placement and optimization hints.
        is_trainable: Whether weights can be updated.
        checkpoint_path: Path to model checkpoint.

    Example:
        >>> vision_encoder = NeuralBlock(
        ...     name="vision_encoder",
        ...     model_ref="siglip_vit_so400m",
        ...     precision_policy=PrecisionPolicy(preferred=Precision.FP8_E4M3),
        ... )
    """

    model_ref: str = ""
    precision_policy: PrecisionPolicy | None = None
    memory_hint: MemoryHint | None = None
    is_trainable: bool = False
    checkpoint_path: str | None = None

    def __post_init__(self) -> None:
        """Set node type."""
        self.node_type = NodeType.NEURAL

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "type": "NeuralBlock",
            "name": self.name,
            "model_ref": self.model_ref,
            "precision_policy": self.precision_policy.to_dict() if self.precision_policy else None,
            "is_trainable": self.is_trainable,
            "checkpoint_path": self.checkpoint_path,
        }


@dataclass
class FusionBlock(ComputeNode):
    """
    Multi-modal fusion block.

    Used for combining features from multiple modalities or
    implementing expert routing in mixture-of-experts models.

    Attributes:
        fusion_type: Type of fusion ("concat", "attention", "gating", "moe").
        input_modalities: List of input modality names.
        output_dim: Output feature dimension.
        num_experts: Number of experts (for MoE).
        top_k: Top-k experts to route to (for MoE).
    """

    fusion_type: str = "concat"
    input_modalities: list[str] = field(default_factory=list)
    output_dim: int = 512
    num_experts: int | None = None
    top_k: int | None = None

    def __post_init__(self) -> None:
        """Set node type."""
        self.node_type = NodeType.FUSION

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "type": "FusionBlock",
            "name": self.name,
            "fusion_type": self.fusion_type,
            "input_modalities": self.input_modalities,
            "output_dim": self.output_dim,
            "num_experts": self.num_experts,
            "top_k": self.top_k,
        }


@dataclass
class ControlLogicNode(ComputeNode):
    """
    Control logic node for non-neural operations.

    Used for implementing safety filters, mode selection,
    and other control flow logic.

    Attributes:
        logic_type: Type of control logic.
        config: Configuration dictionary.
    """

    logic_type: ControlLogicType = ControlLogicType.SELECTOR
    config: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Set node type."""
        self.node_type = NodeType.CONTROL

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "type": "ControlLogicNode",
            "name": self.name,
            "logic_type": self.logic_type.name,
            "config": self.config,
        }


@dataclass
class WorldModelBlock(ComputeNode):
    """
    World model block for latent dynamics prediction.

    Used in World Action (WA) models for latent rollout
    and action planning.

    Attributes:
        latent_dim: Latent state dimension.
        rollout_steps: Number of rollout steps.
        compression_ratio: Observation compression ratio.
        action_conditioned: Whether dynamics is action-conditioned.
        stochastic: Whether to use stochastic dynamics.
    """

    latent_dim: int = 512
    rollout_steps: int = 1
    compression_ratio: float = 1.0
    action_conditioned: bool = True
    stochastic: bool = False

    def __post_init__(self) -> None:
        """Set node type."""
        self.node_type = NodeType.WORLD_MODEL

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "type": "WorldModelBlock",
            "name": self.name,
            "latent_dim": self.latent_dim,
            "rollout_steps": self.rollout_steps,
            "compression_ratio": self.compression_ratio,
            "action_conditioned": self.action_conditioned,
            "stochastic": self.stochastic,
        }
