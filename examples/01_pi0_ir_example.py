"""Example: Create Robot IR for Pi0 policy.

This example demonstrates how to create a Robot IR module
for the Pi0 policy architecture.
"""

from robot_ir import (
    RobotModule,
    TensorShape,
    DataType,
    SensorNode,
    Modality,
    BufferPolicy,
    BufferType,
    StateBuffer,
    Persistence,
    ReuseStrategy,
    ComputeGraph,
    DataEdge,
    NeuralBlock,
    PrecisionPolicy,
    Precision,
    ControlLoop,
    PipelineMode,
    SchedulingHints,
    DeploymentConfig,
    TargetPlatform,
)
from robot_ir.control.loop import RateSpec


def create_pi0_module() -> RobotModule:
    """Create a Pi0 policy Robot IR module."""

    # Create module
    module = RobotModule(name="pi0_policy", version="0.1.0")

    # === Sensors ===
    # Front camera
    camera_front = SensorNode(
        name="camera_front",
        modality=Modality.VISION,
        shape=TensorShape((3, 224, 224), DataType.FLOAT16),
        rate_hz=30.0,
        buffering=BufferPolicy(type=BufferType.STREAM, reuse_allowed=True),
    )
    module.add_sensor(camera_front)

    # Wrist camera
    camera_wrist = SensorNode(
        name="camera_wrist",
        modality=Modality.VISION,
        shape=TensorShape((3, 224, 224), DataType.FLOAT16),
        rate_hz=30.0,
    )
    module.add_sensor(camera_wrist)

    # Proprioceptive state
    proprio = SensorNode(
        name="proprio",
        modality=Modality.PROPRIO,
        shape=TensorShape((14,), DataType.FLOAT32),
        rate_hz=100.0,
    )
    module.add_sensor(proprio)

    # === State Buffers ===
    # Transformer KV cache
    kv_cache = StateBuffer(
        name="transformer_kv",
        shape=TensorShape((24, 2, 256, 64), DataType.FLOAT16),
        persistence=Persistence.STEP,
        reuse_strategy=ReuseStrategy.KV_CACHE,
        placement="gpu",
    )
    module.add_state(kv_cache)

    # Action chunk buffer
    action_buffer = StateBuffer(
        name="action_chunk",
        shape=TensorShape((50, 7), DataType.FLOAT32),
        persistence=Persistence.STEP,
        reuse_strategy=ReuseStrategy.ACTION_BUFFER,
    )
    module.add_state(action_buffer)

    # === Compute Graph ===
    graph = ComputeGraph()

    # Vision encoder (SigLIP)
    vision_encoder = NeuralBlock(
        name="vision_encoder",
        model_ref="siglip_so400m_patch14_224",
        precision_policy=PrecisionPolicy(
            preferred=Precision.FP8_E4M3,
            critical_layers=["patch_embed", "norm"],
        ),
    )
    graph.add_node(vision_encoder)

    # Policy decoder
    policy_decoder = NeuralBlock(
        name="policy_decoder",
        model_ref="pi0_action_expert",
        precision_policy=PrecisionPolicy(preferred=Precision.FP16),
    )
    graph.add_node(policy_decoder)

    # Action head (flow matching)
    action_head = NeuralBlock(
        name="action_head",
        model_ref="flow_matching_head",
        precision_policy=PrecisionPolicy(preferred=Precision.FP32),
    )
    graph.add_node(action_head)

    # Connect nodes
    graph.add_edge(DataEdge("vision_encoder", "policy_decoder"))
    graph.add_edge(DataEdge("policy_decoder", "action_head"))

    graph.entry_points = ["vision_encoder"]
    graph.exit_points = ["action_head"]

    module.graph = graph

    # === Control Loop ===
    module.control = ControlLoop(
        frequency_hz=20.0,
        latency_budget_ms=40.0,
        pipeline_mode=PipelineMode.SYNC,
        multi_rate=[
            RateSpec("vision", 10.0, priority=1),
            RateSpec("policy", 20.0, priority=2),
        ],
    )

    # === Scheduling Hints ===
    module.schedule = SchedulingHints(
        memory_bound_priority=True,
        kv_reuse_threshold=0.8,
        enable_cuda_graphs=True,
        tensorrt_enabled=True,
    )

    # === Deployment Config ===
    module.deploy = DeploymentConfig(
        target=TargetPlatform.THOR,
        realtime=True,
        ros_bridge=False,
    )

    return module


if __name__ == "__main__":
    # Create the module
    module = create_pi0_module()

    # Validate
    errors = module.validate()
    if errors:
        print("Validation errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("Module validated successfully!")

    # Print summary
    print(f"\nModule: {module.name}")
    print(f"Sensors: {len(module.sensors)}")
    print(f"State buffers: {len(module.states)}")
    print(f"Compute nodes: {len(module.graph.nodes) if module.graph else 0}")
    print(f"Control frequency: {module.control.frequency_hz}Hz")

    # Serialize to dict
    import json
    print("\nSerialized IR:")
    print(json.dumps(module.to_dict(), indent=2, default=str))
