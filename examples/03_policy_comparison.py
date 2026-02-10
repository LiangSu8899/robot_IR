#!/usr/bin/env python3
"""Example: Comparing different policy architectures in Robot IR.

This example demonstrates how different robot learning policies
(ACT, Diffusion Policy, OpenVLA, Pi0) map to Robot IR.
"""

import sys
sys.path.insert(0, ".")

from robot_ir.integration import (
    get_policy_mapping,
    list_supported_policies,
    ACTMapping,
    ACTConfig,
    DiffusionPolicyMapping,
    DiffusionPolicyConfig,
    OpenVLAMapping,
    OpenVLAConfig,
    Pi0Mapping,
)


def print_module_summary(module, policy_name: str) -> None:
    """Print a summary of a Robot IR module."""
    print(f"\n{'=' * 60}")
    print(f"  {policy_name.upper()}")
    print(f"{'=' * 60}")

    print(f"\n  Module: {module.name}")

    # Sensors
    print(f"\n  Sensors ({len(module.sensors)}):")
    for sensor in module.sensors:
        print(f"    - {sensor.name}: {sensor.modality.name}, shape={sensor.shape.dims}")

    # States
    print(f"\n  State Buffers ({len(module.states)}):")
    for state in module.states:
        print(f"    - {state.name}: {state.shape.dims}, {state.reuse_strategy.name}")

    # Compute graph
    if module.graph:
        print(f"\n  Compute Graph:")
        print(f"    Nodes: {[n.name for n in module.graph.nodes]}")
        print(f"    Entry: {module.graph.entry_points}")
        print(f"    Exit: {module.graph.exit_points}")

    # Control loop
    if module.control:
        print(f"\n  Control Loop:")
        print(f"    Frequency: {module.control.frequency_hz} Hz")
        print(f"    Latency budget: {module.control.latency_budget_ms} ms")
        print(f"    Pipeline mode: {module.control.pipeline_mode.name}")

    # Scheduling
    if module.schedule:
        print(f"\n  Scheduling Hints:")
        print(f"    TensorRT: {module.schedule.tensorrt_enabled}")
        print(f"    CUDA Graphs: {module.schedule.enable_cuda_graphs}")


def compare_policies():
    """Compare different policy architectures."""
    print("=" * 60)
    print("  Robot IR Policy Comparison")
    print("=" * 60)

    print(f"\nSupported policies: {list_supported_policies()}")

    # 1. ACT (Action Chunking with Transformers)
    act_config = ACTConfig(
        chunk_size=100,
        action_dim=7,
        use_temporal_ensemble=True,
        control_frequency_hz=50.0,
    )
    act_mapping = ACTMapping(config=act_config)
    act_module = act_mapping.create_module("act_aloha")
    print_module_summary(act_module, "ACT (Action Chunking with Transformers)")

    # 2. Diffusion Policy
    diff_config = DiffusionPolicyConfig(
        action_dim=2,
        num_diffusion_steps=100,
        num_inference_steps=10,  # DDIM for fast inference
        use_transformer=False,  # U-Net based
    )
    diff_mapping = DiffusionPolicyMapping(config=diff_config)
    diff_module = diff_mapping.create_module("diffusion_pusht")
    print_module_summary(diff_module, "Diffusion Policy")

    # 3. OpenVLA
    vla_config = OpenVLAConfig(
        action_dim=7,
        max_language_tokens=512,
        control_frequency_hz=5.0,  # Slower due to LLM
    )
    vla_mapping = OpenVLAMapping(config=vla_config)
    vla_module = vla_mapping.create_module("openvla_7b")
    print_module_summary(vla_module, "OpenVLA")

    # 4. Pi0 (for comparison)
    pi0_mapping = Pi0Mapping()
    pi0_module = pi0_mapping.create_module("pi0_policy")
    print_module_summary(pi0_module, "Pi0")


def policy_characteristics():
    """Summarize key characteristics of each policy."""
    print("\n" + "=" * 60)
    print("  Policy Characteristics Summary")
    print("=" * 60)

    characteristics = {
        "ACT": {
            "Architecture": "Transformer Encoder-Decoder + CVAE",
            "Action Generation": "Chunk prediction (100 steps)",
            "Typical Frequency": "50 Hz",
            "Use Case": "Bimanual manipulation (ALOHA)",
            "Key Feature": "Temporal ensemble for smoothing",
        },
        "Diffusion Policy": {
            "Architecture": "U-Net / Transformer Denoiser",
            "Action Generation": "Iterative denoising (10-100 steps)",
            "Typical Frequency": "10 Hz",
            "Use Case": "General manipulation, precise control",
            "Key Feature": "Multi-modal action distributions",
        },
        "OpenVLA": {
            "Architecture": "VLM (Vision-Language Model) + Action Tokenizer",
            "Action Generation": "Autoregressive token generation",
            "Typical Frequency": "5 Hz",
            "Use Case": "Language-conditioned manipulation",
            "Key Feature": "Natural language instructions",
        },
        "Pi0": {
            "Architecture": "SigLIP + Action Expert + Flow Matching",
            "Action Generation": "Chunk prediction with flow matching",
            "Typical Frequency": "20 Hz",
            "Use Case": "General-purpose robot control",
            "Key Feature": "Action chunking with temporal caching",
        },
    }

    for policy, chars in characteristics.items():
        print(f"\n  {policy}:")
        for key, value in chars.items():
            print(f"    {key}: {value}")


def using_get_policy_mapping():
    """Demonstrate using the unified get_policy_mapping function."""
    print("\n" + "=" * 60)
    print("  Using get_policy_mapping()")
    print("=" * 60)

    for policy_name in list_supported_policies():
        mapping = get_policy_mapping(policy_name)
        module = mapping.create_module(f"{policy_name}_module")

        # Get runtime config
        runtime_config = mapping.to_runtime_config()

        print(f"\n  {policy_name}:")
        print(f"    Module: {module.name}")
        print(f"    Runtime config: {runtime_config}")


def main():
    """Run all demonstrations."""
    compare_policies()
    policy_characteristics()
    using_get_policy_mapping()

    print("\n" + "=" * 60)
    print("  Policy comparison complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
