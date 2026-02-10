#!/usr/bin/env python3
"""Example: Robot IR to robot_runtime integration.

This example demonstrates the complete workflow from
Robot IR module creation to robot_runtime configuration.
"""

import sys
sys.path.insert(0, ".")

from robot_ir.core import RobotModule, to_json, from_json
from robot_ir.integration.pi_mapping import Pi0Mapping, Pi0Config
from robot_ir.integration.runtime_bridge import RuntimeBridge, IRCompiler


def main():
    """Demonstrate IR to runtime integration."""

    print("=" * 60)
    print("Robot IR to robot_runtime Integration Example")
    print("=" * 60)

    # Step 1: Create IR module from Pi0 mapping
    print("\n1. Creating Pi0 IR module...")
    config = Pi0Config(
        image_size=224,
        action_dim=7,
        state_dim=14,
        chunk_size=50,
        num_cameras=2,
        control_frequency_hz=20.0,
        latency_budget_ms=40.0,
    )
    mapping = Pi0Mapping(config=config)
    module = mapping.create_module("my_pi0_policy")

    print(f"   Module: {module.name}")
    print(f"   Sensors: {[s.name for s in module.sensors]}")
    print(f"   States: {[s.name for s in module.states]}")
    print(f"   Compute nodes: {[n.name for n in module.graph.nodes]}")

    # Step 2: Validate for runtime compatibility
    print("\n2. Validating for robot_runtime...")
    bridge = RuntimeBridge(device="cuda:0")
    errors = bridge.validate_for_runtime(module)

    if errors:
        print(f"   Validation errors: {errors}")
    else:
        print("   Validation passed!")

    # Step 3: Extract specifications
    print("\n3. Extracting specifications...")
    obs_spec = bridge.extract_observation_spec(module)
    action_spec = bridge.extract_action_spec(module)

    print(f"   Observation spec: {obs_spec}")
    print(f"   Action spec: {action_spec}")
    print(f"   Adapter type: {bridge.get_adapter_type(module)}")

    # Step 4: Generate runtime configurations
    print("\n4. Generating runtime configurations...")
    runtime_config, adapter_config = bridge.create_runtime_configs(module)

    print("   RuntimeConfig:")
    for key, value in runtime_config.items():
        print(f"      {key}: {value}")

    print("   PI0Config:")
    for key, value in adapter_config.items():
        print(f"      {key}: {value}")

    # Step 5: Compile with optimization passes
    print("\n5. Compiling with optimization passes...")
    compiler = IRCompiler()
    compiler.add_default_passes()
    report = compiler.get_optimization_report(module)

    print(f"   Optimizations: {report['optimizations']}")

    # Step 6: Serialize to JSON
    print("\n6. Serializing to JSON...")
    json_str = to_json(module)
    print(f"   JSON size: {len(json_str)} bytes")

    # Step 7: Deserialize and verify
    print("\n7. Deserializing and verifying...")
    restored = from_json(json_str)
    print(f"   Restored module: {restored.name}")
    print(f"   Sensors match: {len(restored.sensors) == len(module.sensors)}")
    print(f"   States match: {len(restored.states) == len(module.states)}")

    # Step 8: Show usage with robot_runtime (pseudo-code)
    print("\n8. Usage with robot_runtime (pseudo-code):")
    print("""
    # After loading your policy model:
    from robot_runtime import RuntimeConfig, PI0Config, PI0Adapter

    # Create configs from IR
    runtime_cfg = RuntimeConfig(**runtime_config)
    pi0_cfg = PI0Config(**adapter_config)

    # Create adapter
    adapter = PI0Adapter(policy_model, runtime_cfg, pi0_cfg)

    # Warmup
    adapter.warmup(sample_observation)

    # Run inference
    action, state, stats = adapter.step(observation, state)
    """)

    print("\n" + "=" * 60)
    print("Integration example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
