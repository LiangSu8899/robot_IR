#!/usr/bin/env python3
"""Example: Multi-rate execution scheduling in Robot IR.

This example demonstrates how to configure and analyze multi-rate
execution schedules for robot policies with different component rates.

Typical robot policy execution involves:
- Vision processing: 10-30 Hz (heavy computation)
- Policy inference: 10-50 Hz (action generation)
- Control output: 50-200 Hz (low-level servo control)
"""

import sys
sys.path.insert(0, ".")

from robot_ir.control import (
    ControlLoop,
    PipelineMode,
    RateSpec,
    MultiRateSchedule,
    MultiRateScheduler,
    RateConversionBuffer,
    PipelineStage,
    AsyncSensorConfig,
    InterpolationMode,
    SensorSyncMode,
    create_vla_schedule,
    create_diffusion_schedule,
    create_act_schedule,
    create_schedule_from_module,
)
from robot_ir.integration import Pi0Mapping


def print_schedule_analysis(schedule: MultiRateSchedule, name: str) -> None:
    """Print analysis of a multi-rate schedule."""
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")

    # Stages
    print(f"\n  Pipeline Stages ({len(schedule.stages)}):")
    for stage in schedule.stages:
        deps = f" <- {stage.dependencies}" if stage.dependencies else ""
        skip = " [skippable]" if stage.can_skip else ""
        print(f"    {stage.name}: {stage.rate_hz} Hz, "
              f"max {stage.max_latency_ms}ms{deps}{skip}")

    # Rate buffers
    print(f"\n  Rate Conversion Buffers ({len(schedule.rate_buffers)}):")
    for buffer in schedule.rate_buffers:
        direction = "↑" if buffer.is_upsampling else "↓"
        print(f"    {buffer.name}: {buffer.source_rate_hz} Hz {direction} "
              f"{buffer.target_rate_hz} Hz ({buffer.interpolation.name})")

    # Sensor configs
    if schedule.sensor_configs:
        print(f"\n  Sensor Configurations ({len(schedule.sensor_configs)}):")
        for config in schedule.sensor_configs:
            print(f"    {config.name}: {config.rate_hz} Hz, {config.sync_mode.name}")

    # Execution order
    print(f"\n  Execution Order: {' -> '.join(schedule.get_execution_order())}")
    print(f"  Hyperperiod: {schedule.compute_hyperperiod():.1f} ms")

    # Validation
    errors = schedule.validate()
    if errors:
        print(f"\n  Validation Errors:")
        for error in errors:
            print(f"    - {error}")
    else:
        print(f"\n  Validation: PASSED")

    # Schedulability analysis
    scheduler = MultiRateScheduler(schedule)
    schedulable, warnings = scheduler.check_schedulability()

    print(f"\n  Schedulability Analysis:")
    print(f"    Schedulable: {schedulable}")

    utilization = scheduler.estimate_cpu_utilization()
    total_util = sum(utilization.values())
    print(f"    Total Utilization: {total_util:.1f}%")
    for stage, util in utilization.items():
        print(f"      {stage}: {util:.1f}%")

    if warnings:
        print(f"\n  Warnings:")
        for warning in warnings:
            print(f"    - {warning}")


def demonstrate_execution_plan():
    """Demonstrate execution plan generation."""
    print("\n" + "=" * 60)
    print("  Execution Plan Demonstration")
    print("=" * 60)

    schedule = create_vla_schedule(vision_hz=10.0, policy_hz=20.0, control_hz=100.0)
    scheduler = MultiRateScheduler(schedule)

    # Generate 50ms plan
    plan = scheduler.create_execution_plan(50.0)

    print("\n  Execution plan for 50ms window:")
    print("  " + "-" * 40)
    for event in plan:
        stages = ", ".join(event["stages"])
        print(f"    t={event['tick_ms']:5.1f}ms: [{stages}]")

    print(f"\n  Total execution events: {len(plan)}")


def demonstrate_custom_schedule():
    """Demonstrate creating a custom schedule."""
    print("\n" + "=" * 60)
    print("  Custom Schedule Example")
    print("=" * 60)

    # Create a custom multi-rate schedule
    schedule = MultiRateSchedule(base_rate_hz=200.0, deadline_enforcement=True)

    # Add stages with explicit configuration
    schedule.add_stage(PipelineStage(
        name="camera_processing",
        components=["debayer", "undistort", "resize"],
        rate_hz=30.0,
        priority=1,
        max_latency_ms=15.0,
        can_skip=True,  # Can skip under load
    ))

    schedule.add_stage(PipelineStage(
        name="vision_encoder",
        components=["siglip_encoder", "vision_projector"],
        rate_hz=10.0,
        priority=2,
        max_latency_ms=50.0,
        dependencies=["camera_processing"],
    ))

    schedule.add_stage(PipelineStage(
        name="policy_inference",
        components=["transformer_backbone", "action_head"],
        rate_hz=20.0,
        priority=3,
        max_latency_ms=30.0,
        dependencies=["vision_encoder"],
    ))

    schedule.add_stage(PipelineStage(
        name="action_interpolation",
        components=["action_smoother", "safety_filter"],
        rate_hz=200.0,
        priority=4,
        max_latency_ms=2.0,
        dependencies=["policy_inference"],
    ))

    # Add rate conversion buffers
    schedule.add_rate_buffer(RateConversionBuffer(
        name="camera_to_vision",
        source_rate_hz=30.0,
        target_rate_hz=10.0,
        interpolation=InterpolationMode.ZERO_ORDER_HOLD,  # Downsample
    ))

    schedule.add_rate_buffer(RateConversionBuffer(
        name="vision_to_policy",
        source_rate_hz=10.0,
        target_rate_hz=20.0,
        interpolation=InterpolationMode.ZERO_ORDER_HOLD,  # Upsample (hold)
    ))

    schedule.add_rate_buffer(RateConversionBuffer(
        name="policy_to_action",
        source_rate_hz=20.0,
        target_rate_hz=200.0,
        interpolation=InterpolationMode.LINEAR,  # Smooth interpolation
        max_staleness_ms=60.0,
    ))

    # Add sensor configurations
    schedule.add_sensor_config(AsyncSensorConfig(
        name="camera_left",
        rate_hz=30.0,
        sync_mode=SensorSyncMode.TIMESTAMP,
        queue_size=2,
    ))

    schedule.add_sensor_config(AsyncSensorConfig(
        name="camera_right",
        rate_hz=30.0,
        sync_mode=SensorSyncMode.TIMESTAMP,
        queue_size=2,
    ))

    schedule.add_sensor_config(AsyncSensorConfig(
        name="joint_state",
        rate_hz=200.0,
        sync_mode=SensorSyncMode.LATEST,
        required=True,
    ))

    print_schedule_analysis(schedule, "Custom High-Rate Control Schedule")


def demonstrate_schedule_from_module():
    """Demonstrate creating schedule from IR module."""
    print("\n" + "=" * 60)
    print("  Schedule from Pi0 Module")
    print("=" * 60)

    # Create Pi0 module
    mapping = Pi0Mapping()
    module = mapping.create_module("pi0_policy")

    # Create schedule from module
    schedule = create_schedule_from_module(module)

    print(f"\n  Module: {module.name}")
    print(f"  Control frequency: {module.control.frequency_hz} Hz")
    print(f"  Latency budget: {module.control.latency_budget_ms} ms")

    if module.control.multi_rate:
        print(f"\n  Multi-rate specs:")
        for spec in module.control.multi_rate:
            print(f"    {spec.name}: {spec.frequency_hz} Hz (priority {spec.priority})")

    print(f"\n  Generated schedule:")
    print(f"    Stages: {[s.name for s in schedule.stages]}")
    print(f"    Sensors: {[c.name for c in schedule.sensor_configs]}")


def compare_policy_schedules():
    """Compare schedules for different policy architectures."""
    print("\n" + "=" * 60)
    print("  Policy Schedule Comparison")
    print("=" * 60)

    schedules = {
        "VLA (Vision-Language-Action)": create_vla_schedule(
            vision_hz=10.0, policy_hz=20.0, control_hz=100.0
        ),
        "Diffusion Policy": create_diffusion_schedule(
            vision_hz=30.0, diffusion_hz=10.0, action_hz=10.0
        ),
        "ACT (Action Chunking)": create_act_schedule(
            vision_hz=30.0, policy_hz=50.0
        ),
    }

    print("\n  Schedule Comparison Summary:")
    print("  " + "-" * 55)
    print(f"  {'Policy':<25} {'Stages':<8} {'Max Rate':<10} {'Hyperperiod':<12}")
    print("  " + "-" * 55)

    for name, schedule in schedules.items():
        max_rate = max(s.rate_hz for s in schedule.stages)
        hyperperiod = schedule.compute_hyperperiod()
        print(f"  {name:<25} {len(schedule.stages):<8} {max_rate:<10.0f} {hyperperiod:<12.1f}")

    # Detailed comparison
    for name, schedule in schedules.items():
        print_schedule_analysis(schedule, name)


def main():
    """Run all demonstrations."""
    print("=" * 60)
    print("  Robot IR Multi-Rate Scheduling Examples")
    print("=" * 60)

    # 1. Compare policy schedules
    compare_policy_schedules()

    # 2. Custom schedule example
    demonstrate_custom_schedule()

    # 3. Schedule from module
    demonstrate_schedule_from_module()

    # 4. Execution plan
    demonstrate_execution_plan()

    print("\n" + "=" * 60)
    print("  Multi-rate scheduling demonstration complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
