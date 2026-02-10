"""Unit tests for multi-rate execution infrastructure."""

import sys
sys.path.insert(0, ".")

import pytest

from robot_ir.control import (
    ControlLoop,
    PipelineMode,
    RateSpec,
)
from robot_ir.control.multirate import (
    InterpolationMode,
    ExtrapolationMode,
    SensorSyncMode,
    RateConversionBuffer,
    AsyncSensorConfig,
    PipelineStage,
    MultiRateSchedule,
    MultiRateScheduler,
    create_schedule_from_module,
    create_vla_schedule,
    create_diffusion_schedule,
    create_act_schedule,
)
from robot_ir.core.module import RobotModule
from robot_ir.integration import Pi0Mapping


class TestRateConversionBuffer:
    """Tests for RateConversionBuffer."""

    def test_create_buffer(self):
        """Test creating a rate conversion buffer."""
        buffer = RateConversionBuffer(
            name="test_buffer",
            source_rate_hz=10.0,
            target_rate_hz=20.0,
        )
        assert buffer.name == "test_buffer"
        assert buffer.source_rate_hz == 10.0
        assert buffer.target_rate_hz == 20.0

    def test_upsampling_detection(self):
        """Test upsampling detection."""
        buffer = RateConversionBuffer(
            name="upsample",
            source_rate_hz=10.0,
            target_rate_hz=20.0,
        )
        assert buffer.is_upsampling is True
        assert buffer.is_downsampling is False
        assert buffer.rate_ratio == 2.0

    def test_downsampling_detection(self):
        """Test downsampling detection."""
        buffer = RateConversionBuffer(
            name="downsample",
            source_rate_hz=100.0,
            target_rate_hz=20.0,
        )
        assert buffer.is_upsampling is False
        assert buffer.is_downsampling is True
        assert buffer.rate_ratio == 0.2

    def test_period_calculation(self):
        """Test period calculation."""
        buffer = RateConversionBuffer(
            name="periods",
            source_rate_hz=10.0,
            target_rate_hz=50.0,
        )
        assert buffer.source_period_ms == 100.0
        assert buffer.target_period_ms == 20.0

    def test_invalid_rates(self):
        """Test validation of invalid rates."""
        with pytest.raises(ValueError):
            RateConversionBuffer(name="bad", source_rate_hz=0, target_rate_hz=10)

        with pytest.raises(ValueError):
            RateConversionBuffer(name="bad", source_rate_hz=10, target_rate_hz=-1)

    def test_to_dict(self):
        """Test serialization."""
        buffer = RateConversionBuffer(
            name="serialize_test",
            source_rate_hz=10.0,
            target_rate_hz=20.0,
            interpolation=InterpolationMode.LINEAR,
        )
        d = buffer.to_dict()
        assert d["name"] == "serialize_test"
        assert d["interpolation"] == "LINEAR"
        assert d["is_upsampling"] is True


class TestAsyncSensorConfig:
    """Tests for AsyncSensorConfig."""

    def test_create_config(self):
        """Test creating sensor config."""
        config = AsyncSensorConfig(
            name="camera",
            rate_hz=30.0,
            sync_mode=SensorSyncMode.LATEST,
        )
        assert config.name == "camera"
        assert config.rate_hz == 30.0
        assert config.sync_mode == SensorSyncMode.LATEST

    def test_to_dict(self):
        """Test serialization."""
        config = AsyncSensorConfig(
            name="imu",
            rate_hz=100.0,
            queue_size=4,
        )
        d = config.to_dict()
        assert d["name"] == "imu"
        assert d["rate_hz"] == 100.0
        assert d["queue_size"] == 4


class TestPipelineStage:
    """Tests for PipelineStage."""

    def test_create_stage(self):
        """Test creating a pipeline stage."""
        stage = PipelineStage(
            name="vision",
            components=["encoder", "projector"],
            rate_hz=10.0,
            priority=1,
        )
        assert stage.name == "vision"
        assert len(stage.components) == 2
        assert stage.rate_hz == 10.0

    def test_dependencies(self):
        """Test stage dependencies."""
        stage = PipelineStage(
            name="policy",
            dependencies=["vision", "proprio"],
        )
        assert "vision" in stage.dependencies
        assert len(stage.dependencies) == 2


class TestMultiRateSchedule:
    """Tests for MultiRateSchedule."""

    def test_create_schedule(self):
        """Test creating a schedule."""
        schedule = MultiRateSchedule()
        assert len(schedule.stages) == 0
        assert len(schedule.rate_buffers) == 0

    def test_add_stages(self):
        """Test adding stages."""
        schedule = MultiRateSchedule()
        schedule.add_stage(PipelineStage(name="vision", rate_hz=10.0))
        schedule.add_stage(PipelineStage(name="policy", rate_hz=20.0))

        assert len(schedule.stages) == 2
        assert schedule.get_stage("vision") is not None
        assert schedule.get_stage("nonexistent") is None

    def test_add_rate_buffers(self):
        """Test adding rate buffers."""
        schedule = MultiRateSchedule()
        schedule.add_rate_buffer(RateConversionBuffer(
            name="v2p",
            source_rate_hz=10.0,
            target_rate_hz=20.0,
        ))

        assert len(schedule.rate_buffers) == 1
        assert schedule.get_rate_buffer("v2p") is not None

    def test_execution_order(self):
        """Test topological sort of stages."""
        schedule = MultiRateSchedule()
        schedule.add_stage(PipelineStage(name="control", dependencies=["policy"]))
        schedule.add_stage(PipelineStage(name="vision"))
        schedule.add_stage(PipelineStage(name="policy", dependencies=["vision"]))

        order = schedule.get_execution_order()
        assert order.index("vision") < order.index("policy")
        assert order.index("policy") < order.index("control")

    def test_circular_dependency_detection(self):
        """Test circular dependency detection."""
        schedule = MultiRateSchedule()
        schedule.add_stage(PipelineStage(name="a", dependencies=["c"]))
        schedule.add_stage(PipelineStage(name="b", dependencies=["a"]))
        schedule.add_stage(PipelineStage(name="c", dependencies=["b"]))

        errors = schedule.validate()
        assert any("Circular" in e for e in errors)

    def test_missing_dependency_detection(self):
        """Test missing dependency detection."""
        schedule = MultiRateSchedule()
        schedule.add_stage(PipelineStage(name="policy", dependencies=["nonexistent"]))

        errors = schedule.validate()
        assert any("unknown stage" in e for e in errors)

    def test_hyperperiod_calculation(self):
        """Test hyperperiod calculation."""
        schedule = MultiRateSchedule()
        schedule.add_stage(PipelineStage(name="fast", rate_hz=100.0))  # 10ms
        schedule.add_stage(PipelineStage(name="slow", rate_hz=10.0))   # 100ms

        hyperperiod = schedule.compute_hyperperiod()
        assert hyperperiod >= 100.0  # At least 100ms


class TestMultiRateScheduler:
    """Tests for MultiRateScheduler."""

    def test_create_scheduler(self):
        """Test creating a scheduler."""
        schedule = create_vla_schedule()
        scheduler = MultiRateScheduler(schedule)
        assert scheduler is not None

    def test_get_stages_for_tick(self):
        """Test getting stages for a tick."""
        schedule = MultiRateSchedule(base_rate_hz=100.0)
        schedule.add_stage(PipelineStage(name="fast", rate_hz=100.0))
        schedule.add_stage(PipelineStage(name="slow", rate_hz=10.0))

        scheduler = MultiRateScheduler(schedule)

        # At t=0, both should run
        stages = scheduler.get_stages_for_tick(0.0)
        assert "fast" in stages
        assert "slow" in stages

        # At t=10ms, only fast should run (slow runs at 100ms intervals)
        stages = scheduler.get_stages_for_tick(10.0)
        assert "fast" in stages

    def test_execution_plan(self):
        """Test execution plan creation."""
        schedule = create_vla_schedule(control_hz=100.0)
        scheduler = MultiRateScheduler(schedule)

        plan = scheduler.create_execution_plan(100.0)  # 100ms
        assert len(plan) > 0
        assert all("tick_ms" in event for event in plan)
        assert all("stages" in event for event in plan)

    def test_cpu_utilization(self):
        """Test CPU utilization estimation."""
        schedule = create_vla_schedule()
        scheduler = MultiRateScheduler(schedule)

        util = scheduler.estimate_cpu_utilization()
        assert "vision" in util
        assert "policy" in util
        assert "control" in util
        assert all(0 <= u <= 100 for u in util.values())

    def test_schedulability_check(self):
        """Test schedulability analysis."""
        schedule = create_vla_schedule()
        scheduler = MultiRateScheduler(schedule)

        schedulable, warnings = scheduler.check_schedulability()
        assert isinstance(schedulable, bool)
        assert isinstance(warnings, list)


class TestSchedulePresets:
    """Tests for preset schedule creation functions."""

    def test_vla_schedule(self):
        """Test VLA schedule preset."""
        schedule = create_vla_schedule(vision_hz=10.0, policy_hz=20.0, control_hz=100.0)

        assert len(schedule.stages) == 3
        assert schedule.get_stage("vision") is not None
        assert schedule.get_stage("policy") is not None
        assert schedule.get_stage("control") is not None

        # Check dependencies
        policy = schedule.get_stage("policy")
        assert "vision" in policy.dependencies

        control = schedule.get_stage("control")
        assert "policy" in control.dependencies

        # Check rate buffers
        assert len(schedule.rate_buffers) == 2

    def test_diffusion_schedule(self):
        """Test diffusion policy schedule preset."""
        schedule = create_diffusion_schedule()

        assert len(schedule.stages) == 3
        assert schedule.get_stage("vision") is not None
        assert schedule.get_stage("diffusion") is not None
        assert schedule.get_stage("action") is not None

        # Diffusion stage should have high latency allowance
        diffusion = schedule.get_stage("diffusion")
        assert diffusion.max_latency_ms >= 50.0

    def test_act_schedule(self):
        """Test ACT schedule preset."""
        schedule = create_act_schedule(vision_hz=30.0, policy_hz=50.0)

        assert len(schedule.stages) == 4
        assert schedule.get_stage("vision") is not None
        assert schedule.get_stage("encoder") is not None
        assert schedule.get_stage("decoder") is not None
        assert schedule.get_stage("ensemble") is not None


class TestScheduleFromModule:
    """Tests for creating schedules from IR modules."""

    def test_schedule_from_pi0(self):
        """Test creating schedule from Pi0 module."""
        mapping = Pi0Mapping()
        module = mapping.create_module("pi0_test")

        schedule = create_schedule_from_module(module)

        assert schedule is not None
        assert len(schedule.stages) > 0
        assert len(schedule.sensor_configs) > 0

    def test_schedule_from_module_preserves_rates(self):
        """Test that rates are preserved from module control loop."""
        module = RobotModule(name="test")
        module.control = ControlLoop(
            frequency_hz=20.0,
            multi_rate=[
                RateSpec("vision", 10.0, priority=1),
                RateSpec("policy", 20.0, priority=2),
            ],
        )

        schedule = create_schedule_from_module(module)

        vision_stage = schedule.get_stage("vision")
        assert vision_stage is not None
        assert vision_stage.rate_hz == 10.0

        policy_stage = schedule.get_stage("policy")
        assert policy_stage is not None
        assert policy_stage.rate_hz == 20.0


class TestInterpolationModes:
    """Tests for interpolation mode configuration."""

    def test_all_interpolation_modes(self):
        """Test all interpolation modes can be used."""
        for mode in InterpolationMode:
            buffer = RateConversionBuffer(
                name=f"test_{mode.name}",
                source_rate_hz=10.0,
                target_rate_hz=20.0,
                interpolation=mode,
            )
            assert buffer.interpolation == mode

    def test_all_extrapolation_modes(self):
        """Test all extrapolation modes can be used."""
        for mode in ExtrapolationMode:
            buffer = RateConversionBuffer(
                name=f"test_{mode.name}",
                source_rate_hz=10.0,
                target_rate_hz=20.0,
                extrapolation=mode,
            )
            assert buffer.extrapolation == mode


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
