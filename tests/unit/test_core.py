"""Tests for core IR types and module."""

import pytest

from robot_ir.core.types import TensorShape, DataType
from robot_ir.core.module import RobotModule
from robot_ir.core.config import DeploymentConfig, TargetPlatform


class TestTensorShape:
    """Tests for TensorShape."""

    def test_create_shape(self):
        shape = TensorShape((3, 224, 224), DataType.FLOAT16)
        assert shape.dims == (3, 224, 224)
        assert shape.dtype == DataType.FLOAT16
        assert shape.rank == 3

    def test_dynamic_shape(self):
        shape = TensorShape((-1, 3, 224, 224))
        assert shape.is_dynamic
        assert shape.num_elements is None

    def test_static_shape_elements(self):
        shape = TensorShape((2, 3, 4))
        assert not shape.is_dynamic
        assert shape.num_elements == 24

    def test_with_batch(self):
        shape = TensorShape((3, 224, 224))
        batched = shape.with_batch(16)
        assert batched.dims == (16, 224, 224)


class TestRobotModule:
    """Tests for RobotModule."""

    def test_create_module(self):
        module = RobotModule(name="test_policy")
        assert module.name == "test_policy"
        assert module.version == "0.1.0"

    def test_module_requires_name(self):
        with pytest.raises(ValueError):
            RobotModule(name="")

    def test_validate_empty_module(self):
        module = RobotModule(name="test")
        errors = module.validate()
        assert len(errors) > 0
        assert "sensor" in errors[0].lower()


class TestDeploymentConfig:
    """Tests for DeploymentConfig."""

    def test_default_config(self):
        config = DeploymentConfig()
        assert config.target == TargetPlatform.THOR
        assert config.realtime is True

    def test_is_jetson(self):
        thor_config = DeploymentConfig(target=TargetPlatform.THOR)
        orin_config = DeploymentConfig(target=TargetPlatform.ORIN)
        x86_config = DeploymentConfig(target=TargetPlatform.X86)

        assert thor_config.is_jetson()
        assert orin_config.is_jetson()
        assert not x86_config.is_jetson()
