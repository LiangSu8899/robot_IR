"""Core module definitions."""

from robot_ir.core.module import RobotModule
from robot_ir.core.types import TensorShape, DataType
from robot_ir.core.config import DeploymentConfig, TargetPlatform

__all__ = ["RobotModule", "TensorShape", "DataType", "DeploymentConfig", "TargetPlatform"]
