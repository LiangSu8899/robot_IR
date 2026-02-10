"""Core module definitions."""

from robot_ir.core.module import RobotModule
from robot_ir.core.types import TensorShape, DataType
from robot_ir.core.config import DeploymentConfig, TargetPlatform
from robot_ir.core.serialization import (
    IRSerializer,
    IRDeserializer,
    save_module,
    load_module,
    to_json,
    from_json,
)

__all__ = [
    "RobotModule",
    "TensorShape",
    "DataType",
    "DeploymentConfig",
    "TargetPlatform",
    "IRSerializer",
    "IRDeserializer",
    "save_module",
    "load_module",
    "to_json",
    "from_json",
]
