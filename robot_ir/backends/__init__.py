"""Backend implementations for IR execution.

Supported backends:
- TensorRT: High-performance inference on NVIDIA GPUs
- Triton: Custom kernel generation
- ROS2: ROS2 executor integration
"""

from robot_ir.backends.base import Backend, BackendConfig

__all__ = ["Backend", "BackendConfig"]
