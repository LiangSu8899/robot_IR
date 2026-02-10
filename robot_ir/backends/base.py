"""Base backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from robot_ir.core.module import RobotModule


@dataclass
class BackendConfig:
    """
    Backend configuration.

    Attributes:
        device: Target device ("cuda:0", "cpu").
        optimize: Enable optimizations.
        debug: Enable debug mode.
        options: Backend-specific options.
    """

    device: str = "cuda:0"
    optimize: bool = True
    debug: bool = False
    options: dict = field(default_factory=dict)


class Backend(ABC):
    """
    Abstract backend for IR execution.

    Backends translate Robot IR to executable form for specific
    hardware or runtime environments.
    """

    name: str = "base"

    def __init__(self, config: BackendConfig | None = None) -> None:
        """Initialize backend."""
        self.config = config or BackendConfig()

    @abstractmethod
    def compile(self, module: RobotModule) -> Any:
        """
        Compile IR module to backend-specific representation.

        Args:
            module: Robot IR module to compile.

        Returns:
            Backend-specific executable.
        """
        pass

    @abstractmethod
    def execute(self, compiled: Any, inputs: dict) -> dict:
        """
        Execute compiled module.

        Args:
            compiled: Compiled module from `compile()`.
            inputs: Input tensors.

        Returns:
            Output tensors.
        """
        pass

    def supports(self, module: RobotModule) -> bool:
        """
        Check if backend supports the given module.

        Override for backend-specific checks.
        """
        return True
