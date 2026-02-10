"""Base classes for IR lowering passes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from robot_ir.core.module import RobotModule
    from robot_ir.core.config import DeploymentConfig


@dataclass
class LoweringContext:
    """
    Context passed through lowering passes.

    Attributes:
        target: Target deployment configuration.
        optimizations: List of applied optimizations.
        metadata: Additional metadata from passes.
        errors: Accumulated errors during lowering.
        warnings: Accumulated warnings during lowering.
    """

    target: DeploymentConfig | None = None
    optimizations: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_optimization(self, name: str) -> None:
        """Record an applied optimization."""
        self.optimizations.append(name)

    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.errors.append(error)

    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)

    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return len(self.errors) > 0


class LoweringPass(ABC):
    """
    Base class for IR lowering/optimization passes.

    Passes transform the Robot IR to optimize for specific targets
    or prepare for hardware lowering.
    """

    name: str = "base_pass"

    @abstractmethod
    def run(self, module: RobotModule, ctx: LoweringContext) -> RobotModule:
        """
        Execute the lowering pass.

        Args:
            module: Input Robot IR module.
            ctx: Lowering context.

        Returns:
            Transformed module.
        """
        pass

    def should_run(self, module: RobotModule, ctx: LoweringContext) -> bool:
        """
        Check if this pass should run.

        Override for conditional execution.
        """
        return True


class PassPipeline:
    """
    Pipeline of lowering passes.

    Executes passes in order, passing context between them.
    """

    def __init__(self, passes: list[LoweringPass] | None = None) -> None:
        """Initialize pipeline with optional passes."""
        self.passes = passes or []

    def add_pass(self, pass_: LoweringPass) -> None:
        """Add a pass to the pipeline."""
        self.passes.append(pass_)

    def run(self, module: RobotModule, ctx: LoweringContext | None = None) -> RobotModule:
        """
        Run all passes in the pipeline.

        Args:
            module: Input module.
            ctx: Optional context. Created if not provided.

        Returns:
            Transformed module.
        """
        if ctx is None:
            ctx = LoweringContext()

        result = module
        for pass_ in self.passes:
            if pass_.should_run(result, ctx):
                result = pass_.run(result, ctx)
                ctx.add_optimization(pass_.name)

        return result
