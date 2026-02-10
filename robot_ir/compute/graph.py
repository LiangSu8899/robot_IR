"""Compute graph representation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from robot_ir.compute.nodes import ComputeNode


@dataclass
class DataEdge:
    """
    Edge connecting two compute nodes.

    Attributes:
        source: Source node name.
        target: Target node name.
        source_output: Output index from source node.
        target_input: Input index to target node.
        tensor_name: Name of the tensor being transferred.
    """

    source: str
    target: str
    source_output: int = 0
    target_input: int = 0
    tensor_name: str = ""

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "source": self.source,
            "target": self.target,
            "source_output": self.source_output,
            "target_input": self.target_input,
            "tensor_name": self.tensor_name,
        }


@dataclass
class ComputeGraph:
    """
    Directed acyclic graph of compute nodes.

    Represents the full computation flow from sensors to actions.

    Attributes:
        nodes: List of compute nodes.
        edges: List of data edges connecting nodes.
        entry_points: Names of entry point nodes (typically sensors).
        exit_points: Names of exit point nodes (typically action heads).

    Example:
        >>> graph = ComputeGraph(
        ...     nodes=[vision_encoder, policy_decoder, action_head],
        ...     edges=[
        ...         DataEdge("vision_encoder", "policy_decoder"),
        ...         DataEdge("policy_decoder", "action_head"),
        ...     ],
        ... )
    """

    nodes: list[ComputeNode] = field(default_factory=list)
    edges: list[DataEdge] = field(default_factory=list)
    entry_points: list[str] = field(default_factory=list)
    exit_points: list[str] = field(default_factory=list)

    def add_node(self, node: ComputeNode) -> None:
        """Add a compute node to the graph."""
        if self.get_node(node.name) is not None:
            raise ValueError(f"Node with name '{node.name}' already exists")
        self.nodes.append(node)

    def add_edge(self, edge: DataEdge) -> None:
        """Add an edge to the graph."""
        # Validate source and target exist
        if self.get_node(edge.source) is None:
            raise ValueError(f"Source node '{edge.source}' not found")
        if self.get_node(edge.target) is None:
            raise ValueError(f"Target node '{edge.target}' not found")
        self.edges.append(edge)

    def get_node(self, name: str) -> ComputeNode | None:
        """Get node by name."""
        for node in self.nodes:
            if node.name == name:
                return node
        return None

    def get_predecessors(self, node_name: str) -> list[str]:
        """Get predecessor node names."""
        return [edge.source for edge in self.edges if edge.target == node_name]

    def get_successors(self, node_name: str) -> list[str]:
        """Get successor node names."""
        return [edge.target for edge in self.edges if edge.source == node_name]

    def topological_sort(self) -> list[str]:
        """
        Return nodes in topological order.

        Returns:
            List of node names in execution order.
        """
        in_degree = {node.name: 0 for node in self.nodes}
        for edge in self.edges:
            in_degree[edge.target] += 1

        queue = [name for name, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            node_name = queue.pop(0)
            result.append(node_name)
            for successor in self.get_successors(node_name):
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    queue.append(successor)

        if len(result) != len(self.nodes):
            raise ValueError("Graph contains a cycle")

        return result

    def validate(self) -> list[str]:
        """
        Validate graph structure.

        Returns:
            List of validation error messages.
        """
        errors = []

        # Check for orphan nodes
        connected = set()
        for edge in self.edges:
            connected.add(edge.source)
            connected.add(edge.target)

        if len(self.nodes) > 1:
            for node in self.nodes:
                if node.name not in connected and node.name not in self.entry_points:
                    errors.append(f"Node '{node.name}' is not connected")

        # Check for cycles
        try:
            self.topological_sort()
        except ValueError as e:
            errors.append(str(e))

        return errors

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "entry_points": self.entry_points,
            "exit_points": self.exit_points,
        }
