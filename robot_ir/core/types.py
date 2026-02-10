"""Basic type definitions for Robot IR."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Sequence


class DataType(Enum):
    """Supported data types."""

    FLOAT32 = auto()
    FLOAT16 = auto()
    BFLOAT16 = auto()
    FLOAT8_E4M3 = auto()
    FLOAT8_E5M2 = auto()
    INT8 = auto()
    INT4 = auto()
    UINT8 = auto()
    INT32 = auto()
    INT64 = auto()
    BOOL = auto()


@dataclass(frozen=True)
class TensorShape:
    """
    Tensor shape representation with optional dynamic dimensions.

    Attributes:
        dims: Tuple of dimension sizes. -1 indicates dynamic dimension.
        dtype: Data type of the tensor.
        layout: Memory layout (e.g., "NCHW", "NHWC").
    """

    dims: tuple[int, ...]
    dtype: DataType = DataType.FLOAT32
    layout: str = "NCHW"

    @property
    def is_dynamic(self) -> bool:
        """Whether shape has dynamic dimensions."""
        return -1 in self.dims

    @property
    def rank(self) -> int:
        """Number of dimensions."""
        return len(self.dims)

    @property
    def num_elements(self) -> int | None:
        """Total number of elements, None if dynamic."""
        if self.is_dynamic:
            return None
        result = 1
        for d in self.dims:
            result *= d
        return result

    def with_batch(self, batch_size: int) -> TensorShape:
        """Return new shape with specified batch dimension."""
        return TensorShape(
            dims=(batch_size,) + self.dims[1:],
            dtype=self.dtype,
            layout=self.layout,
        )

    @classmethod
    def from_list(
        cls,
        dims: Sequence[int],
        dtype: DataType = DataType.FLOAT32,
        layout: str = "NCHW",
    ) -> TensorShape:
        """Create TensorShape from a list of dimensions."""
        return cls(dims=tuple(dims), dtype=dtype, layout=layout)


@dataclass
class TensorSpec:
    """
    Full tensor specification including shape and metadata.

    Attributes:
        name: Tensor identifier.
        shape: Tensor shape.
        is_optional: Whether this tensor is optional.
        description: Human-readable description.
    """

    name: str
    shape: TensorShape
    is_optional: bool = False
    description: str = ""
