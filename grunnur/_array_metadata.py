from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from .dtypes import _normalize_type

if TYPE_CHECKING:  # pragma: no cover
    import numpy
    from numpy.typing import DTypeLike, NDArray

    from ._array import Array


class AsArrayMetadata(ABC):
    """An abstract class for any object allowing conversion to :py:class:`ArrayMetadata`."""

    @abstractmethod
    def as_array_metadata(self) -> ArrayMetadata:
        """Returns array metadata representing this object."""
        ...


class ArrayMetadata(AsArrayMetadata):
    """
    A helper object for array-like classes that handles shape/strides/buffer size checks
    without actual data attached to it.
    """

    shape: tuple[int, ...]
    """Array shape."""

    dtype: numpy.dtype[Any]
    """Array item data type."""

    strides: tuple[int, ...]
    """Array strides."""

    buffer_size: int
    """The size of the buffer this array resides in."""

    span: int
    """The minimum size of the buffer that fits all the elements described by this metadata."""

    min_offset: int
    """The minimum offset of an array element described by this metadata."""

    first_element_offset: int
    """The offset of the first element (that is, the one with the all indices equal to 0)."""

    is_contiguous: bool
    """If ``True``, means that array's data forms a continuous chunk of memory."""

    @classmethod
    def from_arraylike(cls, array_like: AsArrayMetadata | NDArray[Any]) -> ArrayMetadata:
        if isinstance(array_like, AsArrayMetadata):
            return array_like.as_array_metadata()

        return cls(shape=array_like.shape, dtype=array_like.dtype, strides=array_like.strides)

    def __init__(
        self,
        shape: Iterable[int] | int,
        dtype: DTypeLike,
        *,
        strides: Iterable[int] | None = None,
        first_element_offset: int | None = None,
        buffer_size: int | None = None,
    ):
        shape = tuple(shape) if isinstance(shape, Iterable) else (shape,)

        if len(shape) == 0:
            raise ValueError("Array shape cannot be an empty sequence")

        dtype = _normalize_type(dtype)

        default_strides = _get_strides(shape, dtype.itemsize)
        strides = default_strides if strides is None else tuple(strides)

        self.shape = shape
        self.dtype = dtype
        self.strides = strides

        # Note that these are minimum and maximum offsets
        # when the first element offset is 0.
        min_offset, max_offset = _get_range(shape, dtype.itemsize, strides)
        self.span = max_offset - min_offset

        if first_element_offset is None:
            first_element_offset = -min_offset
        elif first_element_offset < -min_offset:
            raise ValueError(f"First element offset is smaller than the minimum {-min_offset}")
        self.first_element_offset = first_element_offset
        self.min_offset = first_element_offset + min_offset

        min_buffer_size = self.first_element_offset + max_offset
        if buffer_size is None:
            buffer_size = min_buffer_size
        elif buffer_size < min_buffer_size:
            raise ValueError(f"Buffer size is smaller than the minimum {min_buffer_size}")
        self.buffer_size = buffer_size

        # Technically, an array with non-default (e.g., overlapping) strides
        # can be contioguous, but that's too hard to determine.
        self.is_contiguous = strides == default_strides

        self._default_strides = strides == default_strides

    def as_array_metadata(self) -> ArrayMetadata:
        return self

    def with_(self, dtype: DTypeLike | None = None) -> ArrayMetadata:
        """Replaces a property of the metadata and returns a new metadata object."""
        return ArrayMetadata(
            shape=self.shape,
            dtype=self.dtype if dtype is None else dtype,
            strides=self.strides,
            first_element_offset=self.first_element_offset,
            buffer_size=self.buffer_size,
        )

    def _basis(self) -> tuple[numpy.dtype[Any], tuple[int, ...], tuple[int, ...], int, int]:
        return (
            self.dtype,
            self.shape,
            self.strides,
            self.first_element_offset,
            self.buffer_size,
        )

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ArrayMetadata) and self._basis() == other._basis()

    def __hash__(self) -> int:
        return hash((type(self), self._basis()))

    def get_sub_region(self, origin: int, size: int) -> ArrayMetadata:
        """
        Returns the same metadata shape-wise, but for the given subregion
        of the original buffer.
        """
        # The size errors will be checked by ArrayMetadata constructor
        return ArrayMetadata(
            shape=self.shape,
            dtype=self.dtype,
            strides=self.strides,
            first_element_offset=self.first_element_offset - origin,
            buffer_size=size,
        )

    def __getitem__(self, slices: slice | tuple[slice, ...]) -> ArrayMetadata:
        """
        Returns the view of this metadata with the given ranges,
        with the offsets and buffer size corresponding to the original buffer.
        """
        if isinstance(slices, slice):
            slices = (slices,)
        if len(slices) < len(self.shape):
            slices += (slice(None),) * (len(self.shape) - len(slices))
        offset, new_shape, new_strides = _get_view(self.shape, self.strides, slices)
        return ArrayMetadata(
            shape=new_shape,
            dtype=self.dtype,
            strides=new_strides,
            first_element_offset=self.first_element_offset + offset,
            buffer_size=self.buffer_size,
        )

    def __repr__(self) -> str:
        args = [f"dtype={self.dtype}", f"shape={self.shape}"]
        if not self._default_strides:
            args.append(f"strides={self.strides}")
        if self.first_element_offset != 0:
            args.append(f"first_element_offset={self.first_element_offset}")
        if self.buffer_size != self.min_offset + self.span:
            args.append(f"buffer_size={self.buffer_size}")
        args_str = ", ".join(args)
        return f"ArrayMetadata({args_str})"


def _get_strides(shape: tuple[int, ...], itemsize: int) -> tuple[int, ...]:
    # Constructs strides for a contiguous array of shape ``shape`` and item size ``itemsize``.
    strides = []
    stride = itemsize
    for length in reversed(shape):
        strides.append(stride)
        stride *= length
    return tuple(reversed(strides))


def _normalize_slice(length: int, stride: int, slice_: slice) -> tuple[int, int, int]:
    """
    Given a slice over an array of length ``length`` with the stride ``stride`` between elements,
    return a tuple ``(offset, last, stride)`` where ``offset`` is the offset of the first
    element of the resulting view, ``last`` is the index of the last element of the view (0-based),
    and ``stride`` is the new stride between elements.
    """
    start, stop, step = slice_.indices(length)

    offset = start * stride
    total_elems = abs(stop - start)
    length = (total_elems - 1) // abs(step) + 1
    new_stride = stride * step

    return offset, length, new_stride


def _get_view(
    shape: tuple[int, ...], strides: tuple[int, ...], slices: tuple[slice, ...]
) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
    """
    Given an array shape and strides, and a sequence of slices defining a view,
    returns a tuple of three elements: the offset of the first element of the view,
    the view shape and the view strides.
    """
    if len(strides) != len(shape):
        raise ValueError("Shape and strides must have the same length")
    if len(slices) != len(shape):
        raise ValueError("Shape and slices must have the same length")

    offsets, lengths, strides = zip(
        *[
            _normalize_slice(length, stride, slice_)
            for length, stride, slice_ in zip(shape, strides, slices, strict=True)
        ],
        strict=True,
    )

    return sum(offsets), tuple(lengths), tuple(strides)


def _get_range(shape: tuple[int, ...], itemsize: int, strides: tuple[int, ...]) -> tuple[int, int]:
    """
    Given an array shape, item size (in bytes), and a sequence of strides,
    returns a pair ``(min_offset, max_offset)``,
    where ``min_offset`` is the minimum byte offset of an array element,
    and ``max_offset`` is the maximum byte offset of an array element plus itemsize.
    """
    if len(strides) != len(shape):
        raise ValueError("Shape and strides must have the same length")

    # Now the address of an element (i1, i2, ...) of the resulting view is
    #     addr = i1 * stride1 + i2 * stride2 + ...,
    #     where 0 <= i_k <= length_k - 1
    # We want to find the minimum and the maximum value of addr,
    # keeping in mind that strides may be negative.
    # Since it is a linear function of each index, the extrema will be located
    # at the ends of intervals, so we can find minima and maxima for each term separately.

    # Since we separated the offsets already, for each dimension the address
    # of the first element is 0. We calculate the address of the last byte in each dimension.
    last_addrs = [(length - 1) * stride for length, stride in zip(shape, strides, strict=True)]

    # Sort the pairs (0, last_addr)
    pairs = [(0, last_addr) if last_addr > 0 else (last_addr, 0) for last_addr in last_addrs]
    minima, maxima = zip(*pairs, strict=True)

    min_offset = sum(minima)
    max_offset = sum(maxima) + itemsize

    return min_offset, max_offset
