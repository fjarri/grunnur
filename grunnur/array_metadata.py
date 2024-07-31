from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol, runtime_checkable

from .dtypes import _normalize_type

if TYPE_CHECKING:  # pragma: no cover
    import numpy
    from numpy.typing import DTypeLike


@runtime_checkable
class ArrayMetadataLike(Protocol):
    """
    A protocol for an object providing array metadata.
    :py:class:`numpy.ndarray` or :py:class:`Array` follow this protocol.
    """

    @property
    def shape(self) -> tuple[int, ...]:
        """Array shape."""

    @property
    def dtype(self) -> numpy.dtype[Any]:
        """The type of an array element."""


class NormalizedArgs(NamedTuple):
    shape: tuple[int, ...]
    dtype: numpy.dtype[Any]
    strides: tuple[int, ...]
    default_strides: bool


def _normalize_args(
    shape: Sequence[int] | int,
    dtype: DTypeLike,
    *,
    strides: Sequence[int] | None = None,
) -> NormalizedArgs:
    shape = tuple(shape) if isinstance(shape, Sequence) else (shape,)

    if len(shape) == 0:
        raise ValueError("Array shape cannot be an empty sequence")

    dtype = _normalize_type(dtype)

    default_strides = _get_strides(shape, dtype.itemsize)
    strides = default_strides if strides is None else tuple(strides)

    return NormalizedArgs(
        shape=shape, dtype=dtype, strides=strides, default_strides=strides == default_strides
    )


class ArrayMetadata:
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

    is_contiguous: bool
    """If ``True``, means that array's data forms a continuous chunk of memory."""

    @classmethod
    def from_arraylike(cls, array: ArrayMetadataLike) -> ArrayMetadata:
        return cls(shape=array.shape, dtype=array.dtype, strides=getattr(array, "strides", None))

    @classmethod
    def padded(
        cls,
        shape: Sequence[int] | int,
        dtype: DTypeLike,
        *,
        pad: Sequence[int] | int,
    ) -> ArrayMetadata:
        normalized = _normalize_args(shape=shape, dtype=dtype)
        pad = tuple(pad) if isinstance(pad, Sequence) else (pad,) * len(normalized.shape)

        if len(normalized.shape) != len(pad):
            raise ValueError(
                "`pad` must be either an integer or a sequence of the same length as `shape`"
            )

        padded_shape = [
            dim_len + dim_pad * 2 for dim_len, dim_pad in zip(normalized.shape, pad, strict=True)
        ]

        # A little inefficiency here, we will be normalizing the arguments twice
        full_metadata = cls(shape=padded_shape, dtype=normalized.dtype)

        slices = tuple(
            slice(dim_pad, dim_len + dim_pad)
            for dim_len, dim_pad in zip(normalized.shape, pad, strict=True)
        )
        return full_metadata[slices]

    def __init__(
        self,
        shape: Sequence[int] | int,
        dtype: DTypeLike,
        *,
        strides: Sequence[int] | None = None,
        first_element_offset: int = 0,
        buffer_size: int | None = None,
    ):
        normalized = _normalize_args(shape=shape, dtype=dtype, strides=strides)
        shape = normalized.shape
        dtype = normalized.dtype
        strides = normalized.strides

        min_offset, max_offset = _get_range(shape, dtype.itemsize, strides)
        default_buffer_size = first_element_offset + max_offset
        if buffer_size is None:
            buffer_size = default_buffer_size

        full_min_offset = first_element_offset + min_offset
        if full_min_offset < 0 or full_min_offset + dtype.itemsize > buffer_size:
            raise ValueError(
                f"The minimum offset for given strides ({full_min_offset}) "
                f"is outside the given buffer range ({buffer_size})"
            )

        full_max_offset = first_element_offset + max_offset
        if full_max_offset > buffer_size:
            raise ValueError(
                f"The maximum offset for given strides ({full_max_offset}) "
                f"is outside the given buffer range ({buffer_size})"
            )

        self.shape = shape
        self.dtype = dtype
        self.strides = strides
        self.first_element_offset = first_element_offset
        self.buffer_size = buffer_size

        # Technically, an array with non-default (e.g., overlapping) strides
        # can be contioguous, but that's too hard to determine.
        self.is_contiguous = normalized.default_strides

        self._full_min_offset = full_min_offset
        self._full_max_offset = full_max_offset
        self._default_strides = normalized.default_strides
        self._default_buffer_size = buffer_size == default_buffer_size

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, ArrayMetadata)
            and self.shape == other.shape
            and self.dtype == other.dtype
            and self.strides == other.strides
            and self.first_element_offset == other.first_element_offset
            and self.buffer_size == other.buffer_size
        )

    def __hash__(self) -> int:
        return hash((type(self), self.dtype, self.shape, self.strides, self.first_element_offset))

    def minimal_subregion(self) -> tuple[int, int, ArrayMetadata]:
        """
        Returns the metadata for the minimal subregion that fits all the data in this view,
        along with the subgregion offset in the current buffer and the required subregion length.
        """
        subregion_origin = self._full_min_offset
        subregion_size = self._full_max_offset - self._full_min_offset
        new_metadata = ArrayMetadata(
            self.shape,
            self.dtype,
            strides=self.strides,
            first_element_offset=self.first_element_offset - self._full_min_offset,
            buffer_size=subregion_size,
        )
        return subregion_origin, subregion_size, new_metadata

    def __getitem__(self, slices: slice | tuple[slice, ...]) -> ArrayMetadata:
        if isinstance(slices, slice):
            slices = (slices,)
        if len(slices) < len(self.shape):
            slices += (slice(None),) * (len(self.shape) - len(slices))
        new_fe_offset, new_shape, new_strides = _get_view(self.shape, self.strides, slices)
        return ArrayMetadata(
            new_shape,
            self.dtype,
            strides=new_strides,
            first_element_offset=new_fe_offset,
            buffer_size=self.buffer_size,
        )

    def __repr__(self) -> str:
        args = [f"dtype={self.dtype}", f"shape={self.shape}"]
        if self.first_element_offset != 0:
            args.append(f"first_element_offset={self.first_element_offset}")
        if not self._default_strides:
            args.append(f"strides={self.strides}")
        if not self._default_buffer_size:
            args.append(f"buffer_size={self.buffer_size}")
        args_str = ", ".join(args)
        return f"ArrayMetadata({args_str})"


def _get_strides(shape: Sequence[int], itemsize: int) -> tuple[int, ...]:
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
    shape: Sequence[int], strides: Sequence[int], slices: Sequence[slice]
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


def _get_range(shape: Sequence[int], itemsize: int, strides: Sequence[int]) -> tuple[int, int]:
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
