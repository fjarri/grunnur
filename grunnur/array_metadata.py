from __future__ import annotations

from typing import Tuple, Iterable, Optional

import numpy

from .dtypes import normalize_type
from .utils import wrap_in_tuple


class ArrayMetadata:
    """
    A helper object for array-like classes that handles shape/strides/buffer size checks
    without actual data attached to it.
    """

    def __init__(
            self, shape: Iterable[int], dtype: numpy.dtype,
            strides: Optional[Iterable[int]]=None,
            first_element_offset: int=0,
            buffer_size: Optional[int]=None):

        shape = wrap_in_tuple(shape)
        dtype = normalize_type(dtype)

        if strides is None:
            strides = get_strides(shape, dtype.itemsize)

        min_offset, max_offset = get_range(shape, dtype.itemsize, strides)
        if buffer_size is None:
            buffer_size = first_element_offset + max_offset

        full_min_offset = first_element_offset + min_offset
        if full_min_offset < 0 or full_min_offset + dtype.itemsize > buffer_size:
            raise ValueError(
                f"The minimum offset for given strides ({full_min_offset}) "
                f"is outside the given buffer range ({buffer_size})")

        full_max_offset = first_element_offset + max_offset
        if full_max_offset > buffer_size:
            raise ValueError(
                f"The maximum offset for given strides ({full_max_offset}) "
                f"is outside the given buffer range ({buffer_size})")

        self.shape = shape
        self.dtype = dtype
        self.strides = strides
        self.first_element_offset = first_element_offset
        self._full_min_offset = full_min_offset
        self._full_max_offset = full_max_offset
        self.buffer_size = buffer_size

    def minimal_subregion(self) -> Tuple[int, int, ArrayMetadata]:
        """
        Returns the metadata for the minimal subregion that fits all the data in this view,
        along with the subgregion offset in the current buffer and the required subregion length.
        """
        subregion_origin = self._full_min_offset
        subregion_size = self._full_max_offset - self._full_min_offset
        new_metadata = ArrayMetadata(
            self.shape, self.dtype,
            strides=self.strides,
            first_element_offset=self.first_element_offset - self._full_min_offset,
            buffer_size=subregion_size)
        return subregion_origin, subregion_size, new_metadata

    def __getitem__(self, slices):
        slices = wrap_in_tuple(slices)
        new_fe_offset, new_shape, new_strides = get_view(self.shape, self.strides, slices)
        return ArrayMetadata(
            new_shape, self.dtype, strides=new_strides, first_element_offset=new_fe_offset)


def get_strides(shape: Iterable[int], itemsize: int) -> Tuple[int]:
    # Constructs strides for a contiguous array of shape ``shape`` and item size ``itemsize``.
    strides = []
    stride = itemsize
    for length in reversed(shape):
        strides.append(stride)
        stride *= length
    return tuple(reversed(strides))


def normalize_slice(length: int, stride: int, slice_: slice) -> Tuple[int, int, int]:
    """
    Given a slice over an array of length ``length`` with the stride ``stride`` between elements,
    return a tuple ``(offset, last, stride)`` where ``offset`` is the offset of the first
    element of the resulting view, ``last`` is the index of the last element of the view (0-based),
    and ``stride`` is the new stride between elements.
    """
    start = 0 if slice_.start is None else slice_.start
    stop = length - 1 if slice_.stop is None else slice_.stop - 1
    step = 1 if slice_.step is None else slice_.step

    start = start if start >= 0 else length + start
    stop = stop if stop >= 0 else length + stop

    offset = start * stride
    length = (stop - start + 1) // step
    new_stride = stride * step

    return offset, length, new_stride


def get_view(
        shape: Iterable[int], strides: Iterable[int], slices: Iterable[slice]) \
        -> Tuple[int, Tuple[int], Tuple[int]]:
    """
    Given an array shape and strides, and a sequence of slices defining a view,
    returns a tuple of three elements: the offset of the first element of the view,
    the view shape and the view strides.
    """
    assert len(slices) == len(shape)
    assert len(strides) == len(shape)

    offsets, lengths, strides = zip(*[
        normalize_slice(length, stride, slice_)
        for length, stride, slice_ in zip(shape, strides, slices)])

    return sum(offsets), tuple(lengths), tuple(strides)


def get_range(shape: Iterable[int], itemsize: int, strides: Iterable[int]) -> Tuple[int, int]:
    """
    Given an array shape, item size (in bytes), and a sequence of strides,
    returns a pair ``(min_offset, max_offset)``,
    where ``min_offset`` is the minimum byte offset of an array element,
    and ``max_offset`` is the maximum byte offset of an array element plus itemsize.
    """
    assert len(strides) == len(shape)

    # Now the address of an element (i1, i2, ...) of the resulting view is
    #     addr = i1 * stride1 + i2 * stride2 + ...,
    #     where 0 <= i_k <= length_k - 1
    # We want to find the minimum and the maximum value of addr,
    # keeping in mind that strides may be negative.
    # Since it is a linear function of each index, the extrema will be located
    # at the ends of intervals, so we can find minima and maxima for each term separately.

    # Since we separated the offsets already, for each dimension the address
    # of the first element is 0. We calculate the address of the last byte in each dimension.
    last_addrs = [(length - 1) * stride for length, stride in zip(shape, strides)]

    # Sort the pairs (0, last_addr)
    pairs = [(0, last_addr) if last_addr > 0 else (last_addr, 0) for last_addr in last_addrs]
    minima, maxima = zip(*pairs)

    min_offset = sum(minima)
    max_offset = sum(maxima) + itemsize

    return min_offset, max_offset
