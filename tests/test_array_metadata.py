import itertools
import re

import numpy
import pytest

from grunnur import ArrayMetadata
from grunnur._array_metadata import (
    _get_range,
    _get_strides,
    _get_view,
    _normalize_slice,
)


def test_normalize_slice() -> None:
    # default slice - returns zero offset and unchanged length/stride
    assert _normalize_slice(10, 4, slice(None)) == (0, 10, 4)
    # a slice with end-based limits
    assert _normalize_slice(10, 4, slice(-3, -1)) == (7 * 4, 2, 4)
    # a slice with a negative step
    assert _normalize_slice(10, 4, slice(-1, -5, -2)) == (9 * 4, 2, -8)

    assert _normalize_slice(5, 4, slice(1, 4, 2)) == (1 * 4, 2, 8)


def test_get_view() -> None:
    ref = numpy.empty((5, 6), numpy.int32)
    slices = (slice(1, 4, 2), slice(-5, -1, 2))
    ref_v = ref[slices]
    assert _get_view(ref.shape, ref.strides, slices) == (
        1 * 24 + (6 - 5) * 4,  # the offset of the first element in the view: (1, -5)
        ref_v.shape,
        ref_v.strides,
    )

    with pytest.raises(ValueError, match="Shape and strides must have the same length"):
        _get_view(ref.shape, ref.strides[:-1], slices)

    with pytest.raises(ValueError, match="Shape and slices must have the same length"):
        _get_view(ref.shape, ref.strides, slices[:-1])


def ref_range(shape: tuple[int, ...], itemsize: int, strides: tuple[int, ...]) -> tuple[int, int]:
    # Reference range calculation - simply find the minimum and the maximum across all indices.
    indices = numpy.meshgrid(*[numpy.arange(length) for length in shape], indexing="ij")
    addresses = sum(
        (idx * stride for idx, stride in zip(indices, strides, strict=True)),
        start=numpy.zeros(shape),
    )
    min_offset = addresses.min()
    max_offset = addresses.max() + itemsize
    return min_offset, max_offset


def test_get_range() -> None:
    min_offset, max_offset = _get_range((3, 4, 5), 4, (100, -30, -5))
    ref_min_offset, ref_max_offset = ref_range((3, 4, 5), 4, (100, -30, -5))
    assert min_offset == ref_min_offset
    assert max_offset == ref_max_offset

    with pytest.raises(ValueError, match="Shape and strides must have the same length"):
        _get_range((3, 4, 5), 4, (100, -30))


def test_get_strides() -> None:
    ref = numpy.empty((5, 6), numpy.int32)
    strides = _get_strides((5, 6), ref.dtype.itemsize)
    assert strides == ref.strides


def check_metadata(meta: ArrayMetadata) -> None:
    """
    Checks array metadata by testing that address of every element of the array
    lies within the range specified in the metadata.
    """
    itemsize = meta.dtype.itemsize

    addresses = [
        sum(idx * stride for idx, stride in zip(indices, meta.strides, strict=True))
        for indices in itertools.product(*[range(length) for length in meta.shape])
    ]

    addresses_array = numpy.array(addresses)

    assert addresses_array.max() + itemsize - addresses_array.min() == meta.span
    assert meta.first_element_offset + addresses_array.min() == meta.min_offset
    assert meta.first_element_offset + addresses_array.max() + itemsize <= meta.buffer_size


def test_metadata_constructor() -> None:
    # a scalar shape is converted into a tuple
    check_metadata(ArrayMetadata([5], numpy.float64))

    # strides are created automatically if not provided
    meta = ArrayMetadata((6, 7), numpy.complex128)
    check_metadata(meta)
    ref = numpy.empty((6, 7), numpy.complex128)
    assert meta.strides == ref.strides

    # setting strides manually
    check_metadata(ArrayMetadata((6, 7), numpy.complex128, strides=[200, 20]))
    check_metadata(ArrayMetadata((6, 7), numpy.complex128, strides=[200, -20]))

    # Buffer overflow
    with pytest.raises(ValueError, match="First element offset is smaller than the minimum 20"):
        ArrayMetadata((5, 6), numpy.int32, strides=(24, -4), first_element_offset=8)
    with pytest.raises(ValueError, match="Buffer size is smaller than the minimum 120"):
        ArrayMetadata((5, 6), numpy.int32, buffer_size=5 * 6 * 4 - 1)


def test_eq() -> None:
    meta1 = ArrayMetadata((5, 6), numpy.int32)
    meta2 = ArrayMetadata((5, 6), numpy.int32)
    meta3 = ArrayMetadata((5, 6), numpy.int32, strides=(48, 4))
    assert meta1 == meta2
    assert meta1 != meta3


def test_hash() -> None:
    meta1 = ArrayMetadata((5, 6), numpy.int32)
    meta2 = ArrayMetadata((5, 6), numpy.int32)
    meta3 = ArrayMetadata((5, 6), numpy.int32, strides=(48, 4))
    assert hash(meta1) == hash(meta2)
    assert hash(meta1) != hash(meta3)


def test_repr() -> None:
    meta = ArrayMetadata((5, 6), numpy.int32)
    assert repr(meta) == "ArrayMetadata(dtype=int32, shape=(5, 6))"

    meta = ArrayMetadata((5, 6), numpy.int32, strides=(24, 4))
    assert repr(meta) == "ArrayMetadata(dtype=int32, shape=(5, 6))"
    meta = ArrayMetadata((5, 6), numpy.int32, strides=(48, 4))
    assert repr(meta) == "ArrayMetadata(dtype=int32, shape=(5, 6), strides=(48, 4))"

    meta = ArrayMetadata((5, 6), numpy.int32, first_element_offset=0)
    assert repr(meta) == "ArrayMetadata(dtype=int32, shape=(5, 6))"
    meta = ArrayMetadata((5, 6), numpy.int32, first_element_offset=12)
    assert repr(meta) == "ArrayMetadata(dtype=int32, shape=(5, 6), first_element_offset=12)"

    meta = ArrayMetadata((5, 6), numpy.int32, buffer_size=5 * 6 * 4)
    assert repr(meta) == "ArrayMetadata(dtype=int32, shape=(5, 6))"
    meta = ArrayMetadata((5, 6), numpy.int32, buffer_size=512)
    assert repr(meta) == "ArrayMetadata(dtype=int32, shape=(5, 6), buffer_size=512)"


def test_from_arraylike() -> None:
    meta = ArrayMetadata.from_arraylike(numpy.empty((5, 6), numpy.int32))
    assert meta.shape == (5, 6)
    assert meta.dtype == numpy.int32
    assert meta.strides == (24, 4)

    meta = ArrayMetadata.from_arraylike(
        ArrayMetadata(shape=(5, 6), dtype=numpy.int32, strides=(48, 4))
    )
    assert meta.shape == (5, 6)
    assert meta.dtype == numpy.int32
    assert meta.strides == (48, 4)


def test_view() -> None:
    meta = ArrayMetadata((5, 6), numpy.int32)
    view = meta[1:4, -1:-5:-2]
    ref_view = numpy.empty(meta.shape, meta.dtype)[1:4, -1:-5:-2]
    # First element is [1, -1] == [1, 5] of the original array
    assert view.first_element_offset == 1 * 6 * 4 + 5 * 4
    assert view.shape == ref_view.shape
    assert view.strides == ref_view.strides
    assert view.buffer_size == meta.buffer_size
    # The element with the minimum address is [1, -3] == [1, 3] of the original array
    assert view.min_offset == 1 * 6 * 4 + 3 * 4
    # The element with the maximum address is [3, -1] == [3, 5] of the original array.
    # It is located 2 * 4 * 6 (2 elements in the dimension 0 times the stride 4 * 6)
    # bytes after the first element of the view.
    # The minimum buffer size is the first element offset plus this offset plus the element size (4)
    assert view.span == (view.first_element_offset + 2 * 4 * 6 + 4) - view.min_offset

    meta = ArrayMetadata((5, 6), numpy.int32)
    view = meta[1:4]  # omitting the innermost slices
    ref_view = numpy.empty(meta.shape, meta.dtype)[1:4]
    # First element is [1, 0] of the original array
    assert view.first_element_offset == 1 * 6 * 4
    assert view.shape == ref_view.shape
    assert view.strides == ref_view.strides
    assert view.buffer_size == meta.buffer_size
    # The leftmost element is also the first
    assert view.min_offset == view.first_element_offset
    # Three lines of stride 4 * 6 each
    assert view.span == 3 * 4 * 6


def test_empty_shape() -> None:
    with pytest.raises(ValueError, match="Array shape cannot be an empty sequence"):
        ArrayMetadata((), numpy.int32)


def test_get_sub_region() -> None:
    meta = ArrayMetadata((5, 6), numpy.int32)
    view = meta[1:4]

    view_region = view.get_sub_region(0, meta.buffer_size)
    assert view_region.first_element_offset == view.first_element_offset
    assert view_region.buffer_size == view.buffer_size

    span = 3 * 6 * 4  # 3 lines of 6 elements of 4 bytes each
    # The new first element offset will be 24 - 8 == 16
    view_region = view.get_sub_region(8, 16 + span + 1)
    assert view_region.first_element_offset == view.first_element_offset - 8
    assert view_region.buffer_size == 16 + span + 1


def test_with() -> None:
    meta = ArrayMetadata(
        (5, 6), numpy.int32, strides=(1, 2), first_element_offset=8, buffer_size=1000
    )
    meta2 = meta.with_(dtype=numpy.float32)
    assert meta2 == ArrayMetadata(
        (5, 6), numpy.float32, strides=(1, 2), first_element_offset=8, buffer_size=1000
    )
