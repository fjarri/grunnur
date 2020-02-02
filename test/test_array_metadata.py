import itertools

import pytest
import numpy

from grunnur.array_metadata import ArrayMetadata, normalize_slice, get_view, get_range, get_strides


def test_normalize_slice():
    # default slice - returns zero offset and unchanged length/stride
    assert normalize_slice(10, 4, slice(None)) == (0, 10, 4)
    # a slice with end-based limits
    assert normalize_slice(10, 4, slice(-3, -1)) == (7 * 4, 2, 4)
    # a slice with a negative step
    assert normalize_slice(10, 4, slice(-1, -5, -2)) == (9 * 4, 2, -8)

    assert normalize_slice(5, 4, slice(1, 4, 2)) == (1 * 4, 2, 8)


def test_get_view():
    ref = numpy.empty((5, 6), numpy.int32)
    slices = (slice(1, 4, 2), slice(-5, -1, 2))
    ref_v = ref[slices]
    assert (
        get_view(ref.shape, ref.strides, slices) ==
        (
            1 * 24 + (6 - 5) * 4, # the offset of the first element in the view: (1, -5)
            ref_v.shape,
            ref_v.strides))


def ref_range(shape, itemsize, strides):
    # Reference range calculation - simply find the minimum and the maximum across all indices.
    indices = numpy.meshgrid(*[numpy.arange(length) for length in shape], indexing="ij")
    addresses = sum(idx * stride for idx, stride in zip(indices, strides))
    min_offset = addresses.min()
    max_offset = addresses.max() + itemsize
    return min_offset, max_offset


def test_get_range():
    min_offset, max_offset = get_range((3, 4, 5), 4, (100, -30, -5))
    ref_min_offset, ref_max_offset = ref_range((3, 4, 5), 4, (100, -30, -5))
    assert min_offset == ref_min_offset
    assert max_offset == ref_max_offset


def test_get_strides():
    ref = numpy.empty((5, 6), numpy.int32)
    strides = get_strides((5, 6), ref.dtype.itemsize)
    assert strides == ref.strides


def check_metadata(meta, check_max=False):
    """
    Checks array metadata by creating a buffer of the size specified in the metadata
    and trying to access every element there.
    """
    buf = numpy.zeros(meta.buffer_size, numpy.uint8)

    itemsize = meta.dtype.itemsize

    for indices in itertools.product(*[range(length) for length in meta.shape]):
        flat_idx = sum(idx * stride for idx, stride in zip(indices, meta.strides))
        addr = flat_idx + meta.first_element_offset
        buf[addr:addr+itemsize] = 1

    nz = numpy.flatnonzero(buf)
    min_addr = nz[0]
    max_addr = nz[-1] + 1

    if check_max:
        assert max_addr == meta.buffer_size


def test_metadata_constructor():
    # a scalar shape is converted into a tuple
    check_metadata(ArrayMetadata(5, numpy.float64), check_max=True)

    # strides are created automatically if not provided
    meta = ArrayMetadata((6, 7), numpy.complex128)
    check_metadata(meta, check_max=True)
    ref = numpy.empty((6, 7), numpy.complex128)
    assert meta.strides == ref.strides

    # setting strides manually
    check_metadata(ArrayMetadata((6, 7), numpy.complex128, strides=[200, 20]), check_max=True)

    # setting buffer size
    check_metadata(ArrayMetadata((5, 6), numpy.complex64, strides=[100, 10], buffer_size=1000))

    # Minimum offset is too small
    with pytest.raises(ValueError):
        meta = ArrayMetadata(
            (4, 5), numpy.int32, strides=(20, -4),
            first_element_offset=0, buffer_size=100)

    # Minimum offset is too big
    with pytest.raises(ValueError):
        meta = ArrayMetadata(
            (4, 5), numpy.int32, strides=(20, 4),
            first_element_offset=120, buffer_size=100)

    # Maximum offset is too big
    with pytest.raises(ValueError):
        meta = ArrayMetadata(
            (4, 5), numpy.int32, strides=(20, 4),
            first_element_offset=0, buffer_size=70)


def test_view():
    meta = ArrayMetadata((5, 6), numpy.int32)
    view = meta[1:4, -1:-5:-2]
    ref_view = numpy.empty(meta.shape, meta.dtype)[1:4, -1:-5:-2]
    assert view.shape == ref_view.shape
    assert view.strides == ref_view.strides

    meta = ArrayMetadata((5, 6), numpy.int32)
    view = meta[1:4] # omitting the innermost slices
    ref_view = numpy.empty(meta.shape, meta.dtype)[1:4]
    assert view.shape == ref_view.shape
    assert view.strides == ref_view.strides


def test_minimal_subregion():
    meta = ArrayMetadata((5, 6), numpy.int32)
    view = meta[1:4, -1:-5:-2]
    origin, size, new_meta = view.minimal_subregion()

    # the address of the elem with the lowest address, that is (1, -3) == (1, 3)
    assert origin == 1 * meta.strides[0] + 3 * meta.strides[1]

    # the distance between the lowest address ((1, -3) == (1, 3))
    # and the highest one ((3, -1) == (3, 5)) plus itemsize
    assert size == (
        (3 * meta.strides[0] + 5 * meta.strides[1]) + meta.dtype.itemsize
        - (1 * meta.strides[0] + 3 * meta.strides[1]))

    # The new first element address is the address of (1, -1) == (1, 5)
    # minus the origin.
    assert new_meta.first_element_offset == 1 * meta.strides[0] + 5 * meta.strides[1] - origin
    assert new_meta.buffer_size == size
