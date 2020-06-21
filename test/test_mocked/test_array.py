import numpy
import pytest

from grunnur import Array, Buffer, Queue
from grunnur.array_metadata import ArrayMetadata

from ..test_on_device.test_array import _test_single_device, _test_multi_device


def test_single_device(mock_context):
    _test_single_device(context=mock_context)


def test_multi_device(mock_4_device_context):
    _test_multi_device(context=mock_4_device_context)


def test_from_host(mock_context):
    context = mock_context
    arr = numpy.arange(100)

    queue = Queue.from_device_idxs(context)
    arr_dev = Array.from_host(queue, arr)
    assert (arr_dev.get() == arr).all()


def test_empty(mock_context):
    context = mock_context
    queue = Queue.from_device_idxs(context)
    arr_dev = Array.empty(queue, 100, numpy.int32)
    arr = arr_dev.get()
    assert arr.shape == (100,)
    assert arr.dtype == numpy.int32


def test_custom_allocator(mock_context):
    context = mock_context
    queue = Queue.from_device_idxs(context)
    allocated = []
    def allocator(size):
        allocated.append(size)
        return Buffer.allocate(context, size)
    arr_dev = Array.empty(queue, 100, numpy.int32, allocator=allocator)
    arr = arr_dev.get()
    assert arr.shape == (100,)
    assert arr.dtype == numpy.int32
    assert allocated == [arr.size * arr.dtype.itemsize]


def test_custom_buffer(mock_context):
    context = mock_context
    queue = Queue.from_device_idxs(context)

    arr = numpy.arange(100).astype(numpy.int32)
    metadata = ArrayMetadata.from_arraylike(arr)

    data = Buffer.allocate(context, 100)
    with pytest.raises(ValueError, match="Provided data buffer is not big enough to hold the array"):
        Array(queue, metadata, data=data)

    bigger_data = Buffer.allocate(context, arr.size * arr.dtype.itemsize)
    bigger_data.set(queue, arr)

    arr_dev = Array(queue, metadata, data=bigger_data)
    res = arr_dev.get()
    assert (res == arr).all()
