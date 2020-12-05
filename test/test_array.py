import numpy
import pytest

from grunnur import Array, Buffer, Queue
from grunnur.array_metadata import ArrayMetadata


def test_single_device(mock_or_real_context):

    context, _mocked = mock_or_real_context

    arr = numpy.arange(100)
    queue = Queue.on_all_devices(context)
    arr_dev = Array.from_host(queue, arr)

    # get to a new array
    assert (arr_dev.get() == arr).all()

    # get to an existing array
    res = numpy.empty(arr.shape, arr.dtype)
    arr_dev.get(dest=res)
    assert (res == arr).all()

    # async get
    res = arr_dev.get(async_=True)
    queue.synchronize()
    assert (res == arr).all()

    # async set
    arr2 = numpy.arange(100) + 1
    arr_dev.set(arr2)
    assert (arr_dev.get() == arr2).all()

    # sync set
    arr2 = numpy.arange(100) + 2
    arr_dev.set(arr2, no_async=True)
    assert (arr_dev.get() == arr2).all()

    # async set from another array
    arr2 = numpy.arange(100) + 3
    arr2_dev = Array.from_host(queue, arr2)
    arr_dev.set(arr2_dev)
    assert (arr_dev.get() == arr2).all()

    # sync set from another array
    arr2 = numpy.arange(100) + 4
    arr2_dev = Array.from_host(queue, arr2)
    arr_dev.set(arr2_dev, no_async=True)
    assert (arr_dev.get() == arr2).all()


def test_set_from_non_contiguous(mock_or_real_context):

    context, _mocked = mock_or_real_context

    queue = Queue.on_all_devices(context)
    arr = Array.empty(queue, (10, 20), numpy.int32)
    arr2 = Array.empty(queue, (20, 20), numpy.int32)

    with pytest.raises(ValueError, match="Setting from a non-contiguous device array is not supported"):
        arr.set(arr2[::2, :])

    # Can set from a non-contiguous numpy array though
    arr.set(numpy.ones((20, 20), numpy.int32)[::2, :])
    assert (arr.get() == 1).all()


def test_multi_device(mock_4_device_context):

    context = mock_4_device_context

    arr = numpy.arange(100)
    queue = Queue.on_all_devices(context)
    arr_dev = Array.from_host(queue, arr)

    with pytest.raises(ValueError, match="The device number must be one of those present in the queue"):
        arr_dev.single_device_view(len(context.devices))

    arr_dev0 = arr_dev.single_device_view(0)[:50]
    arr_dev1 = arr_dev.single_device_view(1)[50:]

    assert (arr_dev0.get() == arr[:50]).all()
    assert (arr_dev1.get() == arr[50:]).all()


def test_from_host(mock_context):
    context = mock_context
    arr = numpy.arange(100)

    queue = Queue.on_all_devices(context)
    arr_dev = Array.from_host(queue, arr)
    assert (arr_dev.get() == arr).all()


def test_empty(mock_context):
    context = mock_context
    queue = Queue.on_all_devices(context)
    arr_dev = Array.empty(queue, 100, numpy.int32)
    arr = arr_dev.get()
    assert arr.shape == (100,)
    assert arr.dtype == numpy.int32


def test_custom_allocator(mock_context):
    context = mock_context
    queue = Queue.on_all_devices(context)
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
    queue = Queue.on_all_devices(context)

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


def test_set_checks_shape(mock_context):
    context = mock_context
    queue = Queue.on_all_devices(context)
    arr = Array.empty(queue, (10, 20), numpy.int32)

    with pytest.raises(ValueError, match="Shape mismatch: expected \\(10, 20\\), got \\(10, 30\\)"):
        arr.set(numpy.zeros((10, 30), numpy.int32))

    with pytest.raises(ValueError, match="Dtype mismatch: expected int32, got int64"):
        arr.set(numpy.zeros((10, 20), numpy.int64))


def test_set_from_wrong_type(mock_context):
    context = mock_context
    queue = Queue.on_all_devices(context)
    arr = Array.empty(queue, (10, 20), numpy.int32)
    with pytest.raises(TypeError, match="Cannot set from an object of type <class 'int'>"):
        arr.set(1)
