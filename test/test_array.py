import numpy
import pytest

from grunnur import Array, MultiArray, Buffer, Queue, MultiQueue
from grunnur.array_metadata import ArrayMetadata


def _check_array_operations(context, queue_cls, array_cls):

    arr = numpy.arange(100)
    queue = queue_cls(context)
    arr_dev = array_cls.from_host(queue, arr)

    # get to a new array
    assert (arr_dev.get(queue) == arr).all()

    # get to an existing array
    res = numpy.empty(arr.shape, arr.dtype)
    arr_dev.get(queue, dest=res)
    assert (res == arr).all()

    # async get
    res = arr_dev.get(queue, async_=True)
    queue.synchronize()
    assert (res == arr).all()

    # async set
    arr2 = numpy.arange(100) + 1
    arr_dev.set(queue, arr2)
    assert (arr_dev.get(queue) == arr2).all()

    # sync set
    arr2 = numpy.arange(100) + 2
    arr_dev.set(queue, arr2, no_async=True)
    assert (arr_dev.get(queue) == arr2).all()

    # async set from another array
    arr2 = numpy.arange(100) + 3
    arr2_dev = array_cls.from_host(queue, arr2)
    arr_dev.set(queue, arr2_dev)
    assert (arr_dev.get(queue) == arr2).all()

    # sync set from another array
    arr2 = numpy.arange(100) + 4
    arr2_dev = array_cls.from_host(queue, arr2)
    arr_dev.set(queue, arr2_dev, no_async=True)
    assert (arr_dev.get(queue) == arr2).all()


def test_single_device(mock_or_real_context):
    context, _mocked = mock_or_real_context
    _check_array_operations(context, Queue, Array)


def test_set_from_non_contiguous(mock_or_real_context):

    context, _mocked = mock_or_real_context

    queue = Queue(context)
    arr = Array.empty(context, (10, 20), numpy.int32)
    arr2 = Array.empty(context, (20, 20), numpy.int32)

    with pytest.raises(ValueError, match="Setting from a non-contiguous device array is not supported"):
        arr.set(queue, arr2[::2, :])

    # Can set from a non-contiguous numpy array though
    arr.set(queue, numpy.ones((20, 20), numpy.int32)[::2, :])
    assert (arr.get(queue) == 1).all()


def test_from_host(mock_or_real_context):
    context, _mocked = mock_or_real_context
    queue = Queue(context)
    arr = numpy.arange(100)
    arr_dev = Array.from_host(queue, arr)
    assert (arr_dev.get(queue) == arr).all()


def test_empty(mock_or_real_context):
    context, _mocked = mock_or_real_context
    queue = Queue(context)
    arr_dev = Array.empty(context, 100, numpy.int32)
    arr = arr_dev.get(queue)
    assert arr.shape == (100,)
    assert arr.dtype == numpy.int32


def test_multi_device(mock_or_real_multi_device_context):
    context, _mocked = mock_or_real_multi_device_context
    _check_array_operations(context, MultiQueue, MultiArray)


def test_multi_device_from_host(mock_or_real_multi_device_context):
    context, _mocked = mock_or_real_multi_device_context
    mqueue = MultiQueue(context)
    arr = numpy.arange(100)
    arr_dev = MultiArray.from_host(mqueue, arr)
    assert (arr_dev.get(mqueue) == arr).all()


def test_multi_device_empty(mock_or_real_multi_device_context):
    context, _mocked = mock_or_real_multi_device_context
    mqueue = MultiQueue(context)

    arr_dev = MultiArray.empty(context, 100, numpy.int32)
    arr = arr_dev.get(mqueue)
    assert arr.shape == (100,)
    assert arr.dtype == numpy.int32

    # explicit device_idxs
    arr_dev = MultiArray.empty(context, 100, numpy.int32, device_idxs=[0])
    assert list(arr_dev.subarrays.keys()) == [0]
    arr = arr_dev.get(mqueue)
    assert arr.shape == (100,)
    assert arr.dtype == numpy.int32

    # explicit splay
    arr_dev = MultiArray.empty(context, 100, numpy.int32, splay=MultiArray.EqualSplay())
    arr = arr_dev.get(mqueue)
    assert arr.shape == (100,)
    assert arr.dtype == numpy.int32
    assert (arr_dev.subarrays[0].get(mqueue.queues[0]) == arr[:50]).all()
    assert (arr_dev.subarrays[1].get(mqueue.queues[1]) == arr[50:]).all()


def test_multi_device_mismatched_set(mock_or_real_multi_device_context):
    context, _mocked = mock_or_real_multi_device_context
    mqueue = MultiQueue(context)
    arr_dev = MultiArray.empty(context, 100, numpy.int32)
    arr_dev2 = MultiArray.empty(context, 100, numpy.int32, device_idxs=[0])
    with pytest.raises(ValueError, match="Mismatched device sets in the source and the destination"):
        arr_dev.set(mqueue, arr_dev2)


def test_equal_splay(mock_or_real_multi_device_context):
    context, _mocked = mock_or_real_multi_device_context
    mqueue = MultiQueue(context)
    arr = numpy.arange(101)
    arr_dev = MultiArray.from_host(mqueue, arr, splay=MultiArray.EqualSplay())
    assert (arr_dev.subarrays[0].get(mqueue.queues[0]) == arr[:51]).all()
    assert (arr_dev.subarrays[1].get(mqueue.queues[1]) == arr[51:]).all()

    # Check that the default splay is EqualSplay
    arr_dev = MultiArray.from_host(mqueue, arr)
    assert (arr_dev.subarrays[0].get(mqueue.queues[0]) == arr[:51]).all()
    assert (arr_dev.subarrays[1].get(mqueue.queues[1]) == arr[51:]).all()


def test_clone_splay(mock_or_real_multi_device_context):
    context, _mocked = mock_or_real_multi_device_context
    mqueue = MultiQueue(context)
    arr = numpy.arange(101)
    arr_dev = MultiArray.from_host(mqueue, arr, splay=MultiArray.CloneSplay())
    assert (arr_dev.subarrays[0].get(mqueue.queues[0]) == arr).all()
    assert (arr_dev.subarrays[1].get(mqueue.queues[1]) == arr).all()


def test_custom_allocator(mock_context):
    context = mock_context
    queue = Queue(context)
    allocated = []
    def allocator(size, device_idx):
        allocated.append(size)
        return Buffer.allocate(context, size, device_idx=device_idx)
    arr_dev = Array.empty(context, 100, numpy.int32, allocator=allocator)
    arr = arr_dev.get(queue)
    assert arr.shape == (100,)
    assert arr.dtype == numpy.int32
    assert allocated == [arr.size * arr.dtype.itemsize]


def test_custom_buffer(mock_context):
    context = mock_context
    queue = Queue(context)

    arr = numpy.arange(100).astype(numpy.int32)
    metadata = ArrayMetadata.from_arraylike(arr)

    data = Buffer.allocate(context, 100)
    with pytest.raises(ValueError, match="Provided data buffer is not big enough to hold the array"):
        Array(context, metadata, data=data)

    bigger_data = Buffer.allocate(context, arr.size * arr.dtype.itemsize)
    bigger_data.set(queue, arr)

    arr_dev = Array(context, metadata, data=bigger_data)
    res = arr_dev.get(queue)
    assert (res == arr).all()


def test_set_checks_shape(mock_context):
    context = mock_context
    queue = Queue(context)
    arr = Array.empty(context, (10, 20), numpy.int32)

    with pytest.raises(ValueError, match="Shape mismatch: expected \\(10, 20\\), got \\(10, 30\\)"):
        arr.set(queue, numpy.zeros((10, 30), numpy.int32))

    with pytest.raises(ValueError, match="Dtype mismatch: expected int32, got int64"):
        arr.set(queue, numpy.zeros((10, 20), numpy.int64))


def test_set_from_wrong_type(mock_context):
    context = mock_context
    queue = Queue(context)
    arr = Array.empty(context, (10, 20), numpy.int32)
    with pytest.raises(TypeError, match="Cannot set from an object of type <class 'int'>"):
        arr.set(queue, 1)


def test_empty_in_multi_device_context(mock_4_device_context):
    context = mock_4_device_context
    with pytest.raises(ValueError, match="device_idx must be specified in a multi-device context"):
        Array.empty(context, (10, 20), numpy.int32)
