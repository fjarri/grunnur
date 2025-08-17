import re
import typing

import numpy
import pytest

from grunnur import (
    Array,
    ArrayMetadata,
    BoundDevice,
    Buffer,
    Context,
    MultiArray,
    MultiQueue,
    Queue,
)


# TODO: The `array_cls` choice inside confuses `mypy`.
# Can be solved by introducing an ABC for `Array`/`MultiArray` handling these operations.
@typing.no_type_check
def _check_array_operations(queue: Queue | MultiQueue) -> None:
    array_cls = Array if isinstance(queue, Queue) else MultiArray

    arr = numpy.arange(100)
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
    arr_dev.set(queue, arr2, sync=True)
    assert (arr_dev.get(queue) == arr2).all()

    # async set from another array
    arr2 = numpy.arange(100) + 3
    arr2_dev = array_cls.from_host(queue, arr2)
    arr_dev.set(queue, arr2_dev)
    assert (arr_dev.get(queue) == arr2).all()

    # sync set from another array
    arr2 = numpy.arange(100) + 4
    arr2_dev = array_cls.from_host(queue, arr2)
    arr_dev.set(queue, arr2_dev, sync=True)
    assert (arr_dev.get(queue) == arr2).all()


def test_single_device(mock_or_real_context: tuple[Context, bool]) -> None:
    context, _mocked = mock_or_real_context
    _check_array_operations(Queue(context.device))


def test_set_from_non_contiguous(mock_or_real_context: tuple[Context, bool]) -> None:
    context, _mocked = mock_or_real_context

    queue = Queue(context.device)
    arr = Array.empty(context.device, (10, 20), numpy.int32)
    arr2 = Array.empty(context.device, (20, 20), numpy.int32)

    with pytest.raises(
        ValueError, match="Setting from a non-contiguous device array is not supported"
    ):
        arr.set(queue, arr2[::2, :])

    # Can set from a non-contiguous numpy array though
    arr.set(queue, numpy.ones((20, 20), numpy.int32)[::2, :])
    assert (arr.get(queue) == 1).all()


def test_from_host(mock_or_real_context: tuple[Context, bool]) -> None:
    context, _mocked = mock_or_real_context
    queue = Queue(context.device)
    arr = numpy.arange(100)
    arr_dev = Array.from_host(queue, arr)
    assert (arr_dev.get(queue) == arr).all()

    # Create from a host array given only a device - that is, synchronously
    arr_dev = Array.from_host(context.devices[0], arr)
    assert (arr_dev.get(queue) == arr).all()


def test_empty(mock_or_real_context: tuple[Context, bool]) -> None:
    context, _mocked = mock_or_real_context
    queue = Queue(context.device)
    arr_dev = Array.empty(context.device, [100], numpy.int32)
    arr = arr_dev.get(queue)
    assert arr.shape == (100,)
    assert arr.dtype == numpy.int32


def test_empty_like(mock_or_real_context: tuple[Context, bool]) -> None:
    context, _mocked = mock_or_real_context
    arr = numpy.arange(100).reshape(5, 20).astype(numpy.int32)
    arr_dev = Array.empty_like(context.device, arr)
    assert arr_dev.shape == arr.shape
    assert arr_dev.dtype == arr.dtype
    assert arr_dev.strides == arr.strides

    arr_dev2 = Array.empty_like(context.device, arr_dev)
    assert arr_dev2.metadata == arr_dev.metadata


def test_multi_device(mock_or_real_multi_device_context: tuple[Context, bool]) -> None:
    context, _mocked = mock_or_real_multi_device_context
    _check_array_operations(MultiQueue.on_devices([context.devices[0]]))


def test_multi_device_from_host(mock_or_real_multi_device_context: tuple[Context, bool]) -> None:
    context, _mocked = mock_or_real_multi_device_context
    mqueue = MultiQueue.on_devices(context.devices)
    arr = numpy.arange(100)
    arr_dev = MultiArray.from_host(mqueue, arr)
    assert (arr_dev.get(mqueue) == arr).all()


def test_multi_device_empty(mock_or_real_multi_device_context: tuple[Context, bool]) -> None:
    context, _mocked = mock_or_real_multi_device_context
    mqueue = MultiQueue.on_devices(context.devices)

    arr_dev = MultiArray.empty(context.devices, [100], numpy.int32)
    arr = arr_dev.get(mqueue)
    assert arr.shape == (100,)
    assert arr.dtype == numpy.int32

    # explicit device
    arr_dev = MultiArray.empty(context.devices[0:1], [100], numpy.int32)
    assert list(arr_dev.subarrays.keys()) == [context.devices[0]]
    arr = arr_dev.get(mqueue)
    assert arr.shape == (100,)
    assert arr.dtype == numpy.int32

    # explicit splay
    arr_dev = MultiArray.empty(context.devices, [100], numpy.int32, splay=MultiArray.EqualSplay())
    arr = arr_dev.get(mqueue)
    assert arr.shape == (100,)
    assert arr.dtype == numpy.int32

    device0, device1 = context.devices
    assert (arr_dev.subarrays[device0].get(mqueue.queues[device0]) == arr[:50]).all()
    assert (arr_dev.subarrays[device1].get(mqueue.queues[device1]) == arr[50:]).all()


def test_multi_device_mismatched_set(
    mock_or_real_multi_device_context: tuple[Context, bool],
) -> None:
    context, _mocked = mock_or_real_multi_device_context
    mqueue = MultiQueue.on_devices(context.devices)
    arr_dev = MultiArray.empty(context.devices, [100], numpy.int32)
    arr_dev2 = MultiArray.empty(context.devices[0:1], [100], numpy.int32)
    with pytest.raises(
        ValueError, match="Mismatched device sets in the source and the destination"
    ):
        arr_dev.set(mqueue, arr_dev2)


def test_equal_splay(mock_or_real_multi_device_context: tuple[Context, bool]) -> None:
    context, _mocked = mock_or_real_multi_device_context
    mqueue = MultiQueue.on_devices(context.devices)
    arr = numpy.arange(101)
    arr_dev = MultiArray.from_host(mqueue, arr, splay=MultiArray.EqualSplay())

    device0, device1 = context.devices

    assert (arr_dev.subarrays[device0].get(mqueue.queues[device0]) == arr[:51]).all()
    assert (arr_dev.subarrays[device1].get(mqueue.queues[device1]) == arr[51:]).all()

    # Check that the default splay is EqualSplay
    arr_dev = MultiArray.from_host(mqueue, arr)
    assert (arr_dev.subarrays[device0].get(mqueue.queues[device0]) == arr[:51]).all()
    assert (arr_dev.subarrays[device1].get(mqueue.queues[device1]) == arr[51:]).all()

    message = "The number of devices to splay to cannot be greater than the outer array dimension"
    with pytest.raises(ValueError, match=message):
        MultiArray.from_host(
            mqueue, numpy.arange(101).reshape(1, 101), splay=MultiArray.EqualSplay()
        )


def test_clone_splay(mock_or_real_multi_device_context: tuple[Context, bool]) -> None:
    context, _mocked = mock_or_real_multi_device_context
    mqueue = MultiQueue.on_devices(context.devices)
    arr = numpy.arange(101)
    arr_dev = MultiArray.from_host(mqueue, arr, splay=MultiArray.CloneSplay())
    device0, device1 = context.devices
    assert (arr_dev.subarrays[device0].get(mqueue.queues[device0]) == arr).all()
    assert (arr_dev.subarrays[device1].get(mqueue.queues[device1]) == arr).all()


def test_custom_allocator(mock_context: Context) -> None:
    context = mock_context
    queue = Queue(context.device)
    allocated = []

    def allocator(device: BoundDevice, size: int) -> Buffer:
        allocated.append(size)
        return Buffer.allocate(device, size)

    arr_dev = Array.empty(context.device, [100], numpy.int32, allocator=allocator)
    arr = arr_dev.get(queue)
    assert arr.shape == (100,)
    assert arr.dtype == numpy.int32
    assert allocated == [arr.size * arr.dtype.itemsize]


def test_custom_buffer(mock_context: Context) -> None:
    context = mock_context
    queue = Queue(context.device)

    arr = numpy.arange(100).astype(numpy.int32)
    metadata = ArrayMetadata.from_arraylike(arr)

    data = Buffer.allocate(context.device, 100)
    message = re.escape(
        "The buffer size required by the given metadata (400) "
        "is larger than the given buffer size (100)"
    )
    with pytest.raises(ValueError, match=message):
        Array(metadata, data)

    bigger_data = Buffer.allocate(context.device, arr.size * arr.dtype.itemsize)
    bigger_data.set(queue, arr)

    arr_dev = Array(metadata, bigger_data)
    res = arr_dev.get(queue)
    assert (res == arr).all()


def test_minimum_subregion(mock_context: Context) -> None:
    context = mock_context
    arr = Array.empty(context.device, (5, 6), numpy.int32)
    arr_view = arr[1:4]
    assert arr.data == arr_view.data

    arr_view_min = arr_view.minimum_subregion()
    assert arr_view_min.metadata.first_element_offset == 0
    assert arr_view_min.metadata.buffer_size == 3 * 6 * 4
    assert arr_view_min.data.size == 3 * 6 * 4


def test_set_checks_shape(mock_context: Context) -> None:
    context = mock_context
    queue = Queue(context.device)
    arr = Array.empty(context.device, (10, 20), numpy.int32)

    with pytest.raises(ValueError, match="Shape mismatch: expected \\(10, 20\\), got \\(10, 30\\)"):
        arr.set(queue, numpy.zeros((10, 30), numpy.int32))

    with pytest.raises(ValueError, match="Dtype mismatch: expected int32, got int64"):
        arr.set(queue, numpy.zeros((10, 20), numpy.int64))


def test_set_from_wrong_type(mock_context: Context) -> None:
    context = mock_context
    queue = Queue(context.device)
    arr = Array.empty(context.device, (10, 20), numpy.int32)
    with pytest.raises(TypeError, match="Cannot set from an object of type <class 'int'>"):
        arr.set(queue, 1)  # type: ignore[arg-type]
