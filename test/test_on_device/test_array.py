import numpy
import pytest

from grunnur import Array, Buffer, Queue


def _test_single_device(context):
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

    # sync set
    arr2 = numpy.arange(100) + 1
    arr_dev.set(arr2, no_async=True)
    assert (arr_dev.get() == arr2).all()


def test_single_device(context):
    _test_single_device(context=context)


def _test_multi_device(context):
    arr = numpy.arange(100)
    queue = Queue.on_all_devices(context)
    arr_dev = Array.from_host(queue, arr)

    with pytest.raises(ValueError, match="The device number must be one of those present in the queue"):
        arr_dev.single_device_view(len(context.devices))

    arr_dev0 = arr_dev.single_device_view(0)[:50]
    arr_dev1 = arr_dev.single_device_view(1)[50:]

    assert (arr_dev0.get() == arr[:50]).all()
    assert (arr_dev1.get() == arr[50:]).all()
