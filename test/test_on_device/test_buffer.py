import numpy
import pytest

from grunnur import Buffer, Queue


def _test_allocate(context):
    length = 100
    dtype = numpy.dtype('int32')
    size = length * dtype.itemsize

    arr = numpy.arange(length).astype(dtype)

    buf = Buffer.allocate(context, size)
    assert buf.size == size
    assert buf.offset == 0

    # Just covering the existence of the attribute.
    # Hard to actually check it without running a kernel
    assert buf.kernel_arg is not None

    queue = Queue.from_device_idxs(context)
    buf.set(queue, arr)

    # Read the whole buffer
    res = numpy.empty_like(arr)
    buf.get(queue, res)
    queue.synchronize()
    assert (res == arr).all()

    # Read a subregion
    buf_region = buf.get_sub_region(25 * dtype.itemsize, 50 * dtype.itemsize)
    arr_region = arr[25:25+50]
    res_region = numpy.empty_like(arr_region)
    buf_region.get(queue, res_region)
    queue.synchronize()
    assert (res_region == arr_region).all()

    # Write a subregion
    arr_region = (numpy.ones(50) * 100).astype(dtype)
    arr[25:25+50] = arr_region
    buf_region.set(queue, arr_region)
    buf.get(queue, res)
    queue.synchronize()
    assert (res == arr).all()


def test_allocate(context):
    _test_allocate(context=context)


def _test_migrate(context):

    length = 100
    dtype = numpy.dtype('int32')
    size = length * dtype.itemsize

    arr = numpy.arange(length).astype(dtype)

    buf = Buffer.allocate(context, size)
    assert buf.size == size
    assert buf.offset == 0

    queue0 = Queue.from_device_idxs(context, [0])
    queue1 = Queue.from_device_idxs(context, [1])

    res = numpy.empty_like(arr)

    with pytest.raises(RuntimeError, match="This buffer has not been bound to any device yet"):
        buf.set(queue0, arr)

    with pytest.raises(RuntimeError, match="This buffer has not been bound to any device yet"):
        buf.get(queue0, arr)

    with pytest.raises(RuntimeError, match="This buffer has not been bound to any device yet"):
        buf.migrate(queue0)

    idx = len(context.devices)
    with pytest.raises(ValueError, match=f"Device index {idx} out of available range for this context"):
        buf.bind(idx)

    buf.bind(0)

    # Binding to the same device produces no effect
    buf.bind(0)

    with pytest.raises(ValueError, match=f"The buffer is already bound to device 0"):
        buf.bind(1)

    buf.set(queue0, arr)
    queue0.synchronize()

    buf1 = buf.get_sub_region(0, size)
    buf1.bind(1)
    buf1.migrate(queue1)

    buf1.get(queue1, res)
    queue1.synchronize()
    assert (res == arr).all()


def test_migrate(multi_device_context):
    _test_migrate(context=multi_device_context)
