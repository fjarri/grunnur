import numpy
import pytest

from grunnur import API, Context, Buffer, Queue, cuda_api_id


def test_allocate_and_copy(mock_or_real_context):

    context, _mocked = mock_or_real_context

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

    queue = Queue.on_all_devices(context)
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

    # Subregion of subregion
    if context.api.id == cuda_api_id():
        # In OpenCL that leads to segfault, but with CUDA we just emulate that with pointers.
        arr_region2 = (numpy.ones(20) * 200).astype(dtype)
        arr[25+20:25+40] = arr_region2
        buf_region2 = buf_region.get_sub_region(20 * dtype.itemsize, 20 * dtype.itemsize)
        buf_region2.set(queue, arr_region2)
        buf.get(queue, res)
        queue.synchronize()
        assert (res == arr).all()

    # Device-to-device copy
    buf2 = Buffer.allocate(context, size * 2)
    buf2.set(queue, numpy.ones(length * 2, dtype))
    buf2_view = buf2.get_sub_region(50 * dtype.itemsize, 100 * dtype.itemsize)
    buf2_view.set(queue, buf)
    res2 = numpy.empty(length * 2, dtype)
    buf2.get(queue, res2)
    queue.synchronize()
    assert (res2[50:150] == arr).all()
    assert (res2[:50] == 1).all()
    assert (res2[150:] == 1).all()

    # Device-to-device copy (no_async)
    buf2 = Buffer.allocate(context, size * 2)
    buf2.set(queue, numpy.ones(length * 2, dtype))
    buf2_view = buf2.get_sub_region(50 * dtype.itemsize, 100 * dtype.itemsize)
    buf2_view.set(queue, buf, no_async=True)
    res2 = numpy.empty(length * 2, dtype)
    buf2.get(queue, res2)
    queue.synchronize()
    assert (res2[50:150] == arr).all()
    assert (res2[:50] == 1).all()
    assert (res2[150:] == 1).all()


def test_migrate(mock_or_real_multi_device_context):

    context, _mocked = mock_or_real_multi_device_context

    length = 100
    dtype = numpy.dtype('int32')
    size = length * dtype.itemsize

    arr = numpy.arange(length).astype(dtype)

    buf = Buffer.allocate(context, size)
    assert buf.size == size
    assert buf.offset == 0

    queue0 = Queue.on_device_idxs(context, [0])
    queue1 = Queue.on_device_idxs(context, [1])

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


def test_flags(mock_backend_pyopencl):
    backend = mock_backend_pyopencl

    normal_flags = backend.pyopencl.mem_flags.READ_WRITE
    special_flags = normal_flags | backend.pyopencl.mem_flags.ALLOC_HOST_PTR

    backend.add_platform_with_devices('Apple', ['GeForce', 'Foo', 'Bar'])
    backend.add_platform_with_devices('Baz', ['GeForce', 'Foo', 'Bar'])

    # Multi-device on Apple platform with one of the devices being GeForce: need special Buffer flags
    api = API.from_api_id(backend.api_id)
    context = Context.from_devices([api.platforms[0].devices[0], api.platforms[0].devices[1]])
    buf = Buffer.allocate(context, 100)
    assert buf._buffer_adapter.pyopencl_buffer.flags == special_flags

    # None of the devices is GeForce
    context = Context.from_devices([api.platforms[0].devices[1], api.platforms[0].devices[2]])
    buf = Buffer.allocate(context, 100)
    assert buf._buffer_adapter.pyopencl_buffer.flags == normal_flags

    # Only one device
    context = Context.from_devices([api.platforms[0].devices[0]])
    buf = Buffer.allocate(context, 100)
    assert buf._buffer_adapter.pyopencl_buffer.flags == normal_flags

    # Not an Apple platform
    context = Context.from_devices([api.platforms[1].devices[0], api.platforms[1].devices[1]])
    buf = Buffer.allocate(context, 100)
    assert buf._buffer_adapter.pyopencl_buffer.flags == normal_flags


def test_set_from_wrong_type(mock_context):
    buf = Buffer.allocate(mock_context, 100)
    queue = Queue.on_all_devices(mock_context)
    with pytest.raises(TypeError, match="Cannot set from an object of type <class 'int'>"):
        buf.set(queue, 1)
