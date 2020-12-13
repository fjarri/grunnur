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

    queue = Queue(context)
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


def test_flags(mock_backend_pyopencl):
    backend = mock_backend_pyopencl

    normal_flags = backend.pyopencl.mem_flags.READ_WRITE
    special_flags = normal_flags | backend.pyopencl.mem_flags.ALLOC_HOST_PTR

    backend.add_platform_with_devices('Apple', ['GeForce', 'Foo', 'Bar'])
    backend.add_platform_with_devices('Baz', ['GeForce', 'Foo', 'Bar'])

    # Multi-device on Apple platform with one of the devices being GeForce: need special Buffer flags
    api = API.from_api_id(backend.api_id)
    context = Context.from_devices([api.platforms[0].devices[0], api.platforms[0].devices[1]])
    buf = Buffer.allocate(context, 100, device_idx=0)
    assert buf._buffer_adapter.pyopencl_buffer.flags == special_flags

    # None of the devices is GeForce
    context = Context.from_devices([api.platforms[0].devices[1], api.platforms[0].devices[2]])
    buf = Buffer.allocate(context, 100, device_idx=0)
    assert buf._buffer_adapter.pyopencl_buffer.flags == normal_flags

    # Only one device
    context = Context.from_devices([api.platforms[0].devices[0]])
    buf = Buffer.allocate(context, 100)
    assert buf._buffer_adapter.pyopencl_buffer.flags == normal_flags

    # Not an Apple platform
    context = Context.from_devices([api.platforms[1].devices[0], api.platforms[1].devices[1]])
    buf = Buffer.allocate(context, 100, device_idx=0)
    assert buf._buffer_adapter.pyopencl_buffer.flags == normal_flags


def test_set_from_wrong_type(mock_context):
    buf = Buffer.allocate(mock_context, 100)
    queue = Queue(mock_context)
    with pytest.raises(TypeError, match="Cannot set from an object of type <class 'int'>"):
        buf.set(queue, 1)


def test_create_in_multi_device_context(mock_4_device_context):
    context = mock_4_device_context
    with pytest.raises(ValueError, match="device_idx must be specified in a multi-device context"):
        buf = Buffer.allocate(context, 100)


def test_mismatched_devices(mock_4_device_context):
    context = mock_4_device_context
    buf = Buffer.allocate(context, 100, device_idx=0)
    queue = Queue(context, device_idx=1)
    arr = numpy.ones(100, numpy.uint8)

    with pytest.raises(ValueError, match="Mismatched devices: queue on device"):
        buf.get(queue, arr)

    with pytest.raises(ValueError, match="Mismatched devices: queue on device"):
        buf.set(queue, arr)
