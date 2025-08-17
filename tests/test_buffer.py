import numpy
import pytest

from grunnur import API, Buffer, Context, Queue, cuda_api_id, opencl_api_id
from grunnur._adapter_opencl import OclBufferAdapter
from grunnur._testing import MockPyOpenCL


@pytest.mark.parametrize("sync", [False, True], ids=["async", "sync"])
def test_transfer(mock_or_real_context: tuple[Context, bool], *, sync: bool) -> None:
    context, mocked = mock_or_real_context

    length = 100
    dtype = numpy.dtype("int32")
    size = length * dtype.itemsize

    arr = numpy.arange(length).astype(dtype)

    buf = Buffer.allocate(context.device, size)
    assert buf.size == size
    assert buf.offset == 0

    # Just covering the existence of the attribute.
    # Hard to actually check its validity without running a kernel
    assert buf.kernel_arg is not None

    queue = Queue(context.device)

    buf.set(queue, arr, sync=sync)

    # Read the whole buffer
    res = numpy.empty_like(arr)
    buf.get(queue, res, async_=not sync)
    if not sync:
        queue.synchronize()
    assert (res == arr).all()

    # Device-to-device copy
    res = numpy.empty_like(arr)
    buf2 = Buffer.allocate(context.device, size)
    buf2.set(queue, buf, sync=sync)
    buf2.get(queue, res, async_=not sync)
    if not sync:
        queue.synchronize()
    assert (res == arr).all()


@pytest.mark.parametrize("sync", [False, True], ids=["async", "sync"])
def test_subregion(mock_or_real_context: tuple[Context, bool], *, sync: bool) -> None:
    context, mocked = mock_or_real_context

    length = 200
    dtype = numpy.dtype("int32")
    size = length * dtype.itemsize

    arr = numpy.arange(length).astype(dtype)

    buf = Buffer.allocate(context.device, size)

    queue = Queue(context.device)

    region_offset = 64
    region_length = 50

    buf_region = buf.get_sub_region(region_offset * dtype.itemsize, region_length * dtype.itemsize)
    buf.set(queue, arr)  # just to make the test harder, fill the buffer after creating a subregion

    # Read a subregion
    res = numpy.empty_like(arr)
    arr_region = arr[region_offset : region_offset + region_length]
    res_region = numpy.empty_like(arr_region)
    buf_region.get(queue, res_region, async_=not sync)
    if not sync:
        queue.synchronize()
    assert (res_region == arr_region).all()

    # Write a subregion
    res = numpy.empty_like(arr)
    arr_region = (numpy.ones(50) * region_length).astype(dtype)
    arr[region_offset : region_offset + region_length] = arr_region
    buf_region.set(queue, arr_region, sync=sync)
    buf.get(queue, res, async_=not sync)
    if not sync:
        queue.synchronize()
    assert (res == arr).all()


@pytest.mark.parametrize("sync", [False, True], ids=["async", "sync"])
def test_subregion_copy(mock_or_real_context: tuple[Context, bool], *, sync: bool) -> None:
    context, mocked = mock_or_real_context

    length = 100
    dtype = numpy.dtype("int32")
    size = length * dtype.itemsize

    arr = numpy.arange(length).astype(dtype)

    buf = Buffer.allocate(context.device, size)

    queue = Queue(context.device)
    buf.set(queue, arr, sync=sync)

    region_offset = 64
    region_length = 100

    # Device-to-device copy
    buf2 = Buffer.allocate(context.device, size * 2)
    buf2.set(queue, numpy.ones(length * 2, dtype))
    buf2_view = buf2.get_sub_region(region_offset * dtype.itemsize, region_length * dtype.itemsize)
    buf2_view.set(queue, buf, sync=sync)
    res2 = numpy.empty(length * 2, dtype)
    buf2.get(queue, res2, async_=not sync)
    if not sync:
        queue.synchronize()
    assert (res2[region_offset : region_offset + region_length] == arr).all()
    assert (res2[:region_offset] == 1).all()
    assert (res2[region_offset + region_length :] == 1).all()


def test_subregion_of_subregion(mock_or_real_context: tuple[Context, bool]) -> None:
    context, mocked = mock_or_real_context

    length = 200
    dtype = numpy.dtype("int32")
    size = length * dtype.itemsize

    r1_offset = 64
    r1_length = 50

    r2_offset = 20
    r2_length = 20

    buf = Buffer.allocate(context.device, size)
    buf_region = buf.get_sub_region(r1_offset * dtype.itemsize, r1_length * dtype.itemsize)

    if context.api.id == opencl_api_id():
        # Check that our mock can detect taking a sub-region of a sub-region (segfault in OpenCL)
        if mocked:
            with pytest.raises(RuntimeError, match="Cannot create a subregion of subregion"):
                buf_region.get_sub_region(0, 10)
            return

        pytest.skip("Subregions of subregions are not supported in OpenCL")

    queue = Queue(context.device)

    arr = numpy.arange(length).astype(dtype)
    res = numpy.empty_like(arr)

    buf.set(queue, arr)

    arr_region2 = (numpy.ones(r2_length) * 200).astype(dtype)
    arr[r1_offset + r2_offset : r1_offset + r2_offset + r2_length] = arr_region2
    buf_region2 = buf_region.get_sub_region(r2_offset * dtype.itemsize, r2_length * dtype.itemsize)
    buf_region2.set(queue, arr_region2)
    buf.get(queue, res)
    queue.synchronize()
    assert (res == arr).all()


def test_subregion_overflow(mock_context: Context) -> None:
    buf = Buffer.allocate(mock_context.device, 100)
    with pytest.raises(
        ValueError, match="The requested subregion extends beyond the buffer length"
    ):
        buf.get_sub_region(50, 70)


def test_migrate_on_copy(monkeypatch: pytest.MonkeyPatch, mock_4_device_context: Context) -> None:
    context = mock_4_device_context
    size = 100
    src = Buffer.allocate(context.devices[0], size)
    dest = Buffer.allocate(context.devices[0], size)

    queue0 = Queue(context.devices[0])
    queue1 = Queue(context.devices[1])

    arr = numpy.arange(size).astype(numpy.uint8)

    src.set(queue0, arr)
    queue0.synchronize()

    monkeypatch.setattr(queue1, "device", queue0.device)

    if context.api.id == cuda_api_id():
        message = "Trying to access an allocation from a device different to where it was created"
    else:
        message = "Trying to access a buffer from a device different to where it was migrated to"

    with pytest.raises(RuntimeError, match=message):
        dest.set(queue1, src)


def test_flags(mock_backend_pyopencl: MockPyOpenCL) -> None:
    backend = mock_backend_pyopencl

    normal_flags = backend.pyopencl.mem_flags.READ_WRITE
    special_flags = normal_flags | backend.pyopencl.mem_flags.ALLOC_HOST_PTR

    backend.add_platform_with_devices("Apple", ["GeForce", "Foo", "Bar"])
    backend.add_platform_with_devices("Baz", ["GeForce", "Foo", "Bar"])

    # Multi-device on Apple platform with one of the devices being GeForce:
    # need special Buffer flags
    api = API.from_api_id(backend.api_id)
    context = Context.from_devices([api.platforms[0].devices[0], api.platforms[0].devices[1]])
    buf = Buffer.allocate(context.devices[0], 100)
    assert isinstance(buf._buffer_adapter, OclBufferAdapter)
    assert buf._buffer_adapter._pyopencl_buffer.flags == special_flags

    # None of the devices is GeForce
    context = Context.from_devices([api.platforms[0].devices[1], api.platforms[0].devices[2]])
    buf = Buffer.allocate(context.devices[0], 100)
    assert isinstance(buf._buffer_adapter, OclBufferAdapter)
    assert buf._buffer_adapter._pyopencl_buffer.flags == normal_flags

    # Only one device
    context = Context.from_devices([api.platforms[0].devices[0]])
    buf = Buffer.allocate(context.device, 100)
    assert isinstance(buf._buffer_adapter, OclBufferAdapter)
    assert buf._buffer_adapter._pyopencl_buffer.flags == normal_flags

    # Not an Apple platform
    context = Context.from_devices([api.platforms[1].devices[0], api.platforms[1].devices[1]])
    buf = Buffer.allocate(context.devices[0], 100)
    assert isinstance(buf._buffer_adapter, OclBufferAdapter)
    assert buf._buffer_adapter._pyopencl_buffer.flags == normal_flags


def test_set_from_wrong_type(mock_context: Context) -> None:
    buf = Buffer.allocate(mock_context.device, 100)
    queue = Queue(mock_context.device)
    with pytest.raises(TypeError, match="Cannot set from an object of type <class 'int'>"):
        buf.set(queue, 1)  # type: ignore[arg-type]


def test_mismatched_devices(mock_4_device_context: Context) -> None:
    context = mock_4_device_context
    buf = Buffer.allocate(context.devices[0], 100)
    queue = Queue(context.devices[1])
    arr = numpy.ones(100, numpy.uint8)

    with pytest.raises(ValueError, match="Mismatched devices: queue on device"):
        buf.get(queue, arr)

    with pytest.raises(ValueError, match="Mismatched devices: queue on device"):
        buf.set(queue, arr)
