import pytest

from grunnur import API, Context, Buffer, Queue

from ..test_on_device.test_buffer import _test_allocate_and_copy, _test_migrate


def test_allocate_and_copy(mock_context):
    _test_allocate_and_copy(context=mock_context)


def test_migrate(mock_4_device_context):
    _test_migrate(context=mock_4_device_context)


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
