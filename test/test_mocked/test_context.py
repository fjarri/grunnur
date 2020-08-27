import pytest

from grunnur import API, Platform, Device, Context


def test_from_devices(mock_backend):
    mock_backend.add_devices(['Device2', 'Device3'])

    api = API.from_api_id(mock_backend.api_id)

    platform = api.platforms[0]
    devices = platform.devices[:]
    context = Context.from_devices(devices)
    assert context.platform == platform
    assert context.devices == tuple(devices)


def test_from_devices_different_platforms(mock_backend_pyopencl):
    mock_backend_pyopencl.add_platform_with_devices('Platform1', ['Device1', 'Device2'])
    mock_backend_pyopencl.add_platform_with_devices('Platform2', ['Device3', 'Device4'])

    api = API.from_api_id(mock_backend_pyopencl.api_id)

    with pytest.raises(ValueError, match="All devices must belong to the same platform"):
        Context.from_devices([api.platforms[0].devices[0], api.platforms[1].devices[0]])


def test_from_backend_devices_opencl(mock_backend_pyopencl):
    backend = mock_backend_pyopencl

    backend.add_platform_with_devices('Platform1', ['Device1'])
    backend.add_platform_with_devices('Platform2', ['Device2', 'Device3'])

    backend_devices = backend.pyopencl.get_platforms()[1].get_devices()
    context = Context.from_backend_devices(backend_devices)

    assert context.platform.name == 'Platform2'
    assert [device.name for device in context.devices] == ['Device2', 'Device3']


def test_from_backend_contexts_opencl(mock_backend_pyopencl):
    # OpenCL style - one context, many devices

    backend = mock_backend_pyopencl

    backend.add_platform_with_devices('Platform1', ['Device1'])
    backend.add_platform_with_devices('Platform2', ['Device2', 'Device3'])

    backend_devices = backend.pyopencl.get_platforms()[1].get_devices()
    backend_context = backend.pyopencl.Context(backend_devices)

    backend_context2 = backend.pyopencl.Context(backend_devices)
    with pytest.raises(ValueError, match="Cannot make one OpenCL context out of several contexts"):
        Context.from_backend_contexts([backend_context, backend_context2])

    context = Context.from_backend_contexts(backend_context)

    assert context.platform.name == 'Platform2'
    assert [device.name for device in context.devices] == ['Device2', 'Device3']

    with pytest.raises(TypeError):
        Context.from_backend_contexts(1)


@pytest.mark.parametrize('take_ownership', [False, True], ids=['no ownership', 'take ownership'])
def test_from_backend_contexts_cuda_single_device(mock_backend_pycuda, take_ownership):
    # CUDA style - a context per device

    backend = mock_backend_pycuda

    backend.add_devices(['Device1', 'Device2'])

    backend_context = backend.pycuda_driver.Device(1).make_context()

    if not take_ownership:
        # backend context can stay in the stack
        context = Context.from_backend_contexts(backend_context, take_ownership=False)

    else:
        # forgot to pop the backend context off the stack - error
        with pytest.raises(ValueError, match="The given context is already in the context stack"):
            context = Context.from_backend_contexts(backend_context, take_ownership=True)

        backend.pycuda_driver.Context.pop()
        context = Context.from_backend_contexts(backend_context, take_ownership=True)

    # CUDA has no concept of platforms, so the platform name in the mock will be ignored
    assert context.platform.name == 'nVidia CUDA'

    assert [device.name for device in context.devices] == ['Device2']


def test_from_backend_contexts_cuda_multi_device(mock_backend_pycuda):
    # CUDA style - a context per device

    backend = mock_backend_pycuda

    backend.add_devices(['Device1', 'Device2'])

    backend_context1 = backend.pycuda_driver.Device(0).make_context()
    backend.pycuda_driver.Context.pop()

    backend_context2 = backend.pycuda_driver.Device(1).make_context()
    backend.pycuda_driver.Context.pop()

    # Grunnur mast have ownership
    error_msg = "When dealing with multiple CUDA contexts, Grunnur must be the one managing them"
    with pytest.raises(ValueError, match=error_msg):
        Context.from_backend_contexts([backend_context1, backend_context2])

    context = Context.from_backend_contexts(
        [backend_context1, backend_context2], take_ownership=True)

    # CUDA has no concept of platforms, so the platform name in the mock will be ignored
    assert context.platform.name == 'nVidia CUDA'

    assert [device.name for device in context.devices] == ['Device1', 'Device2']


def test_from_criteria(mock_backend_pyopencl):

    backend = mock_backend_pyopencl

    backend.add_platform_with_devices('foo-bar', ['Device1'])
    backend.add_platform_with_devices('bar-baz', ['Device2'])
    backend.add_platform_with_devices(
        'foo-baz',
        [
            'foo-bar', 'foo-baz-1', 'bar-baz', 'foo-baz-1',
            'foo-baz-2', 'foo-baz-2', 'foo-baz-3'])

    api = API.from_api_id(backend.api_id)
    context = Context.from_criteria(
        api, devices_num=2,
        platform_include_masks=['foo'], platform_exclude_masks=['bar'],
        device_include_masks=['foo'], device_exclude_masks=['bar'],
        unique_devices_only=True)

    assert context.platform.name == 'foo-baz'
    assert [device.name for device in context.devices] == ['foo-baz-1', 'foo-baz-2']
