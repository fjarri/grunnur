import pytest

from grunnur import (
    API,
    Platform,
    Device,
    Context,
    Queue,
    opencl_api_id,
    DeviceFilter,
    PlatformFilter,
)
from grunnur.context import BoundMultiDevice


def test_from_devices(mock_backend):
    mock_backend.add_devices(["Device2", "Device3"])

    api = API.from_api_id(mock_backend.api_id)

    platform = api.platforms[0]
    devices = platform.devices[:]
    context = Context.from_devices(devices)
    assert context.platform == platform
    assert [device.as_unbound() for device in context.devices] == devices


def test_from_devices_different_platforms(mock_backend_pyopencl):
    mock_backend_pyopencl.add_platform_with_devices("Platform1", ["Device1", "Device2"])
    mock_backend_pyopencl.add_platform_with_devices("Platform2", ["Device3", "Device4"])

    api = API.from_api_id(mock_backend_pyopencl.api_id)

    with pytest.raises(ValueError, match="All devices must belong to the same platform"):
        Context.from_devices([api.platforms[0].devices[0], api.platforms[1].devices[0]])


def test_from_backend_devices_opencl(mock_backend_pyopencl):
    backend = mock_backend_pyopencl

    backend.add_platform_with_devices("Platform1", ["Device1"])
    backend.add_platform_with_devices("Platform2", ["Device2", "Device3"])

    backend_devices = backend.pyopencl.get_platforms()[1].get_devices()
    context = Context.from_backend_devices(backend_devices)

    assert context.platform.name == "Platform2"
    assert [device.name for device in context.devices] == ["Device2", "Device3"]


def test_from_backend_contexts_opencl(mock_backend_pyopencl):
    # OpenCL style - one context, many devices

    backend = mock_backend_pyopencl

    backend.add_platform_with_devices("Platform1", ["Device1"])
    backend.add_platform_with_devices("Platform2", ["Device2", "Device3"])

    backend_devices = backend.pyopencl.get_platforms()[1].get_devices()
    backend_context = backend.pyopencl.Context(backend_devices)

    backend_context2 = backend.pyopencl.Context(backend_devices)
    with pytest.raises(ValueError, match="Cannot make one OpenCL context out of several contexts"):
        Context.from_backend_contexts([backend_context, backend_context2])

    context = Context.from_backend_contexts([backend_context])

    assert context.platform.name == "Platform2"
    assert [device.name for device in context.devices] == ["Device2", "Device3"]

    with pytest.raises(
        TypeError, match="<class 'int'> objects were not recognized as contexts by any API"
    ):
        Context.from_backend_contexts([1])


def test_from_backend_contexts_several_apis(mock_backend_pycuda, mock_backend_pyopencl):

    backend = mock_backend_pyopencl
    backend.add_platform_with_devices("Platform1", ["Device1"])

    backend_devices = backend.pyopencl.get_platforms()[0].get_devices()
    backend_context = backend.pyopencl.Context(backend_devices)

    # Check that when several backends are available,
    # the correct one is used for the given context objects.
    context = Context.from_backend_contexts([backend_context])
    assert context.api.id == opencl_api_id()
    assert context.platform.name == "Platform1"
    assert [device.name for device in context.devices] == ["Device1"]


@pytest.mark.parametrize("take_ownership", [False, True], ids=["no ownership", "take ownership"])
def test_from_backend_contexts_cuda_single_device(mock_backend_pycuda, take_ownership):
    # CUDA style - a context per device

    backend = mock_backend_pycuda

    backend.add_devices(["Device1", "Device2"])

    backend_context = backend.pycuda_driver.Device(1).make_context()

    if not take_ownership:
        # backend context can stay in the stack
        context = Context.from_backend_contexts([backend_context], take_ownership=False)

    else:
        # forgot to pop the backend context off the stack - error
        with pytest.raises(ValueError, match="The given context is already in the context stack"):
            context = Context.from_backend_contexts([backend_context], take_ownership=True)

        backend.pycuda_driver.Context.pop()
        context = Context.from_backend_contexts([backend_context], take_ownership=True)

    # CUDA has no concept of platforms, so the platform name in the mock will be ignored
    assert context.platform.name == "nVidia CUDA"

    assert [device.name for device in context.devices] == ["Device2"]

    if not take_ownership:
        # Clean up - we didn't take ownership of the context, so someone else has to pop it.
        backend.pycuda_driver.Context.pop()


def test_from_backend_contexts_cuda_multi_device(mock_backend_pycuda):
    # CUDA style - a context per device

    backend = mock_backend_pycuda

    backend.add_devices(["Device1", "Device2"])

    backend_context1 = backend.pycuda_driver.Device(0).make_context()
    backend.pycuda_driver.Context.pop()

    backend_context2 = backend.pycuda_driver.Device(1).make_context()
    backend.pycuda_driver.Context.pop()

    # Grunnur mast have ownership
    error_msg = "When dealing with multiple CUDA contexts, Grunnur must be the one managing them"
    with pytest.raises(ValueError, match=error_msg):
        Context.from_backend_contexts([backend_context1, backend_context2])

    context = Context.from_backend_contexts(
        [backend_context1, backend_context2], take_ownership=True
    )

    # CUDA has no concept of platforms, so the platform name in the mock will be ignored
    assert context.platform.name == "nVidia CUDA"

    assert [device.name for device in context.devices] == ["Device1", "Device2"]


def test_from_criteria(mock_backend_pyopencl):

    backend = mock_backend_pyopencl

    backend.add_platform_with_devices("foo-bar", ["Device1"])
    backend.add_platform_with_devices("bar-baz", ["Device2"])
    backend.add_platform_with_devices(
        "foo-baz",
        ["foo-bar", "foo-baz-1", "bar-baz", "foo-baz-1", "foo-baz-2", "foo-baz-2", "foo-baz-3"],
    )

    api = API.from_api_id(backend.api_id)
    context = Context.from_criteria(
        api,
        devices_num=2,
        device_filter=DeviceFilter(
            include_masks=["foo"],
            exclude_masks=["bar"],
            unique_only=True,
        ),
        platform_filter=PlatformFilter(
            include_masks=["foo"],
            exclude_masks=["bar"],
        ),
    )

    assert context.platform.name == "foo-baz"
    assert [device.name for device in context.devices] == ["foo-baz-1", "foo-baz-2"]


def test_bound_device_eq(mock_backend_pyopencl):
    mock_backend_pyopencl.add_platform_with_devices("Platform1", ["Device1", "Device2"])

    api = API.from_api_id(mock_backend_pyopencl.api_id)

    platform = api.platforms[0]
    devices = platform.devices[:]
    context = Context.from_devices(devices)

    assert context.devices[0] == context.devices[0]
    assert context.devices[0] != context.devices[1]

    context2 = Context.from_devices(devices)

    assert context2.devices[0] != context.devices[0]


def test_bound_device_str(mock_context):
    s = str(mock_context.device)
    assert mock_context.api.id.shortcut in s
    assert "0,0" in s
    assert "Context" in s


def test_bound_multi_device_creation(mock_backend_pyopencl):
    mock_backend_pyopencl.add_platform_with_devices("Platform1", ["Device1", "Device2", "Device3"])

    api = API.from_api_id(mock_backend_pyopencl.api_id)

    platform = api.platforms[0]
    devices = platform.devices[:]
    context = Context.from_devices(devices)
    context2 = Context.from_devices(devices)

    with pytest.raises(
        ValueError, match="All devices in a multi-device must belong to the same context"
    ):
        BoundMultiDevice.from_bound_devices([context.devices[0], context2.devices[1]])

    with pytest.raises(ValueError, match="All devices in a multi-device must be distinct"):
        context.devices[[1, 1]]

    sub_device = context.devices[[2, 1]]
    assert len(sub_device) == 2
    assert sub_device[0].name == "Device3"
    assert sub_device[1].name == "Device2"


def test_bound_multi_device_eq(mock_backend_pyopencl):
    mock_backend_pyopencl.add_platform_with_devices("Platform1", ["Device1", "Device2", "Device3"])

    api = API.from_api_id(mock_backend_pyopencl.api_id)
    context = Context.from_devices(api.platforms[0].devices[:])

    assert context.devices[[2, 1]] == context.devices[[2, 1]]
    assert context.devices[[2, 1]] != context.devices[[1, 2]]


def test_bound_multi_device_issubset(mock_backend_pyopencl):
    mock_backend_pyopencl.add_platform_with_devices("Platform1", ["Device1", "Device2", "Device3"])

    api = API.from_api_id(mock_backend_pyopencl.api_id)
    context = Context.from_devices(api.platforms[0].devices[:])

    assert context.devices[[2, 1]].issubset(context.devices)


def test_device_shortcut(mock_backend_pyopencl):
    mock_backend_pyopencl.add_platform_with_devices("Platform1", ["Device1", "Device2", "Device3"])

    api = API.from_api_id(mock_backend_pyopencl.api_id)
    context = Context.from_devices(api.platforms[0].devices[:])

    with pytest.raises(
        RuntimeError, match="The `device` shortcut only works for single-device contexts"
    ):
        context.device

    context = Context.from_devices([api.platforms[0].devices[2]])
    assert context.device.name == "Device3"


def test_deactivate(mock_backend_pyopencl, mock_backend_pycuda):

    mock_backend_pyopencl.add_platform_with_devices("Platform1", ["Device1"])
    mock_backend_pycuda.add_devices(["Device1"])

    api = API.from_api_id(mock_backend_pyopencl.api_id)
    context = Context.from_devices([api.platforms[0].devices[0]])
    # Does nothing in OpenCL, but we can still call this for the sake of being generic
    context.deactivate()

    backend_context = mock_backend_pycuda.pycuda_driver.Device(0).make_context()
    backend_context.pop()

    api = API.from_api_id(mock_backend_pycuda.api_id)
    context = Context.from_backend_contexts([backend_context], take_ownership=True)
    assert backend_context.is_stacked()
    context.deactivate()
    assert not backend_context.is_stacked()
