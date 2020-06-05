import pytest

from grunnur import API, Platform, Device, Context, OPENCL_API_ID

from .utils import mock_backend


def test_from_devices(monkeypatch):
    mock_backend(monkeypatch, OPENCL_API_ID, [
        ('Platform1', ['Device1']),
        ('Platform2', ['Device2', 'Device3'])])

    api = API.from_api_id(OPENCL_API_ID)

    platform = api[1]
    devices = platform[:]
    context = Context.from_devices(devices)
    assert context.platform == platform
    assert context.devices == devices

    with pytest.raises(ValueError, match="All devices must belong to the same platform"):
        Context.from_devices([api[0][0], api[1][0]])


def test_from_backend_devices(monkeypatch):
    backend = mock_backend(monkeypatch, OPENCL_API_ID, [
        ('Platform1', ['Device1']),
        ('Platform2', ['Device2', 'Device3'])])

    backend_devices = backend.pyopencl.get_platforms()[1].get_devices()
    context = Context.from_backend_devices(backend_devices)

    assert context.platform.name == 'Platform2'
    assert [device.name for device in context.devices] == ['Device2', 'Device3']


def test_from_backend_contexts(monkeypatch):
    backend = mock_backend(monkeypatch, OPENCL_API_ID, [
        ('Platform1', ['Device1']),
        ('Platform2', ['Device2', 'Device3'])])

    backend_devices = backend.pyopencl.get_platforms()[1].get_devices()
    backend_context = backend.pyopencl.Context(backend_devices)

    context = Context.from_backend_contexts(backend_context)

    assert context.platform.name == 'Platform2'
    assert [device.name for device in context.devices] == ['Device2', 'Device3']

    with pytest.raises(TypeError):
        Context.from_backend_contexts(1)


def test_from_criteria(monkeypatch):
    mock_backend(monkeypatch, OPENCL_API_ID, [
        ('foo-bar', ['Device1']),
        ('bar-baz', ['Device2']),
        ('foo-baz', [
            'foo-bar', 'foo-baz-1', 'bar-baz', 'foo-baz-1',
            'foo-baz-2', 'foo-baz-2', 'foo-baz-3'])])
    api = API.from_api_id(OPENCL_API_ID)
    context = Context.from_criteria(
        api, devices_num=2,
        platform_include_masks=['foo'], platform_exclude_masks=['bar'],
        device_include_masks=['foo'], device_exclude_masks=['bar'],
        unique_devices_only=True)

    assert context.platform.name == 'foo-baz'
    assert [device.name for device in context.devices] == ['foo-baz-1', 'foo-baz-2']
