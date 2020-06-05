import pytest

from grunnur import API, Platform, Device, OPENCL_API_ID

from .utils import mock_backend


def test_all(monkeypatch):
    mock_backend(monkeypatch, OPENCL_API_ID, [
        ('Platform1', ['Device1']),
        ('Platform2', ['Device2', 'Device3'])])
    api = API.from_api_id(OPENCL_API_ID)
    platform = Platform.from_index(api, 1)
    devices = Device.all(platform)
    device_names = {device.name for device in devices}
    assert device_names == {'Device2', 'Device3'}


@pytest.mark.parametrize('unique_only', [False, True], ids=['all', 'unique'])
def test_all_by_masks(monkeypatch, unique_only):
    mock_backend(monkeypatch, OPENCL_API_ID, [
        ('Platform1', ['Device1']),
        ('Platform2', ['foo-bar', 'foo-baz', 'bar-baz', 'foo-baz'])])
    api = API.from_api_id(OPENCL_API_ID)
    platform = Platform.from_index(api, 1)
    devices = Device.all_by_masks(
        platform, include_masks=['foo'], exclude_masks=['bar'], unique_only=unique_only)
    if unique_only:
        assert len(devices) == 1
        assert devices[0].name == 'foo-baz'
    else:
        assert len(devices) == 2
        assert devices[0].name == 'foo-baz'
        assert devices[1].name == 'foo-baz'
        assert devices[0] != devices[1]


def test_from_backend_device(monkeypatch, api_id):
    backend = mock_backend(monkeypatch, OPENCL_API_ID, [
        ('Platform1', ['Device1']),
        ('Platform2', ['Device2'])])
    api = API.from_api_id(OPENCL_API_ID)

    backend_device = backend.pyopencl.get_platforms()[0].get_devices()[0]

    with pytest.raises(TypeError, match="was not recognized as a device object"):
        Device.from_backend_device(1)

    device = Device.from_backend_device(backend_device)
    assert device.platform.api == api
    assert device.platform.name == 'Platform1'
    assert device.name == 'Device1'


def test_from_index(monkeypatch):
    mock_backend(monkeypatch, OPENCL_API_ID, [
        ('Platform1', ['Device1']),
        ('Platform2', ['Device2', 'Device3'])])
    api = API.from_api_id(OPENCL_API_ID)
    platform = Platform.from_index(api, 1)
    device = Device.from_index(platform, 1)
    assert device.name == 'Device3'


def test_params(monkeypatch):
    mock_backend(monkeypatch, OPENCL_API_ID, [('Platform1', ['Device1'])])
    api = API.from_api_id(OPENCL_API_ID)
    platform = Platform.from_index(api, 0)
    device = Device.from_index(platform, 0)

    params1 = device.params
    params2 = device.params
    assert params1 is params2 # check caching


