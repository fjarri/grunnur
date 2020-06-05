import pytest

from grunnur import API, Platform, OPENCL_API_ID

from .utils import mock_backend, mock_backend_obj, disable_backend


def test_all(monkeypatch):
    mock_backend(monkeypatch, OPENCL_API_ID, [
        ('Platform1', ['Device1']),
        ('Platform2', ['Device2'])])
    api = API.from_api_id(OPENCL_API_ID)
    platforms = Platform.all(api)
    platform_names = {platform.name for platform in platforms}
    assert platform_names == {'Platform1', 'Platform2'}


def test_all_by_masks(monkeypatch):
    mock_backend(monkeypatch, OPENCL_API_ID, [
        ('foo-bar', ['Device1']),
        ('bar-baz', ['Device2']),
        ('foo-baz', ['Device3'])])
    api = API.from_api_id(OPENCL_API_ID)
    platforms = Platform.all_by_masks(api, include_masks=['foo'], exclude_masks=['bar'])
    assert len(platforms) == 1
    assert platforms[0].name == 'foo-baz'


def test_from_backend_platform(monkeypatch):
    backend = mock_backend(monkeypatch, OPENCL_API_ID, [
        ('Platform1', ['Device1']),
        ('Platform2', ['Device2'])])
    api = API.from_api_id(OPENCL_API_ID)

    backend_platform = backend.pyopencl.get_platforms()[0]
    platform = Platform.from_backend_platform(backend_platform)
    assert platform.api == api
    assert platform.name == 'Platform1'


def test_from_index(monkeypatch):
    mock_backend(monkeypatch, OPENCL_API_ID, [
        ('Platform1', ['Device1']),
        ('Platform2', ['Device2'])])
    api = API.from_api_id(OPENCL_API_ID)

    platform = Platform.from_index(api, 1)
    assert platform.name == 'Platform2'
