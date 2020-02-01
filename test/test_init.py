import pytest

import grunnur
from grunnur.api_discovery import all_api_factories
from grunnur import CUDA_API_ID, OPENCL_API_ID
from grunnur.cuda import CUDA_API_FACTORY
from grunnur.opencl import OPENCL_API_FACTORY

from .utils import mock_backend, disable_backend


def test_cuda_api(monkeypatch):
    mock_backend(monkeypatch, CUDA_API_FACTORY, ['Device1', 'Device2'])
    from grunnur import cuda_api
    assert cuda_api.id == CUDA_API_ID


def test_opencl_api(monkeypatch):
    mock_backend(monkeypatch, OPENCL_API_FACTORY, [('Platform1', ['Device1', 'Device2'])])
    from grunnur import opencl_api
    assert opencl_api.id == OPENCL_API_ID


def test_any_api(monkeypatch):
    mock_backend(monkeypatch, CUDA_API_FACTORY, ['Device1', 'Device2'])
    mock_backend(monkeypatch, OPENCL_API_FACTORY, [('Platform1', ['Device1', 'Device2'])])
    from grunnur import any_api
    assert any_api.id == OPENCL_API_ID or any_api.id == CUDA_API_ID


def test_any_api_none_available(monkeypatch):
    for api_factory in all_api_factories():
        disable_backend(monkeypatch, api_factory)
    with pytest.raises(ImportError):
        from grunnur import any_api


def test_grunnur_import_error():
    # Checks the error branch in `grunnur.__getattr__()`
    with pytest.raises(ImportError):
        from grunnur import non_existant
