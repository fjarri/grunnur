import pytest

from grunnur.api_discovery import all_api_factories
from grunnur import available_apis, find_apis

from .utils import mock_backend, mock_backend_obj


def test_available_apis(monkeypatch):
    api_factory = all_api_factories()[0]
    mock_backend(monkeypatch, api_factory, [('Platform1', ['Device1', 'Device2'])])
    apis = available_apis()
    assert len(apis) > 0
    assert any(api.id == api_factory.api_id for api in apis)


def test_find_apis_specific(monkeypatch, api_factory):
    mock_backend(monkeypatch, api_factory, [('Platform1', ['Device1', 'Device2'])])
    apis = find_apis(api_factory.api_id.shortcut)
    assert len(apis) == 1
    assert apis[0].id == api_factory.api_id


def test_find_apis_unknown_shortcut(monkeypatch):
    api_factory = all_api_factories()[0]
    # Forcing the API to become unavailable
    mock_backend_obj(monkeypatch, api_factory, None)
    with pytest.raises(ValueError):
        find_apis(api_factory.api_id.shortcut)


def test_find_apis_unavailable_api():
    with pytest.raises(ValueError):
        find_apis("something nonexistent")


def test_find_apis_any(monkeypatch):
    api_factory = all_api_factories()[0]
    mock_backend(monkeypatch, api_factory, [('Platform1', ['Device1', 'Device2'])])
    apis = find_apis(None)
    assert len(apis) > 0
    assert any(api.id == api_factory.api_id for api in apis)
