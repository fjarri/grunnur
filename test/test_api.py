import pytest

from grunnur.api import all_api_ids
from grunnur import API

from .utils import mock_backend, mock_backend_obj, disable_backend


def test_available_apis(monkeypatch):
    api_id = all_api_ids()[0]
    mock_backend(monkeypatch, api_id, [('Platform1', ['Device1', 'Device2'])])
    apis = API.all()
    assert len(apis) > 0
    assert any(api.id == api_id for api in apis)


def test_find_apis_specific(monkeypatch, api_id):
    mock_backend(monkeypatch, api_id, [('Platform1', ['Device1', 'Device2'])])
    apis = API.all_by_shortcut(api_id.shortcut)
    assert len(apis) == 1
    assert apis[0].id == api_id


def test_find_apis_unknown_shortcut(monkeypatch):
    api_id = all_api_ids()[0]
    disable_backend(monkeypatch, api_id)
    with pytest.raises(ValueError):
        API.all_by_shortcut(api_id.shortcut)


def test_find_apis_unavailable_api():
    with pytest.raises(ValueError):
        API.all_by_shortcut("something non-existent")


def test_find_apis_any(monkeypatch):
    api_id = all_api_ids()[0]
    mock_backend(monkeypatch, api_id, [('Platform1', ['Device1', 'Device2'])])
    apis = API.all_by_shortcut()
    assert len(apis) > 0
    assert any(api.id == api_id for api in apis)
