import pytest

from grunnur.api import all_api_ids
from grunnur import API

from .utils import mock_backend, mock_backend_obj, disable_backend


def test_all(monkeypatch):
    api_id = all_api_ids()[0]

    for i, api_id in enumerate(all_api_ids()):
        if i == 0:
            mock_backend(monkeypatch, api_id, [('Platform1', ['Device1', 'Device2'])])
        else:
            disable_backend(monkeypatch, api_id)

    apis = API.all()
    assert len(apis) == 1
    assert apis[0].id == all_api_ids()[0]


@pytest.mark.parametrize('api_id_to_mock', all_api_ids(), ids=str)
def test_all_by_shortcut(monkeypatch, api_id_to_mock):
    api_id = api_id_to_mock
    mock_backend(monkeypatch, api_id, [('Platform1', ['Device1', 'Device2'])])
    apis = API.all_by_shortcut(api_id.shortcut)
    assert len(apis) == 1
    assert apis[0].id == api_id


def test_all_by_shortcut_none(monkeypatch):
    api_ids = all_api_ids()
    for api_id in api_ids:
        mock_backend(monkeypatch, api_id, [('Platform1', ['Device1', 'Device2'])])
    apis = API.all_by_shortcut()
    assert len(apis) == len(api_ids)
    assert set(api.id for api in apis) == set(api_ids)


def test_all_by_shortcut_not_available(monkeypatch):
    api_id = all_api_ids()[0]
    disable_backend(monkeypatch, api_id)
    with pytest.raises(ValueError):
        API.all_by_shortcut(api_id.shortcut)


def test_all_by_shortcut_not_found():
    with pytest.raises(ValueError):
        API.all_by_shortcut("something non-existent")
