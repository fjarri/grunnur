import pytest

from grunnur.api import all_api_ids
from grunnur import API


def test_all(mock_backend_factory):
    api_id = all_api_ids()[0]
    mock_backend_factory.mock(api_id)

    apis = API.all()
    assert len(apis) == 1
    assert apis[0].id == api_id


def test_all_by_shortcut(mock_backend):
    api_id = mock_backend.api_id
    mock_backend.add_devices(['Device1', 'Device2'])

    apis = API.all_by_shortcut(api_id.shortcut)
    assert len(apis) == 1
    assert apis[0].id == api_id


def test_all_by_shortcut_none(mock_backend_factory):
    api_ids = all_api_ids()
    for api_id in api_ids:
        backend = mock_backend_factory.mock(api_id)
        backend.add_devices(['Device1', 'Device2'])

    apis = API.all_by_shortcut()
    assert len(apis) == len(api_ids)
    assert set(api.id for api in apis) == set(api_ids)


def test_all_by_shortcut_not_available(mock_backend_factory):
    api_id = all_api_ids()[0]
    with pytest.raises(ValueError):
        API.all_by_shortcut(api_id.shortcut)


def test_all_by_shortcut_not_found():
    with pytest.raises(ValueError):
        API.all_by_shortcut("something non-existent")


def test_from_api_id(mock_backend_factory):
    for api_id in all_api_ids():
        with pytest.raises(ImportError):
            API.from_api_id(api_id)

        mock_backend_factory.mock(api_id)
        api = API.from_api_id(api_id)
        assert api.id == api_id
