import pytest

from grunnur.api import all_api_ids
from grunnur import API


def test_all(mock_backend_factory):
    api_id = all_api_ids()[0]
    mock_backend_factory.mock(api_id)

    apis = API.all_available()
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


def test_eq(mock_backend_factory):
    api_id0 = all_api_ids()[0]
    api_id1 = all_api_ids()[1]

    mock_backend_factory.mock(api_id0)
    mock_backend_factory.mock(api_id1)

    api0_v1 = API.from_api_id(api_id0)
    api0_v2 = API.from_api_id(api_id0)
    api1 = API.from_api_id(api_id1)

    assert api0_v1 is not api0_v2 and api0_v1 == api0_v2
    assert api0_v1 != api1


def test_hash(mock_backend_factory):
    api_id0 = all_api_ids()[0]
    api_id1 = all_api_ids()[1]

    mock_backend_factory.mock(api_id0)
    mock_backend_factory.mock(api_id1)

    api0 = API.from_api_id(api_id0)
    api1 = API.from_api_id(api_id1)

    d = {api0: 0, api1: 1}
    assert d[api0] == 0
    assert d[api1] == 1


def test_getitem(mock_backend_pyopencl):
    api_id = mock_backend_pyopencl.api_id

    mock_backend_pyopencl.add_platform_with_devices('Platform0', ['Device0'])
    mock_backend_pyopencl.add_platform_with_devices('Platform1', ['Device1'])

    api = API.from_api_id(api_id)
    assert api.platforms[0].name == 'Platform0'
    assert api.platforms[1].name == 'Platform1'


def test_attributes(mock_backend):
    api = API.from_api_id(mock_backend.api_id)
    assert str(mock_backend.api_id) == 'id(' + api.shortcut + ')'
    assert api.id == mock_backend.api_id
    assert api.shortcut == mock_backend.api_id.shortcut
    assert str(api) == 'api(' + api.shortcut + ')'
