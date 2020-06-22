import pytest

from grunnur import API, Platform, OPENCL_API_ID


def test_all(mock_backend_pyopencl):
    mock_backend_pyopencl.add_platform_with_devices('Platform1', ['Device1'])
    mock_backend_pyopencl.add_platform_with_devices('Platform2', ['Device2'])
    api = API.from_api_id(mock_backend_pyopencl.api_id)
    platforms = Platform.all(api)
    platform_names = {platform.name for platform in platforms}
    assert platform_names == {'Platform1', 'Platform2'}


def test_all_by_masks(mock_backend_pyopencl):
    mock_backend_pyopencl.add_platform_with_devices('foo-bar', ['Device1'])
    mock_backend_pyopencl.add_platform_with_devices('bar-baz', ['Device2'])
    mock_backend_pyopencl.add_platform_with_devices('foo-baz', ['Device3'])
    api = API.from_api_id(mock_backend_pyopencl.api_id)
    platforms = Platform.all_by_masks(api, include_masks=['foo'], exclude_masks=['bar'])
    assert len(platforms) == 1
    assert platforms[0].name == 'foo-baz'


def test_from_backend_platform(mock_backend_pyopencl):
    mock_backend_pyopencl.add_platform_with_devices('Platform1', ['Device1'])
    mock_backend_pyopencl.add_platform_with_devices('Platform2', ['Device2'])
    api = API.from_api_id(mock_backend_pyopencl.api_id)

    backend_platform = mock_backend_pyopencl.pyopencl.get_platforms()[0]

    with pytest.raises(TypeError, match="was not recognized as a platform object"):
        Platform.from_backend_platform(1)

    platform = Platform.from_backend_platform(backend_platform)
    assert platform.api == api
    assert platform.name == 'Platform1'


def test_from_index(mock_backend_pyopencl):
    mock_backend_pyopencl.add_platform_with_devices('Platform1', ['Device1'])
    mock_backend_pyopencl.add_platform_with_devices('Platform2', ['Device2'])
    api = API.from_api_id(mock_backend_pyopencl.api_id)

    platform = Platform.from_index(api, 1)
    assert platform.name == 'Platform2'


def test_eq(mock_backend_pyopencl):
    mock_backend_pyopencl.add_platform_with_devices('Platform0', ['Device0'])
    mock_backend_pyopencl.add_platform_with_devices('Platform1', ['Device1'])
    api = API.from_api_id(mock_backend_pyopencl.api_id)

    p0_v1 = Platform.from_index(api, 0)
    p0_v2 = Platform.from_index(api, 0)
    p1 = Platform.from_index(api, 1)

    assert p0_v1 is not p0_v2 and p0_v1 == p0_v2
    assert p0_v1 != p1


def test_hash(mock_backend_pyopencl):
    mock_backend_pyopencl.add_platform_with_devices('Platform0', ['Device0'])
    mock_backend_pyopencl.add_platform_with_devices('Platform1', ['Device1'])
    api = API.from_api_id(mock_backend_pyopencl.api_id)

    p0 = Platform.from_index(api, 0)
    p1 = Platform.from_index(api, 1)

    d = {p0: 0, p1: 1}
    assert d[p0] == 0
    assert d[p1] == 1


def test_getitem(mock_backend_pyopencl):
    mock_backend_pyopencl.add_platform_with_devices('Platform0', ['Device0'])
    mock_backend_pyopencl.add_platform_with_devices('Platform1', ['Device1', 'Device2'])
    api = API.from_api_id(mock_backend_pyopencl.api_id)

    p1 = Platform.from_index(api, 1)

    assert p1[0].name == 'Device1'
    assert p1[1].name == 'Device2'
