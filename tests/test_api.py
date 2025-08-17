import pytest

from grunnur import API, all_api_ids, cuda_api_id, opencl_api_id
from grunnur._testing import MockBackendFactory, MockPyCUDA, MockPyOpenCL


def test_all(mock_backend_factory: MockBackendFactory) -> None:
    api_id = all_api_ids()[0]
    mock_backend_factory.mock(api_id)

    apis = API.all_available()
    assert len(apis) == 1
    assert apis[0].id == api_id


def test_all_by_shortcut(mock_backend: MockPyCUDA | MockPyOpenCL) -> None:
    api_id = mock_backend.api_id
    mock_backend.add_devices(["Device1", "Device2"])

    apis = API.all_by_shortcut(api_id.shortcut)
    assert len(apis) == 1
    assert apis[0].id == api_id


def test_all_by_shortcut_none(mock_backend_factory: MockBackendFactory) -> None:
    api_ids = all_api_ids()
    for api_id in api_ids:
        backend = mock_backend_factory.mock(api_id)
        backend.add_devices(["Device1", "Device2"])

    apis = API.all_by_shortcut()
    assert len(apis) == len(api_ids)
    assert {api.id for api in apis} == set(api_ids)


@pytest.mark.usefixtures("mock_backend_factory")
def test_all_by_shortcut_not_available() -> None:
    api_id = all_api_ids()[0]
    with pytest.raises(ValueError, match="cuda API is not available"):
        API.all_by_shortcut(api_id.shortcut)


def test_all_by_shortcut_not_found() -> None:
    with pytest.raises(ValueError, match="Invalid API shortcut: something non-existent"):
        API.all_by_shortcut("something non-existent")


def test_from_api_id(mock_backend_factory: MockBackendFactory) -> None:
    for api_id in all_api_ids():
        with pytest.raises(ImportError):
            API.from_api_id(api_id)

        mock_backend_factory.mock(api_id)
        api = API.from_api_id(api_id)
        assert api.id == api_id


def test_eq(mock_backend_factory: MockBackendFactory) -> None:
    api_id0 = all_api_ids()[0]
    api_id1 = all_api_ids()[1]

    mock_backend_factory.mock(api_id0)
    mock_backend_factory.mock(api_id1)

    api0_v1 = API.from_api_id(api_id0)
    api0_v2 = API.from_api_id(api_id0)
    api1 = API.from_api_id(api_id1)

    assert api0_v1 is not api0_v2
    assert api0_v1 == api0_v2
    assert api0_v1 != api1


def test_hash(mock_backend_factory: MockBackendFactory) -> None:
    api_id0 = all_api_ids()[0]
    api_id1 = all_api_ids()[1]

    mock_backend_factory.mock(api_id0)
    mock_backend_factory.mock(api_id1)

    api0 = API.from_api_id(api_id0)
    api1 = API.from_api_id(api_id1)

    d = {api0: 0, api1: 1}
    assert d[api0] == 0
    assert d[api1] == 1


def test_getitem(mock_backend_pyopencl: MockPyOpenCL) -> None:
    api_id = mock_backend_pyopencl.api_id

    mock_backend_pyopencl.add_platform_with_devices("Platform0", ["Device0"])
    mock_backend_pyopencl.add_platform_with_devices("Platform1", ["Device1"])

    api = API.from_api_id(api_id)
    assert api.platforms[0].name == "Platform0"
    assert api.platforms[1].name == "Platform1"


def test_attributes(mock_backend: MockPyCUDA | MockPyOpenCL) -> None:
    api = API.from_api_id(mock_backend.api_id)
    assert str(mock_backend.api_id) == "id(" + api.shortcut + ")"
    assert api.id == mock_backend.api_id
    assert api.shortcut == mock_backend.api_id.shortcut
    assert str(api) == "api(" + api.shortcut + ")"


def test_cuda_shortcut(mock_backend_pycuda: MockPyCUDA) -> None:
    mock_backend_pycuda.add_devices(["Device1", "Device2"])
    cuda_api = API.cuda()
    assert cuda_api.id == cuda_api_id()


def test_opencl_shortcut(mock_backend_pyopencl: MockPyOpenCL) -> None:
    mock_backend_pyopencl.add_devices(["Device1", "Device2"])
    opencl_api = API.opencl()
    assert opencl_api.id == opencl_api_id()


def test_any_shortcut(mock_backend_factory: MockBackendFactory) -> None:
    mock_backend_factory.mock_pycuda().add_devices(["Device1", "Device2"])
    mock_backend_factory.mock_pyopencl().add_devices(["Device1", "Device2"])
    any_api = API.any()
    assert any_api.id == opencl_api_id() or any_api.id == cuda_api_id()


# `mock_backend_factory` disables all existing backends by default
@pytest.mark.usefixtures("mock_backend_factory")
def test_any_shortcut_none_available() -> None:
    with pytest.raises(
        RuntimeError, match="No APIs are available. Please install either PyCUDA or PyOpenCL"
    ):
        API.any()
