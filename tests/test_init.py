import pytest

import grunnur
from grunnur.api import all_api_ids, cuda_api_id, opencl_api_id


def test_import_cuda_api(mock_backend_pycuda):
    mock_backend_pycuda.add_devices(["Device1", "Device2"])
    from grunnur import cuda_api  # noqa: PLC0415

    assert cuda_api.id == cuda_api_id()


def test_import_opencl_api(mock_backend_pyopencl):
    mock_backend_pyopencl.add_devices(["Device1", "Device2"])
    from grunnur import opencl_api  # noqa: PLC0415

    assert opencl_api.id == opencl_api_id()


def test_import_any_api(mock_backend_factory):
    mock_backend_factory.mock_pycuda().add_devices(["Device1", "Device2"])
    mock_backend_factory.mock_pyopencl().add_devices(["Device1", "Device2"])
    from grunnur import any_api  # noqa: PLC0415

    assert any_api.id == opencl_api_id() or any_api.id == cuda_api_id()


# `mock_backend_factory` disables all existing backends by default
@pytest.mark.usefixtures("mock_backend_factory")
def test_any_api_none_available():
    with pytest.raises(ImportError):
        from grunnur import any_api  # noqa: PLC0415


def test_grunnur_import_error():
    # Checks the error branch in `grunnur.__getattr__()`
    with pytest.raises(ImportError):
        from grunnur import non_existent  # noqa: PLC0415
