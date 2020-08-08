import pytest

import grunnur
from grunnur.api import cuda_api_id, opencl_api_id, all_api_ids


def test_import_cuda_api(mock_backend_pycuda):
    mock_backend_pycuda.add_devices(['Device1', 'Device2'])
    from grunnur import cuda_api
    assert cuda_api.id == cuda_api_id()


def test_import_opencl_api(mock_backend_pyopencl):
    mock_backend_pyopencl.add_devices(['Device1', 'Device2'])
    from grunnur import opencl_api
    assert opencl_api.id == opencl_api_id()


def test_import_any_api(mock_backend_factory):
    mock_backend_factory.mock_pycuda().add_devices(['Device1', 'Device2'])
    mock_backend_factory.mock_pyopencl().add_devices(['Device1', 'Device2'])
    from grunnur import any_api
    assert any_api.id == opencl_api_id() or any_api.id == cuda_api_id()


def test_any_api_none_available(mock_backend_factory):
    with pytest.raises(ImportError):
        from grunnur import any_api


def test_grunnur_import_error():
    # Checks the error branch in `grunnur.__getattr__()`
    with pytest.raises(ImportError):
        from grunnur import non_existent


def test_sphinx_requirements():
    # These attributes are queried by Sphinx when building docs, for whatever reason.
    assert grunnur.__annotations__ == {}
    assert grunnur.__mro__ == ()
