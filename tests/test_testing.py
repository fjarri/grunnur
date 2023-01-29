import pytest

import grunnur
from grunnur import cuda_api_id, opencl_api_id
from grunnur.testing import MockBackendFactory, MockPyCUDA, MockPyOpenCL


def test_mock_backend_factory(monkeypatch):
    fac = MockBackendFactory(monkeypatch)

    backend = fac.mock(opencl_api_id())
    assert isinstance(backend, MockPyOpenCL)
    assert grunnur.adapter_opencl.pyopencl is backend.pyopencl

    fac.disable(opencl_api_id())
    assert grunnur.adapter_opencl.pyopencl is None

    backend = fac.mock(cuda_api_id())
    assert isinstance(backend, MockPyCUDA)
    assert grunnur.adapter_cuda.pycuda_driver is backend.pycuda_driver
    assert grunnur.adapter_cuda.pycuda_compiler is backend.pycuda_compiler

    fac.disable(cuda_api_id())
    assert grunnur.adapter_cuda.pycuda_driver is None
    assert grunnur.adapter_cuda.pycuda_compiler is None

    with pytest.raises(ValueError, match="Unknown API ID: foo"):
        fac.mock("foo")

    with pytest.raises(ValueError, match="Unknown API ID: foo"):
        fac.disable("foo")
