import io

from grunnur import CUDA_API_ID, OPENCL_API_ID

from .mock_pycuda import MockPyCUDA
from .mock_pyopencl import MockPyOpenCL


def mock_input(monkeypatch, inputs):
    inputs_str = "".join(input_ + "\n" for input_ in inputs)
    monkeypatch.setattr('sys.stdin', io.StringIO(inputs_str))


def mock_backend_obj(monkeypatch, api_factory, backend):
    if api_factory.api_id == CUDA_API_ID:
        monkeypatch.setattr('grunnur.cuda.pycuda_drv', backend)
    elif api_factory.api_id == OPENCL_API_ID:
        monkeypatch.setattr('grunnur.opencl.pyopencl', backend)
    else:
        raise ValueError(f"Unknown API ID: {api_factory.api_id}")


def mock_backend(monkeypatch, api_factory, *args):
    if api_factory.api_id == CUDA_API_ID:
        monkeypatch.setattr('grunnur.cuda.pycuda_drv', MockPyCUDA(*args))
    elif api_factory.api_id == OPENCL_API_ID:
        monkeypatch.setattr('grunnur.opencl.pyopencl', MockPyOpenCL(*args))
    else:
        raise ValueError(f"Unknown API ID: {api_factory.api_id}")


def disable_backend(monkeypatch, api_factory):
    if api_factory.api_id == CUDA_API_ID:
        monkeypatch.setattr('grunnur.cuda.pycuda_drv', None)
    elif api_factory.api_id == OPENCL_API_ID:
        monkeypatch.setattr('grunnur.opencl.pyopencl', None)
    else:
        raise ValueError(f"Unknown API ID: {api_factory.api_id}")
