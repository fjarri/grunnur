import io

import numpy

from grunnur.api import CUDA_API_ID, OPENCL_API_ID
import grunnur.dtypes as dtypes
from grunnur.utils import wrap_in_tuple

from .mock_pycuda import MockPyCUDA
from .mock_pyopencl import MockPyOpenCL


def mock_input(monkeypatch, inputs):
    inputs_str = "".join(input_ + "\n" for input_ in inputs)
    monkeypatch.setattr('sys.stdin', io.StringIO(inputs_str))


def mock_backend_obj(monkeypatch, api_id, backend):
    if api_id == CUDA_API_ID:
        monkeypatch.setattr('grunnur.backend_cuda.pycuda_drv', backend)
    elif api_id == OPENCL_API_ID:
        monkeypatch.setattr('grunnur.backend_opencl.pyopencl', backend)
    else:
        raise ValueError(f"Unknown API ID: {api_id}")


def mock_backend(monkeypatch, api_id, *args):
    if api_id == CUDA_API_ID:
        monkeypatch.setattr('grunnur.backend_cuda.pycuda_drv', MockPyCUDA(*args))
    elif api_id == OPENCL_API_ID:
        monkeypatch.setattr('grunnur.backend_opencl.pyopencl', MockPyOpenCL(*args))
    else:
        raise ValueError(f"Unknown API ID: {api_id}")


def disable_backend(monkeypatch, api_id):
    if api_id == CUDA_API_ID:
        monkeypatch.setattr('grunnur.backend_cuda.pycuda_drv', None)
    elif api_id == OPENCL_API_ID:
        monkeypatch.setattr('grunnur.backend_opencl.pyopencl', None)
    else:
        raise ValueError(f"Unknown API ID: {api_id}")


def get_test_array(shape, dtype, strides=None, offset=0, no_zeros=False, high=None):
    shape = wrap_in_tuple(shape)
    dtype = dtypes.normalize_type(dtype)

    if offset != 0:
        raise NotImplementedError()

    if dtype.names is not None:
        result = numpy.empty(shape, dtype)
        for name in dtype.names:
            result[name] = get_test_array(shape, dtype[name], no_zeros=no_zeros, high=high)
    else:
        if dtypes.is_integer(dtype):
            low = 1 if no_zeros else 0
            if high is None:
                high = 100 # will work even with signed chars
            get_arr = lambda: numpy.random.randint(low, high, shape).astype(dtype)
        else:
            low = 0.01 if no_zeros else 0
            if high is None:
                high = 1.0
            get_arr = lambda: numpy.random.uniform(low, high, shape).astype(dtype)

        if dtypes.is_complex(dtype):
            result = get_arr() + 1j * get_arr()
        else:
            result = get_arr()

    if strides is not None:
        result = as_strided(result, result.shape, strides)

    return result
