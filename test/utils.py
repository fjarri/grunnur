import io

import numpy

from grunnur import API, CUDA_API_ID, OPENCL_API_ID
from grunnur.device import select_devices
import grunnur.dtypes as dtypes
from grunnur.utils import wrap_in_tuple

from .mock_pycuda import MockPyCUDA
from .mock_pyopencl import MockPyOpenCL


def mock_input(monkeypatch, inputs):
    inputs_str = "".join(input_ + "\n" for input_ in inputs)
    monkeypatch.setattr('sys.stdin', io.StringIO(inputs_str))


def mock_backend_obj(monkeypatch, api_id, backend):
    if api_id == CUDA_API_ID:
        monkeypatch.setattr('grunnur.adapter_cuda.pycuda_drv', backend.pycuda_drv)
    elif api_id == OPENCL_API_ID:
        monkeypatch.setattr('grunnur.adapter_opencl.pyopencl', backend.pyopencl)
    else:
        raise ValueError(f"Unknown API ID: {api_id}")


def mock_backend(monkeypatch, api_id, *args):
    if api_id == CUDA_API_ID:
        backend = MockPyCUDA(*args)
    elif api_id == OPENCL_API_ID:
        backend = MockPyOpenCL(*args)
    else:
        raise ValueError(f"Unknown API ID: {api_id}")
    mock_backend_obj(monkeypatch, api_id, backend)
    return backend


class BackendDisabled:
    pycuda_drv = None
    pyopencl = None


def disable_backend(monkeypatch, api_id):
    mock_backend_obj(monkeypatch, api_id, BackendDisabled())


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


def check_select_devices(monkeypatch, capsys, platforms_devices, inputs=None, **kwds):

    # CUDA API has a single fixed platform, so using the OpenCL one
    mock_backend(monkeypatch, OPENCL_API_ID, platforms_devices)

    if inputs is not None:
        mock_input(monkeypatch, inputs)

    api = API.from_api_id(OPENCL_API_ID)

    try:
        devices = select_devices(api, **kwds)
    finally:
        if inputs is not None:
            # Otherwise the output will be shown in the console
            captured = capsys.readouterr()

    return devices
