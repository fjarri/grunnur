import io

import numpy

from grunnur import API, CUDA_API_ID, OPENCL_API_ID
from grunnur.device_discovery import select_devices
import grunnur.dtypes as dtypes
from grunnur.utils import wrap_in_tuple


def mock_input(monkeypatch, inputs):
    inputs_str = "".join(input_ + "\n" for input_ in inputs)
    monkeypatch.setattr('sys.stdin', io.StringIO(inputs_str))


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


def check_select_devices(mock_stdin, mock_backend_factory, capsys, platforms_devices, inputs=None, **kwds):

    # CUDA API has a single fixed platform, so using the OpenCL one
    backend = mock_backend_factory.mock_pyopencl()

    for platform_name, device_params in platforms_devices:
        platform = backend.add_platform(platform_name)
        for params in device_params:
            if isinstance(params, str):
                platform.add_device(params)
            else:
                platform.add_device(**params)

    if inputs is not None:
        for line in inputs:
            mock_stdin.line(line)

    api = API.from_api_id(backend.api_id)

    try:
        devices = select_devices(api, **kwds)
        assert mock_stdin.empty()
    finally:
        # Otherwise the output will be shown in the console
        captured = capsys.readouterr()

    return devices
