import numpy

from grunnur import API, dtypes
from grunnur.device_discovery import select_devices


def get_test_array(shape, dtype, *, strides=None, offset=0, no_zeros=False, high=None):
    dtype = numpy.dtype(dtype)  # normalize to access the fields

    rng = numpy.random.default_rng()

    if offset != 0:
        raise NotImplementedError

    if dtype.names is not None:
        result = numpy.empty(shape, dtype)
        for name in dtype.names:
            result[name] = get_test_array(shape, dtype[name], no_zeros=no_zeros, high=high)
    else:
        if dtypes.is_integer(dtype):
            low = 1 if no_zeros else 0
            if high is None:
                high = 100  # will work even with signed chars
            get_arr = lambda: rng.integers(low, high, shape, dtype)
        else:
            low = 0.01 if no_zeros else 0
            if high is None:
                high = 1.0
            get_arr = lambda: rng.uniform(low, high, shape).astype(dtype)

        result = (get_arr() + 1j * get_arr()) if dtypes.is_complex(dtype) else get_arr()

    if strides is not None:
        result = numpy.lib.stride_tricks.as_strided(result, result.shape, strides)

    return result


def check_select_devices(
    mock_stdin,
    mock_backend_factory,
    capsys,
    platforms_devices,
    *,
    inputs=None,
    interactive=False,
    quantity=1,
    device_filter=None,
    platform_filter=None,
):
    # CUDA API has a single fixed platform, so using the OpenCL one
    backend = mock_backend_factory.mock_pyopencl()

    for platform_name, device_infos in platforms_devices:
        backend.add_platform_with_devices(platform_name, device_infos)

    if inputs is not None:
        for line in inputs:
            mock_stdin.line(line)

    api = API.from_api_id(backend.api_id)

    try:
        devices = select_devices(
            api,
            interactive=interactive,
            quantity=quantity,
            device_filter=device_filter,
            platform_filter=platform_filter,
        )
        assert mock_stdin.empty()
    finally:
        # Otherwise the output will be shown in the console
        capsys.readouterr()

    return devices
