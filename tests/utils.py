import io
from collections.abc import Iterable, Sequence
from typing import Any

import numpy
import pytest
from numpy.typing import DTypeLike, NDArray

from grunnur import API, Device, DeviceFilter, PlatformFilter, dtypes, select_devices
from grunnur._testing import MockBackendFactory, MockPyOpenCL, PyOpenCLDeviceInfo


def get_test_array(
    shape: Sequence[int],
    dtype: DTypeLike,
    *,
    strides: None | Sequence[int] = None,
    no_zeros: bool = False,
    high: float | None = None,
) -> NDArray[Any]:
    dtype = numpy.dtype(dtype)  # normalize to access the fields

    rng = numpy.random.default_rng()

    if numpy.issubdtype(dtype, numpy.integer):
        low_int = 1 if no_zeros else 0
        # `100` will work even with signed chars
        high_int = 100 if high is None else int(high)
        result = rng.integers(low_int, high=high_int, size=shape).astype(dtype)
    else:
        low_float = 0.01 if no_zeros else 0
        high_float = 1.0 if high is None else float(high)

        def get_arr() -> NDArray[Any]:
            return rng.uniform(low_float, high_float, size=shape).astype(dtype)

        result = (get_arr() + 1j * get_arr()) if dtypes.is_complex(dtype) else get_arr()

    if strides is not None:
        result = numpy.lib.stride_tricks.as_strided(result, result.shape, strides)

    return result


class MockStdin:
    def __init__(self) -> None:
        self.stream = io.StringIO()
        self.lines = 0

    def line(self, s: str) -> None:
        pos = self.stream.tell()  # preserve the current read pointer
        self.stream.seek(0, io.SEEK_END)
        self.stream.write(s + "\n")
        self.stream.seek(pos)
        self.lines += 1

    def readline(self) -> str:
        assert self.lines > 0
        self.lines -= 1
        return self.stream.readline()

    def empty(self) -> bool:
        return self.lines == 0


def check_select_devices(
    mock_stdin: MockStdin,
    mock_backend_factory: MockBackendFactory,
    capsys: pytest.CaptureFixture[str],
    platforms_devices: Iterable[tuple[str, Sequence[str | PyOpenCLDeviceInfo]]],
    *,
    inputs: Iterable[str] | None = None,
    interactive: bool = False,
    quantity: int | None = 1,
    device_filter: DeviceFilter | None = None,
    platform_filter: PlatformFilter | None = None,
) -> list[Device]:
    # CUDA API has a single fixed platform, so using the OpenCL one
    backend = mock_backend_factory.mock_pyopencl()
    assert isinstance(backend, MockPyOpenCL)

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
