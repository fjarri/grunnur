import re

import pytest

import grunnur
from grunnur import cuda_api_id, opencl_api_id
from grunnur._adapter_base import APIID
from grunnur._testing import MockBackendFactory, MockPyCUDA, MockPyOpenCL


def test_mock_pyopencl(monkeypatch: pytest.MonkeyPatch) -> None:
    fac = MockBackendFactory(monkeypatch)

    backend = fac.mock(opencl_api_id())
    assert isinstance(backend, MockPyOpenCL)
    assert grunnur._adapter_opencl.pyopencl._backend is backend  # type: ignore[attr-defined]

    fac.disable(opencl_api_id())
    assert grunnur._adapter_opencl.pyopencl is None


def test_mock_pycuda(monkeypatch: pytest.MonkeyPatch) -> None:
    fac = MockBackendFactory(monkeypatch)

    backend = fac.mock(cuda_api_id())
    assert isinstance(backend, MockPyCUDA)
    assert grunnur._adapter_cuda.pycuda_driver is backend.pycuda_driver  # type: ignore[comparison-overlap]
    assert grunnur._adapter_cuda.pycuda_compiler is backend.pycuda_compiler  # type: ignore[comparison-overlap]

    fac.disable(cuda_api_id())
    # `mypy` assumes `grunnur._adapter_cuda.pycuda_driver` is always imported, so it's never `None`,
    # so it considers the second assertion unreachable.
    # `getattr()` circumvents the static check (which makes `ruff` unhappy, but we ignore that).
    assert getattr(grunnur._adapter_cuda, "pycuda_driver") is None  # noqa: B009
    assert grunnur._adapter_cuda.pycuda_compiler is None


def test_unknown_id(monkeypatch: pytest.MonkeyPatch) -> None:
    fac = MockBackendFactory(monkeypatch)

    foo_id = APIID("foo")

    with pytest.raises(ValueError, match=re.escape("Unknown API ID: id(foo)")):
        fac.mock(foo_id)

    with pytest.raises(ValueError, match=re.escape("Unknown API ID: id(foo)")):
        fac.disable(foo_id)
