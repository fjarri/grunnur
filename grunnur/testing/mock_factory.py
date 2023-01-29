from typing import Protocol, Any, Optional, Union

from .. import all_api_ids, cuda_api_id, opencl_api_id
from ..adapter_base import APIID

from .mock_pycuda import MockPyCUDA
from .mock_pyopencl import MockPyOpenCL


# Have to mock it like this since we don't want to make `pytest` a dependency for the main library.
class MonkeyPatch(Protocol):
    def setattr(self, attr: str, obj: Any) -> None:  # pragma: no cover
        ...


class MockBackendFactory:
    def __init__(self, monkeypatch: MonkeyPatch):
        self.monkeypatch = monkeypatch

        # Clear out all existing backends
        for api_id in all_api_ids():
            self.disable(api_id)

    def _set_backend_cuda(self, backend: Optional[MockPyCUDA] = None) -> None:
        pycuda_driver = backend.pycuda_driver if backend else None
        pycuda_compiler = backend.pycuda_compiler if backend else None
        self.monkeypatch.setattr("grunnur.adapter_cuda.pycuda_driver", pycuda_driver)
        self.monkeypatch.setattr("grunnur.adapter_cuda.pycuda_compiler", pycuda_compiler)

    def mock_pycuda(self) -> MockPyCUDA:
        backend = MockPyCUDA()
        self._set_backend_cuda(backend)
        return backend

    def disable_pycuda(self) -> None:
        self._set_backend_cuda(None)

    def _set_backend_opencl(self, backend: Optional[MockPyOpenCL] = None) -> None:
        pyopencl = backend.pyopencl if backend else None
        self.monkeypatch.setattr("grunnur.adapter_opencl.pyopencl", pyopencl)

    def mock_pyopencl(self) -> MockPyOpenCL:
        backend = MockPyOpenCL()
        self._set_backend_opencl(backend)
        return backend

    def disable_pyopencl(self) -> None:
        self._set_backend_opencl(None)

    def disable(self, api_id: APIID) -> None:
        if api_id == cuda_api_id():
            return self.disable_pycuda()
        elif api_id == opencl_api_id():
            return self.disable_pyopencl()
        else:
            raise ValueError(f"Unknown API ID: {api_id}")

    def mock(self, api_id: APIID) -> Union[MockPyOpenCL, MockPyCUDA]:
        if api_id == cuda_api_id():
            return self.mock_pycuda()
        elif api_id == opencl_api_id():
            return self.mock_pyopencl()
        else:
            raise ValueError(f"Unknown API ID: {api_id}")
