import io

import pytest

from grunnur import API, Context, CUDA_API_ID, OPENCL_API_ID
from grunnur.api import all_api_ids
from grunnur.pytest_helpers import *

from .mock_pycuda import MockPyCUDA
from .mock_pyopencl import MockPyOpenCL


class MockBackendFactory:

    def __init__(self, monkeypatch):
        self.monkeypatch = monkeypatch

        # Clear out all existing backends
        for api_id in all_api_ids():
            self.mock(api_id, disable=True)

    def _set_backend_cuda(self, backend=None):
        pycuda_driver = backend.pycuda_driver if backend else None
        pycuda_compiler = backend.pycuda_compiler if backend else None
        self.monkeypatch.setattr('grunnur.adapter_cuda.pycuda_driver', pycuda_driver)
        self.monkeypatch.setattr('grunnur.adapter_cuda.pycuda_compiler', pycuda_compiler)

    def mock_cuda(self, disable=False):
        backend = MockPyCUDA() if not disable else None
        self._set_backend_cuda(backend)
        return backend

    def _set_backend_opencl(self, backend=None):
        pyopencl = backend.pyopencl if backend else None
        self.monkeypatch.setattr('grunnur.adapter_opencl.pyopencl', pyopencl)

    def mock_opencl(self, disable=False):
        backend = MockPyOpenCL() if not disable else None
        self._set_backend_opencl(backend)
        return backend

    def mock(self, api_id, disable=False):
        if api_id == CUDA_API_ID:
            return self.mock_cuda(disable=disable)
        elif api_id == OPENCL_API_ID:
            return self.mock_opencl(disable=disable)
        else:
            raise ValueError(f"Unknown API ID: {api_id}")


@pytest.fixture(scope='function')
def mock_backend_factory(request, monkeypatch):
    yield MockBackendFactory(monkeypatch)


@pytest.fixture(
    scope='function',
    params=all_api_ids(),
    ids=lambda api_id: f"mock-{api_id.shortcut}-backend")
def mock_backend(request, mock_backend_factory):
    yield mock_backend_factory.mock(request.param)


@pytest.fixture(scope='function')
def mock_backend_pyopencl(request, mock_backend_factory):
    yield mock_backend_factory.mock_opencl()


@pytest.fixture(scope='function')
def mock_backend_pycuda(request, mock_backend_factory):
    yield mock_backend_factory.mock_cuda()


@pytest.fixture(scope='function')
def mock_context(request, mock_backend):
    mock_backend.add_devices(['Device1'])

    api_id = mock_backend.api_id
    api = API.from_api_id(api_id)
    context = Context.from_criteria(api)
    yield context

    # The yielded object is preserved somewhere inside PyTest, so there is a race condition
    # between `context` destructor and `monkeypatch` rollback, which leads to
    # actual PyCUDA context being popped instead of the mock.
    # So we are releasing the context stack manually in the PyCUDA case.

    if api_id == CUDA_API_ID:
        context._context_adapter._context_stack = None


class MockStdin:

    def __init__(self):
        self.stream = io.StringIO()
        self.lines = 0

    def line(self, s):
        pos = self.stream.tell() # preserve the current read pointer
        self.stream.seek(0, io.SEEK_END)
        self.stream.write(s + '\n')
        self.stream.seek(pos)
        self.lines += 1

    def readline(self):
        assert self.lines > 0
        self.lines -= 1
        return self.stream.readline()

    def empty(self):
        return self.lines == 0


@pytest.fixture(scope='function')
def mock_stdin(monkeypatch):
    mock = MockStdin()
    monkeypatch.setattr('sys.stdin', mock)
    yield mock
    assert mock.empty()


def pytest_addoption(parser):
    addoption(parser)


def pytest_generate_tests(metafunc):
    generate_tests(metafunc)


def pytest_report_header(config):
    report_header(config)
