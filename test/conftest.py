import io

import pytest

from grunnur import API, Context, cuda_api_id, opencl_api_id
from grunnur.api import all_api_ids
from grunnur.virtual_alloc import TrivialManager, ZeroOffsetManager

# Cannot just use the plugin directly since it is loaded before the coverage plugin,
# and all the function definitions in all `grunnur` modules get marked as not covered.
from grunnur.pytest_plugin import context, multi_device_context
from grunnur.pytest_plugin import pytest_addoption as grunnur_pytest_addoption
from grunnur.pytest_plugin import pytest_generate_tests as grunnur_pytest_generate_tests
from grunnur.pytest_plugin import pytest_report_header as grunnur_pytest_report_header

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

    def mock_pycuda(self, disable=False):
        backend = MockPyCUDA() if not disable else None
        self._set_backend_cuda(backend)
        return backend

    def _set_backend_opencl(self, backend=None):
        pyopencl = backend.pyopencl if backend else None
        self.monkeypatch.setattr('grunnur.adapter_opencl.pyopencl', pyopencl)

    def mock_pyopencl(self, disable=False):
        backend = MockPyOpenCL() if not disable else None
        self._set_backend_opencl(backend)
        return backend

    def mock(self, api_id, disable=False):
        if api_id == cuda_api_id():
            return self.mock_pycuda(disable=disable)
        elif api_id == opencl_api_id():
            return self.mock_pyopencl(disable=disable)
        else:
            raise ValueError(f"Unknown API ID: {api_id}")


@pytest.fixture(scope='function')
def mock_backend_factory(request, monkeypatch):
    yield MockBackendFactory(monkeypatch)


@pytest.fixture(
    scope='function',
    params=all_api_ids(),
    ids=lambda api_id: f"mock_backend_{api_id.shortcut}")
def mock_backend(request, mock_backend_factory):
    yield mock_backend_factory.mock(request.param)


@pytest.fixture(scope='function')
def mock_backend_pyopencl(request, mock_backend_factory):
    yield mock_backend_factory.mock_pyopencl()


@pytest.fixture(scope='function')
def mock_backend_pycuda(request, mock_backend_factory):
    yield mock_backend_factory.mock_pycuda()


def make_context(backend, devices_num):
    api_id = backend.api_id
    backend.add_devices(['Device' + str(i) for i in range(devices_num)])
    api = API.from_api_id(api_id)
    return Context.from_criteria(api, devices_num=devices_num)


@pytest.fixture(scope='function')
def mock_context(request, mock_backend):
    yield make_context(mock_backend, 1)


@pytest.fixture(scope='function')
def mock_context_pycuda(request, mock_backend_pycuda):
    yield make_context(mock_backend_pycuda, 1)


@pytest.fixture(scope='function')
def mock_4_device_context(request, mock_backend):
    yield make_context(mock_backend, 4)


@pytest.fixture(scope='function')
def mock_4_device_context_pyopencl(request, mock_backend_pyopencl):
    yield make_context(mock_backend_pyopencl, 4)


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


@pytest.fixture(
    params=[TrivialManager, ZeroOffsetManager],
    ids=['trivial', 'zero_offset'])
def valloc_cls(request):
    yield request.param


def pytest_addoption(parser):
    grunnur_pytest_addoption(parser)


def pytest_generate_tests(metafunc):
    grunnur_pytest_generate_tests(metafunc)


def pytest_report_header(config):
    grunnur_pytest_report_header(config)
