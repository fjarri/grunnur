import io

import pytest

from grunnur import API, Context, Device, cuda_api_id, opencl_api_id
from grunnur.api import all_api_ids
from grunnur.adapter_base import APIID
from grunnur.virtual_alloc import TrivialManager, ZeroOffsetManager
from grunnur.testing import MockPyCUDA, MockPyOpenCL, MockBackendFactory

from pytest_grunnur import get_devices, get_multi_device_sets

# Cannot just use the plugin directly since it is loaded before the coverage plugin,
# and all the things the plugin uses from `grunnur` get marked as not covered.
from pytest_grunnur.plugin import context
import pytest_grunnur.plugin


def pytest_addoption(parser):
    # Manually adding options from `pytest_grunnur` (see the note in the imports)
    pytest_grunnur.plugin.pytest_addoption(parser)


def pytest_report_header(config):
    # Manually adding the header from `pytest_grunnur` (see the note in the imports)
    pytest_grunnur.plugin.pytest_report_header(config)


@pytest.fixture(scope="function")
def mock_backend_factory(request, monkeypatch):
    yield MockBackendFactory(monkeypatch)


@pytest.fixture(
    scope="function", params=all_api_ids(), ids=lambda api_id: f"mock_backend_{api_id.shortcut}"
)
def mock_backend(request, mock_backend_factory):
    yield mock_backend_factory.mock(request.param)


@pytest.fixture(scope="function")
def mock_backend_pyopencl(request, mock_backend_factory):
    yield mock_backend_factory.mock_pyopencl()


@pytest.fixture(scope="function")
def mock_backend_pycuda(request, mock_backend_factory):
    yield mock_backend_factory.mock_pycuda()


def make_context(backend, devices_num):
    api_id = backend.api_id
    backend.add_devices(["Device" + str(i) for i in range(devices_num)])
    api = API.from_api_id(api_id)
    return Context.from_criteria(api, devices_num=devices_num)


@pytest.fixture(
    scope="function", params=all_api_ids(), ids=lambda api_id: f"mock_{api_id.shortcut}"
)
def mock_context(request, mock_backend_factory):
    backend = mock_backend_factory.mock(request.param)
    yield make_context(backend, 1)


@pytest.fixture(scope="function")
def mock_context_pycuda(request, mock_backend_pycuda):
    yield make_context(mock_backend_pycuda, 1)


@pytest.fixture(
    scope="function", params=all_api_ids(), ids=lambda api_id: f"mock_{api_id.shortcut}_4dev"
)
def mock_4_device_context(request, mock_backend_factory):
    backend = mock_backend_factory.mock(request.param)
    yield make_context(backend, 4)


@pytest.fixture(scope="function")
def mock_4_device_context_pyopencl(request, mock_backend_pyopencl):
    yield make_context(mock_backend_pyopencl, 4)


@pytest.fixture(scope="function")
def mock_or_real_context(request, monkeypatch):
    # Since `py.test` does not support concatenating fixtures,
    # we concatenate real contexts and mocked contexts manually.
    # If there are no devices available, we need it to be noticeable that some tests are skipped,
    # hence passing `None` and calling `skip()` explicitly in this case.
    value = request.param
    if value is None:
        pytest.skip("No GPGPU devices available")
    elif isinstance(value, Device):
        yield (Context.from_devices([value]), False)
    elif isinstance(value, APIID):
        factory = MockBackendFactory(monkeypatch)
        backend = factory.mock(value)
        yield (make_context(backend, 1), True)
    else:
        raise TypeError


@pytest.fixture(scope="function")
def mock_or_real_multi_device_context(request, monkeypatch):
    # Same as `mock_or_real_context` above, but for 2-device contexts.
    value = request.param
    if value is None:
        pytest.skip("Could not find 2 suitable GPGPU devices on the same platform")
    elif isinstance(value, list):
        yield (Context.from_devices(value), False)
    elif isinstance(value, APIID):
        factory = MockBackendFactory(monkeypatch)
        backend = factory.mock(value)
        yield (make_context(backend, 2), True)
    else:
        raise TypeError


class MockStdin:
    def __init__(self):
        self.stream = io.StringIO()
        self.lines = 0

    def line(self, s):
        pos = self.stream.tell()  # preserve the current read pointer
        self.stream.seek(0, io.SEEK_END)
        self.stream.write(s + "\n")
        self.stream.seek(pos)
        self.lines += 1

    def readline(self):
        assert self.lines > 0
        self.lines -= 1
        return self.stream.readline()

    def empty(self):
        return self.lines == 0


@pytest.fixture(scope="function")
def mock_stdin(monkeypatch):
    mock = MockStdin()
    monkeypatch.setattr("sys.stdin", mock)
    yield mock
    assert mock.empty()


@pytest.fixture(params=[TrivialManager, ZeroOffsetManager], ids=["trivial", "zero_offset"])
def valloc_cls(request):
    yield request.param


def _mock_or_real_context_make_id(value):
    if value is None:
        return "no_real_devices"
    elif isinstance(value, Device):
        return value.shortcut
    elif isinstance(value, APIID):
        return f"mock_{value.shortcut}"
    else:
        raise TypeError


def _mock_or_real_multi_device_context_make_id(value):
    if value is None:
        return "no_2_real_devices"
    elif isinstance(value, list):
        return "+".join(device.shortcut for device in value)
    elif isinstance(value, APIID):
        return f"mock_{value.shortcut}_2dev"
    else:
        raise TypeError


def pytest_generate_tests(metafunc):
    # Manually adding the fixtures from `pytest_grunnur` (see the note in the imports)
    pytest_grunnur.plugin.pytest_generate_tests(metafunc)

    api_ids = all_api_ids()
    devices = get_devices(metafunc.config)
    device_sets = get_multi_device_sets(metafunc.config)

    if "mock_or_real_context" in metafunc.fixturenames:
        values = (devices if len(devices) > 0 else [None]) + all_api_ids()
        ids = [_mock_or_real_context_make_id(value) for value in values]
        metafunc.parametrize("mock_or_real_context", values, ids=ids, indirect=True)

    if "mock_or_real_multi_device_context" in metafunc.fixturenames:
        values = (device_sets if len(device_sets) > 0 else [None]) + all_api_ids()
        ids = [_mock_or_real_multi_device_context_make_id(value) for value in values]
        metafunc.parametrize("mock_or_real_multi_device_context", values, ids=ids, indirect=True)
