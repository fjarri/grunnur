import io
from collections.abc import Iterator, Sequence
from typing import cast

import pytest
import pytest_grunnur.plugin
from pytest_grunnur import get_devices, get_multi_device_sets

# Cannot just use the plugin directly since it is loaded before the coverage plugin,
# and all the things the plugin uses from `grunnur` get marked as not covered.
from pytest_grunnur.plugin import context, device

from grunnur import (
    API,
    Context,
    Device,
    TrivialManager,
    VirtualManager,
    ZeroOffsetManager,
    all_api_ids,
    cuda_api_id,
    opencl_api_id,
)
from grunnur._adapter_base import APIID
from grunnur._testing import MockBackendFactory, MockPyCUDA, MockPyOpenCL
from utils import MockStdin


def pytest_addoption(parser: pytest.Parser) -> None:
    # Manually adding options from `pytest_grunnur` (see the note in the imports)
    pytest_grunnur.plugin.pytest_addoption(parser)


def pytest_report_header(config: pytest.Config) -> None:
    # Manually adding the header from `pytest_grunnur` (see the note in the imports)
    pytest_grunnur.plugin.pytest_report_header(config)


@pytest.fixture
def mock_backend_factory(monkeypatch: pytest.MonkeyPatch) -> MockBackendFactory:
    return MockBackendFactory(monkeypatch)


@pytest.fixture(params=all_api_ids(), ids=lambda api_id: f"mock_backend_{api_id.shortcut}")
def mock_backend(
    request: pytest.FixtureRequest, mock_backend_factory: MockBackendFactory
) -> MockPyOpenCL | MockPyCUDA:
    return mock_backend_factory.mock(request.param)


@pytest.fixture
def mock_backend_pyopencl(mock_backend_factory: MockBackendFactory) -> MockPyOpenCL:
    return mock_backend_factory.mock_pyopencl()


@pytest.fixture
def mock_backend_pycuda(mock_backend_factory: MockBackendFactory) -> MockPyCUDA:
    return mock_backend_factory.mock_pycuda()


def make_context(backend: MockPyOpenCL | MockPyCUDA, devices_num: int) -> Context:
    api_id = backend.api_id
    backend.add_devices(["Device" + str(i) for i in range(devices_num)])
    api = API.from_api_id(api_id)
    return Context.from_criteria(api, devices_num=devices_num)


@pytest.fixture(params=all_api_ids(), ids=lambda api_id: f"mock_{api_id.shortcut}")
def mock_context(
    request: pytest.FixtureRequest, mock_backend_factory: MockBackendFactory
) -> Context:
    backend = mock_backend_factory.mock(request.param)
    return make_context(backend, 1)


@pytest.fixture
def mock_context_pycuda(mock_backend_pycuda: MockPyCUDA) -> Context:
    return make_context(mock_backend_pycuda, 1)


@pytest.fixture(params=all_api_ids(), ids=lambda api_id: f"mock_{api_id.shortcut}_4dev")
def mock_4_device_context(
    request: pytest.FixtureRequest, mock_backend_factory: MockBackendFactory
) -> Context:
    backend = mock_backend_factory.mock(request.param)
    return make_context(backend, 4)


@pytest.fixture
def mock_4_device_context_pyopencl(mock_backend_pyopencl: MockPyOpenCL) -> Context:
    return make_context(mock_backend_pyopencl, 4)


@pytest.fixture
def mock_or_real_context(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[Context, bool]]:
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


@pytest.fixture
def mock_or_real_multi_device_context(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[Context, bool]]:
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


@pytest.fixture
def mock_stdin(monkeypatch: pytest.MonkeyPatch) -> Iterator[MockStdin]:
    mock = MockStdin()
    monkeypatch.setattr("sys.stdin", mock)
    yield mock
    assert mock.empty()


@pytest.fixture(params=[TrivialManager, ZeroOffsetManager], ids=["trivial", "zero_offset"])
def valloc_cls(request: pytest.FixtureRequest) -> type[VirtualManager]:
    return cast(type[VirtualManager], request.param)


def _mock_or_real_context_make_id(value: None | Device | APIID) -> str:
    if value is None:
        return "no_real_devices"
    if isinstance(value, Device):
        return value.shortcut
    if isinstance(value, APIID):
        return f"mock_{value.shortcut}"
    raise TypeError


def _mock_or_real_multi_device_context_make_id(value: None | list[Device] | APIID) -> str:
    if value is None:
        return "no_2_real_devices"
    if isinstance(value, list):
        return "+".join(device.shortcut for device in value)
    if isinstance(value, APIID):
        return f"mock_{value.shortcut}_2dev"
    raise TypeError


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    # Manually adding the fixtures from `pytest_grunnur` (see the note in the imports)
    pytest_grunnur.plugin.pytest_generate_tests(metafunc)

    api_ids = all_api_ids()
    devices: Sequence[None | Device] = get_devices(metafunc.config)
    device_sets: Sequence[None | list[Device]] = get_multi_device_sets(metafunc.config)

    if "mock_or_real_context" in metafunc.fixturenames:
        values: list[None | Device | APIID] = [*(devices if len(devices) > 0 else [None]), *api_ids]
        ids = [_mock_or_real_context_make_id(value) for value in values]
        metafunc.parametrize("mock_or_real_context", values, ids=ids, indirect=True)

    if "mock_or_real_multi_device_context" in metafunc.fixturenames:
        md_values: list[None | list[Device] | APIID] = [
            *(device_sets if len(device_sets) > 0 else [None]),
            *api_ids,
        ]
        ids = [_mock_or_real_multi_device_context_make_id(value) for value in md_values]
        metafunc.parametrize("mock_or_real_multi_device_context", md_values, ids=ids, indirect=True)
