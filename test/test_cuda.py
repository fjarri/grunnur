from grunnur.cuda import CUDA_API_FACTORY

from mock_pycuda import MockPyCUDA
from utils import mock_backend, mock_backend_obj


def test_context_from_pycuda_devices(monkeypatch):

    api_factory = CUDA_API_FACTORY
    backend = MockPyCUDA(['Device1', 'Device2', 'Device3'])
    mock_backend_obj(monkeypatch, api_factory, backend)

    api = api_factory.make_api()

    devices = [backend.Device(0), backend.Device(2)]
    context = api.create_context(devices)


def test_context_from_pycuda_devices_generic(monkeypatch):

    api_factory = CUDA_API_FACTORY
    backend = MockPyCUDA(['Device1', 'Device2', 'Device3'])
    mock_backend_obj(monkeypatch, api_factory, backend)

    api = api_factory.make_api()

    devices = [backend.Device(0), backend.Device(2)]
    context = api.create_context(devices)

    assert isinstance(context, api._context_class)


def test_context_from_pycuda_contexts(monkeypatch):

    api_factory = CUDA_API_FACTORY
    backend = MockPyCUDA(['Device1', 'Device2', 'Device3'])
    mock_backend_obj(monkeypatch, api_factory, backend)

    api = api_factory.make_api()

    devices = [backend.Device(0), backend.Device(2)]
    contexts = []
    for device in devices:
        context = device.make_context()
        context.pop()
        contexts.append(context)

    context = api.create_context(contexts)


def test_context_from_grunnur_devices(monkeypatch):

    api_factory = CUDA_API_FACTORY
    backend = MockPyCUDA(['Device1', 'Device2', 'Device3'])
    mock_backend_obj(monkeypatch, api_factory, backend)

    api = api_factory.make_api()
    platform = api.get_platforms()[0]
    devices = platform.get_devices()[0:2]
    context = api.create_context(devices)
