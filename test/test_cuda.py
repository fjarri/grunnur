from grunnur import API, CUDA_API_ID, Context, Platform, Device

from .mock_pycuda import MockPyCUDA
from .utils import mock_backend


def test_context_from_pycuda_devices(monkeypatch):

    backend = mock_backend(
        monkeypatch, CUDA_API_ID, [('Platform1', ['Device1', 'Device2', 'Device3'])])

    devices = [backend.pycuda_drv.Device(0), backend.pycuda_drv.Device(2)]
    context = Context.from_backend_devices(devices)


def test_context_from_pycuda_contexts(monkeypatch):

    backend = mock_backend(
        monkeypatch, CUDA_API_ID, [('Platform1', ['Device1', 'Device2', 'Device3'])])

    devices = [backend.pycuda_drv.Device(0), backend.pycuda_drv.Device(2)]
    contexts = []
    for device in devices:
        context = device.make_context()
        context.pop()
        contexts.append(context)

    context = Context.from_backend_contexts(contexts)


def test_context_from_grunnur_devices(monkeypatch):

    backend = mock_backend(
        monkeypatch, CUDA_API_ID, [('Platform1', ['Device1', 'Device2', 'Device3'])])

    api = API.from_api_id(CUDA_API_ID)

    platform = Platform.all(api)[0]
    devices = Device.all(platform)[0:2]

    context = Context.from_devices(devices)
