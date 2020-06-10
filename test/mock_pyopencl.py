from contextlib import contextmanager
from enum import Enum
import weakref

from grunnur import OPENCL_API_ID


class DeviceType(Enum):
    CPU = 1
    GPU = 2


class MockPyOpenCL:

    def __init__(self):
        self.pyopencl = Mock_pyopencl(self)
        self.api_id = OPENCL_API_ID
        self.platforms = []

        self.compilation_succeeds = True

    def add_platform(self, platform_name=None):
        if platform_name is None:
            platform_name = 'Platform' + str(len(self.platforms))
        platform = Platform(self, platform_name)
        self.platforms.append(platform)
        return platform

    def add_platform_with_devices(self, platform_name, device_names):
        platform = self.add_platform(platform_name)
        for device_name in device_names:
            platform.add_device(device_name)
        return platform

    def add_devices(self, device_names):
        # Prevent incorrect usage - this method is added to be similar to that of PyCUDA mock,
        # so it can only be used once.
        assert len(self.platforms) == 0
        return self.add_platform_with_devices(None, device_names)

    @contextmanager
    def make_compilation_fail(self):
        self.compilation_succeeds = False
        yield
        self.compilation_succeeds = True


class PyopenclRuntimeError(Exception):
    pass


class Mock_pyopencl:

    def __init__(self, backend):

        self._backend_ref = weakref.ref(backend)

        self.device_type = DeviceType

        self.Device = Device
        self.Platform = Platform
        self.Context = Context
        self.Program = Program

        self.RuntimeError = PyopenclRuntimeError

    def get_platforms(self):
        return self._backend_ref().platforms


class Platform:

    def __init__(self, backend, name):
        self._backend_ref = weakref.ref(backend)
        self.name = name
        self._devices = []

    def add_device(self, *args, **kwds):
        device = Device(self, *args, **kwds)
        self._devices.append(device)

    def get_devices(self):
        return self._devices


class Device:

    def __init__(self, platform, name, max_work_group_size=1024):
        self.name = name
        self._backend_ref = platform._backend_ref
        self._platform_ref = weakref.ref(platform)

        self.max_work_group_size = max_work_group_size
        self.max_work_item_sizes = [max_work_group_size] * 3
        self.address_bits = 64
        self.type = DeviceType.GPU
        self.extensions = []
        self.vendor = 'Mock Devices'
        self.local_mem_size = 48 * 1024
        self.max_compute_units = 1

    @property
    def platform(self):
        return self._platform_ref()


class Context:

    def __init__(self, devices):
        self._backend_ref = devices[0]._backend_ref
        self.devices = devices


class Program:

    def __init__(self, context, src):
        self._backend_ref = context._backend_ref
        self.context = context
        self.src = src

    def build(self, options=[], devices=None, cache_dir=None):
        assert all(isinstance(option, str) for option in options)
        assert cache_dir is None or isinstance(cache_dir, str)
        assert devices is None or all(device in self.context.devices for device in devices)

        if not self._backend_ref().compilation_succeeds:
            raise PyopenclRuntimeError()
