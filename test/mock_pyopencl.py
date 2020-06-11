from enum import Enum
import weakref

import numpy

from grunnur import OPENCL_API_ID

from .mock_base import MockSourceStr


class DeviceType(Enum):
    CPU = 1
    GPU = 2


class MemFlags(Enum):
    READ_WRITE = 1


class MockPyOpenCL:

    def __init__(self):
        self.pyopencl = Mock_pyopencl(self)
        self.api_id = OPENCL_API_ID
        self.platforms = []

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
        self.CommandQueue = CommandQueue
        self.Buffer = Buffer

        self.RuntimeError = PyopenclRuntimeError

        self.mem_flags = MemFlags

    def get_platforms(self):
        return self._backend_ref().platforms

    def enqueue_copy(self, queue, dest, src, wait_for=None, is_blocking=False):
        if isinstance(dest, Buffer):
            assert queue.context == dest.context
        else:
            assert isinstance(dest, numpy.ndarray)
        if isinstance(src, Buffer):
            assert queue.context == src.context
        else:
            assert isinstance(src, numpy.ndarray)

        assert dest.size >= src.size


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


class CommandQueue:

    def __init__(self, context, device=None):
        self._backend_ref = context._backend_ref
        self.context = context

        if device is None:
            self.device = context.devices[0]
        else:
            assert device in context.devices
            self.device = device


class Program:

    def __init__(self, context, src):
        self._backend_ref = context._backend_ref
        self.context = context
        self.src = src
        self._kernels = {}

    def build(self, options=[], devices=None, cache_dir=None):
        assert isinstance(self.src, MockSourceStr)
        assert all(isinstance(option, str) for option in options)
        assert cache_dir is None or isinstance(cache_dir, str)
        assert devices is None or all(device in self.context.devices for device in devices)

        if self.src.should_fail:
            raise PyopenclRuntimeError()

        self._kernels = {kernel.name: Kernel(self, kernel) for kernel in self.src.kernels}

    def __getattr__(self, name):
        return self._kernels[name]


class Kernel:

    def __init__(self, program, kernel):
        self.program = program
        self.kernel = kernel

    def __call__(self, queue, global_size, local_size, *args, wait_for=None):
        assert isinstance(global_size, tuple)
        assert 1 <= len(global_size) <= 3
        if local_size is not None:
            assert isinstance(local_size, tuple)
            assert len(local_size) == len(global_size)

        for arg in args:
            if isinstance(arg, Buffer):
                assert arg.context == queue.context
            elif isinstance(numpy.number):
                pass
            else:
                raise TypeError(f"Incorrect argument type: {type(arg)}")


class Buffer:

    def __init__(self, context, flags, size):
        self.context = context
        self.flags = flags
        self.size = size
