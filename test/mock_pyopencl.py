from enum import Enum
import weakref

import numpy

from grunnur import OPENCL_API_ID
from grunnur.adapter_base import DeviceType

from .mock_base import MockSourceStr, DeviceInfo


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

    def add_platform_with_devices(self, platform_name, device_infos):
        platform = self.add_platform(platform_name)
        for device_info in device_infos:
            if isinstance(device_info, str):
                device_info = DeviceInfo(name=device_info)
            platform.add_device(device_info)
        return platform

    def add_devices(self, device_infos):
        # Prevent incorrect usage - this method is added to be similar to that of PyCUDA mock,
        # so it can only be used once.
        assert len(self.platforms) == 0
        return self.add_platform_with_devices(None, device_infos)


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
            dest._access_from(queue.device)
            dest_size = dest.size
        else:
            assert isinstance(dest, numpy.ndarray)
            dest_size = dest.size * dest.dtype.itemsize

        if isinstance(src, Buffer):
            assert queue.context == src.context
            src._access_from(queue.device)
            src_size = src.size
        else:
            assert isinstance(src, numpy.ndarray)
            src_size = src.dtype.itemsize

        assert dest_size >= src_size

    def enqueue_marker(self, queue, wait_for=None):
        return Event(queue)

    def enqueue_migrate_mem_objects(self, queue, mem_objects, flags=0, wait_for=None):
        for mem_object in mem_objects:
            mem_object._migrate(queue.device)


class Event:

    def __init__(self, queue):
        self.command_queue = queue


class Platform:

    def __init__(self, backend, name):
        self._backend_ref = weakref.ref(backend)
        self.name = name
        self._devices = []

    def add_device(self, device_info):
        device = Device(self, device_info)
        self._devices.append(device)

    def get_devices(self):
        return self._devices


class Device:

    def __init__(self, platform, device_info):

        self.name = device_info.name
        self._backend_ref = platform._backend_ref
        self._platform_ref = weakref.ref(platform)

        self.max_work_group_size = device_info.max_total_local_size
        self.max_work_item_sizes = [device_info.max_total_local_size] * 3
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

    def finish(self):
        pass


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

        if self.src.mock.should_fail:
            raise PyopenclRuntimeError()

        self._kernels = {kernel.name: Kernel(self, kernel) for kernel in self.src.mock.kernels}

    def __getattr__(self, name):
        return self._kernels[name]


class Kernel:

    def __init__(self, program, kernel):
        self.program = program
        self._kernel = kernel

    def __call__(self, queue, global_size, local_size, *args, wait_for=None):
        assert isinstance(global_size, tuple)
        assert 1 <= len(global_size) <= 3
        if local_size is not None:
            assert isinstance(local_size, tuple)
            assert len(local_size) == len(global_size)

        assert len(args) == len(self._kernel.parameters)

        for arg, param in zip(args, self._kernel.parameters):
            if isinstance(arg, Buffer):
                assert param is None
                assert arg.context == queue.context
            elif isinstance(arg, numpy.number):
                assert arg == param
            else:
                raise TypeError(f"Incorrect argument type: {type(arg)}")


class Buffer:

    def __init__(self, context, flags, size, _migrated_to=None):
        self.context = context
        self.flags = flags
        self.size = size
        self._migrated_to = _migrated_to

    def _migrate(self, device):
        self._migrated_to = device

    def _access_from(self, device):
        if self._migrated_to is None:
            self._migrated_to = device
        else:
            raise RuntimeError("Trying to access a buffer from a different device it was migrated to")

    def get_sub_region(self, origin, size):
        assert origin + size <= self.size
        return Buffer(self.context, self.flags, size, _migrated_to=self._migrated_to)
