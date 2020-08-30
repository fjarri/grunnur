from enum import Enum, IntEnum
import weakref

import numpy

from grunnur import opencl_api_id
from grunnur.adapter_base import DeviceType

from .mock_base import MockSource


class MemFlags(IntEnum):
    READ_WRITE = 1
    ALLOC_HOST_PTR = 16


class MockPyOpenCL:

    def __init__(self):
        self.pyopencl = Mock_pyopencl(self)
        self.api_id = opencl_api_id()
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
                device_info = PyOpenCLDeviceInfo(name=device_info)
            elif isinstance(device_info, PyOpenCLDeviceInfo):
                pass
            else:
                raise TypeError(type(device_info))
            platform.add_device(device_info)
        return platform

    def add_devices(self, device_infos):
        # Prevent incorrect usage - this method is added to be similar to that of PyCUDA mock,
        # so it can only be used once.
        assert len(self.platforms) == 0
        return self.add_platform_with_devices(None, device_infos)


class PyopenclRuntimeError(Exception):
    pass


class KernelWorkGroupInfo(Enum):
    WORK_GROUP_SIZE = 4528


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
        self.kernel_work_group_info = KernelWorkGroupInfo

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
            src_size = src.size * src.dtype.itemsize

        assert dest_size >= src_size

        if isinstance(dest, Buffer):
            dest._set(src)
        elif isinstance(dest, numpy.ndarray) and isinstance(src, Buffer):
            src._get(dest)
        else:
            raise TypeError(f"Not supported: {type(dest)} and {type(src)}")

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
        self.vendor = 'Mock Platforms'
        self.version = 'OpenCL 1.2'

    def add_device(self, device_info):
        device = Device(self, device_info)
        self._devices.append(device)

    def get_devices(self):
        return self._devices


class PyOpenCLDeviceInfo:

    def __init__(
            self,
            name="DefaultDeviceName",
            vendor="Mock Devices",
            type=DeviceType.GPU,
            max_work_group_size=1024,
            max_work_item_sizes=[1024, 1024, 1024],
            local_mem_size=64 * 1024,
            address_bits=32,
            max_compute_units=8,
            extensions=[],
            compute_capability_major_nv=None,
            warp_size_nv=None):
        self.name = name
        self.vendor = vendor
        self.type = type
        self.max_work_group_size = max_work_group_size
        self.max_work_item_sizes = max_work_item_sizes
        self.local_mem_size = local_mem_size
        self.address_bits = address_bits
        self.max_compute_units = max_compute_units
        self.extensions = extensions
        self.compute_capability_major_nv = compute_capability_major_nv
        self.warp_size_nv = warp_size_nv


class Device:

    def __init__(self, platform, device_info):

        self.name = device_info.name
        self._backend_ref = platform._backend_ref
        self._platform_ref = weakref.ref(platform)

        self.max_work_group_size = device_info.max_work_group_size
        self.max_work_item_sizes = device_info.max_work_item_sizes
        self.address_bits = device_info.address_bits
        self.type = device_info.type
        self.extensions = device_info.extensions
        self.vendor = device_info.vendor
        self.local_mem_size = device_info.local_mem_size
        self.max_compute_units = device_info.max_compute_units
        self.compute_capability_major_nv = device_info.compute_capability_major_nv
        self.warp_size_nv = device_info.warp_size_nv

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
        assert isinstance(self.src, MockSource)
        assert all(isinstance(option, str) for option in options)
        assert cache_dir is None or isinstance(cache_dir, str)

        # In Grunnur, we always build separate programs for each device
        assert len(devices) == 1 and devices[0] in self.context.devices

        if self.src.should_fail:
            raise PyopenclRuntimeError()

        self._kernels = {kernel.name: Kernel(self, kernel) for kernel in self.src.kernels}

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
                assert arg.dtype == param
            else:
                raise TypeError(f"Incorrect argument type: {type(arg)}")

    def get_work_group_info(self, attribute, device):
        if attribute == KernelWorkGroupInfo.WORK_GROUP_SIZE:
            device_idx = device.platform.get_devices().index(device)
            return self._kernel.max_total_local_sizes[device_idx]
        else:
            raise ValueError(f"Unknown attribute: {attribute}")


class Buffer:

    def __init__(self, context, flags, size, _migrated_to=None, _offset=0, _base_buffer=None):
        self.context = context
        self.flags = flags
        self.size = size
        self.offset = _offset
        self._migrated_to = _migrated_to

        if _base_buffer is None:
            self._buffer = b"\xef" * size
        self._base_buffer = _base_buffer

    def _migrate(self, device):
        self._migrated_to = device

    def _set(self, arr):
        if isinstance(arr, numpy.ndarray):
            data = arr.tobytes()
        else:
            full_buf = arr._buffer if arr._base_buffer is None else arr._base_buffer._buffer
            data = full_buf[arr.offset:arr.offset + arr.size]

        assert len(data) <= self.size

        insert_data = lambda buf: buf[:self.offset] + data + buf[self.offset+len(data):]

        if self._base_buffer is None:
            self._buffer = insert_data(self._buffer)
        else:
            self._base_buffer._buffer = insert_data(self._base_buffer._buffer)

    def _get(self, arr):
        data = arr.tobytes()
        assert len(data) <= self.size

        full_buf = self._buffer if self._base_buffer is None else self._base_buffer._buffer

        buf = full_buf[self.offset:self.offset + len(data)]
        buf_as_arr = numpy.frombuffer(buf, arr.dtype).reshape(arr.shape)
        numpy.copyto(arr, buf_as_arr)

    def _access_from(self, device):
        if self._migrated_to is None:
            self._migrated_to = device
        elif device != self._migrated_to:
            raise RuntimeError("Trying to access a buffer from a different device it was migrated to")

    def get_sub_region(self, origin, size):
        assert origin + size <= self.size
        # Trying to do that in PyOpenCL leads to segfault
        if self._base_buffer is not None:
            raise RuntimeError("Cannot create a subregion of subregion")
        return Buffer(
            self.context, self.flags, size,
            _migrated_to=self._migrated_to, _offset=origin, _base_buffer=self)
