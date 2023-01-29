from enum import Enum, IntEnum
from typing import Sequence, Optional, Union, List, Any, Dict, Tuple, cast
import weakref

import numpy

from .. import opencl_api_id
from ..adapter_base import DeviceType

from .mock_base import MockSource, MockKernel


class MemFlags(IntEnum):
    READ_WRITE = 1
    ALLOC_HOST_PTR = 16


class MemMigrationFlags(IntEnum):
    CONTENT_UNDEFINED = 2


class MockPyOpenCL:
    def __init__(self) -> None:
        self.pyopencl = Mock_pyopencl(self)
        self.api_id = opencl_api_id()
        self.platforms: List["Platform"] = []

    def add_platform(self, platform_name: Optional[str] = None) -> "Platform":
        if platform_name is None:
            platform_name = "Platform" + str(len(self.platforms))
        platform = Platform(self, platform_name)
        self.platforms.append(platform)
        return platform

    def add_platform_with_devices(
        self, platform_name: Optional[str], device_infos: Sequence[Union["PyOpenCLDeviceInfo", str]]
    ) -> "Platform":
        platform = self.add_platform(platform_name)
        for device_info in device_infos:
            if isinstance(device_info, str):
                device_info = PyOpenCLDeviceInfo(name=device_info)
            platform.add_device(device_info)
        return platform

    def add_devices(self, device_infos: Sequence["PyOpenCLDeviceInfo"]) -> "Platform":
        # Prevent incorrect usage - this method is added to be similar to that of PyCUDA mock,
        # so it can only be used once.
        assert len(self.platforms) == 0
        return self.add_platform_with_devices(None, device_infos)


class PyopenclRuntimeError(Exception):
    pass


class KernelWorkGroupInfo(Enum):
    WORK_GROUP_SIZE = 4528


class Mock_pyopencl:
    def __init__(self, backend: MockPyOpenCL):

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
        self.mem_migration_flags = MemMigrationFlags
        self.kernel_work_group_info = KernelWorkGroupInfo

    def get_platforms(self) -> List["Platform"]:
        backend = self._backend_ref()
        assert backend is not None
        return backend.platforms

    def enqueue_copy(
        self,
        queue: "CommandQueue",
        dest: Union["Buffer", "numpy.ndarray[Any, numpy.dtype[Any]]"],
        src: Union["Buffer", "numpy.ndarray[Any, numpy.dtype[Any]]"],
        is_blocking: bool = False,
    ) -> None:
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

        assert isinstance(dest, Buffer) or isinstance(src, Buffer)
        if isinstance(dest, Buffer):
            dest._set(src)
        else:
            # mypy is not smart enough to combine the `assert` above with the condition of the `if`
            src = cast(Buffer, src)
            src._get(dest)


class Platform:
    def __init__(self, backend: MockPyOpenCL, name: str):
        self._backend_ref = weakref.ref(backend)
        self.name = name
        self._devices: List["Device"] = []
        self.vendor = "Mock Platforms"
        self.version = "OpenCL 1.2"

    def add_device(self, device_info: "PyOpenCLDeviceInfo") -> None:
        device = Device(self, device_info)
        self._devices.append(device)

    def get_devices(self) -> List["Device"]:
        return self._devices


class PyOpenCLDeviceInfo:
    def __init__(
        self,
        name: str = "DefaultDeviceName",
        vendor: str = "Mock Devices",
        type: DeviceType = DeviceType.GPU,
        max_work_group_size: int = 1024,
        max_work_item_sizes: Sequence[int] = (1024, 1024, 1024),
        local_mem_size: int = 64 * 1024,
        address_bits: int = 32,
        max_compute_units: int = 8,
        extensions: Sequence[str] = (),
        compute_capability_major_nv: Optional[int] = None,
        warp_size_nv: Optional[int] = None,
    ):
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
    def __init__(self, platform: Platform, device_info: PyOpenCLDeviceInfo):

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
    def platform(self) -> Platform:
        platform = self._platform_ref()
        assert platform is not None
        return platform


class Context:
    def __init__(self, devices: Sequence[Device]):
        self._backend_ref = devices[0]._backend_ref
        self.devices = devices


class CommandQueue:
    def __init__(self, context: Context, device: Device):
        self._backend_ref = context._backend_ref
        self.context = context
        assert device in context.devices
        self.device = device

    def finish(self) -> None:
        pass


class Program:
    def __init__(self, context: Context, src: MockSource):
        self._backend_ref = context._backend_ref
        self.context = context
        self.src = src
        self._kernels: Dict[str, "Kernel"] = {}
        self._options: List[str] = []

    def build(
        self,
        options: Sequence[str] = [],
        devices: Optional[Sequence[Device]] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        assert isinstance(self.src, MockSource)
        assert all(isinstance(option, str) for option in options)
        assert cache_dir is None or isinstance(cache_dir, str)

        # In Grunnur, we always build separate programs for each device
        assert devices is not None
        assert len(devices) == 1 and devices[0] in self.context.devices

        if self.src.should_fail:
            raise PyopenclRuntimeError()

        self._options = list(options)
        self._kernels = {kernel.name: Kernel(self, kernel) for kernel in self.src.kernels}

    def test_get_options(self) -> List[str]:
        return self._options

    def __getattr__(self, name: str) -> "Kernel":
        return self._kernels[name]


class Kernel:
    def __init__(self, program: Program, kernel: MockKernel):
        self.program = program
        self._kernel = kernel

    def __call__(
        self,
        queue: CommandQueue,
        global_size: Tuple[int, ...],
        local_size: Optional[Tuple[int, ...]],
        *args: Union["Buffer", numpy.generic],
    ) -> None:
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
                assert arg._migrated_to is None or arg._migrated_to == queue.device
            elif isinstance(arg, numpy.generic):
                assert arg.dtype == param
            else:
                raise TypeError(f"Incorrect argument type: {type(arg)}")

    def get_work_group_info(
        self, attribute: KernelWorkGroupInfo, device: Device
    ) -> Tuple[int, ...]:
        if attribute == KernelWorkGroupInfo.WORK_GROUP_SIZE:
            device_idx = device.platform.get_devices().index(device)
            return self._kernel.max_total_local_sizes[device_idx]
        else:  # pragma: no cover
            raise NotImplementedError(f"Unknown attribute: {attribute}")


def insert_data(buf: bytes, offset: int, data: bytes) -> bytes:
    return buf[:offset] + data + buf[offset + len(data) :]


class Buffer:
    def __init__(
        self,
        context: Context,
        flags: Any,
        size: int,
        _migrated_to: Optional[Device] = None,
        _offset: int = 0,
        _base_buffer: Optional["Buffer"] = None,
    ):
        self.context = context
        self.flags = flags
        self.size = size
        self.offset = _offset
        self._migrated_to = _migrated_to

        if _base_buffer is None:
            self._buffer = b"\xef" * size
        self._base_buffer = _base_buffer

    def _set(self, arr: Union["Buffer", "numpy.ndarray[Any, numpy.dtype[Any]]"]) -> None:
        if isinstance(arr, numpy.ndarray):
            data = arr.tobytes()
        else:
            full_buf = arr._buffer if arr._base_buffer is None else arr._base_buffer._buffer
            data = full_buf[arr.offset : arr.offset + arr.size]

        assert len(data) <= self.size

        if self._base_buffer is None:
            self._buffer = insert_data(self._buffer, self.offset, data)
        else:
            self._base_buffer._buffer = insert_data(self._base_buffer._buffer, self.offset, data)

    def _get(self, arr: "numpy.ndarray[Any, numpy.dtype[Any]]") -> None:
        data = arr.tobytes()
        assert len(data) <= self.size

        full_buf = self._buffer if self._base_buffer is None else self._base_buffer._buffer

        buf = full_buf[self.offset : self.offset + len(data)]
        buf_as_arr = numpy.frombuffer(buf, arr.dtype).reshape(arr.shape)
        numpy.copyto(arr, buf_as_arr)

    def _access_from(self, device: Device) -> None:
        if self._migrated_to is None:
            self._migrated_to = device
        elif device != self._migrated_to:
            raise RuntimeError(
                "Trying to access a buffer from a device different to where it was migrated to"
            )

    def get_sub_region(self, origin: int, size: int) -> "Buffer":
        assert origin + size <= self.size
        # Trying to do that in PyOpenCL leads to segfault
        if self._base_buffer is not None:
            raise RuntimeError("Cannot create a subregion of subregion")
        return Buffer(
            self.context,
            self.flags,
            size,
            _migrated_to=self._migrated_to,
            _offset=origin,
            _base_buffer=self,
        )
