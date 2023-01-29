from abc import ABC, abstractmethod
from enum import Enum
from functools import lru_cache
from tempfile import mkdtemp
from typing import Tuple, List, Dict, Protocol, Sequence, Type, Optional, Any, Union, cast
import os.path
import weakref

import numpy

from .. import cuda_api_id
from ..utils import prod

from .mock_base import MockSource, MockKernel


class ConvertibleToInt(Protocol):
    def __int__(self) -> int:  # pragma: no cover
        ...


class MockPyCUDA:
    def __init__(self, cuda_version: Tuple[int, int, int] = (10, 0, 0)):
        self.pycuda_driver = Mock_pycuda_driver(self, cuda_version)
        self.pycuda_compiler = Mock_pycuda_compiler(self)

        self.device_infos: List[PyCUDADeviceInfo] = []
        self._context_stack: List[weakref.ReferenceType[BaseContext]] = []

        # Since we need to cast DeviceAllocation objects to integers (to add offsets),
        # there is no way to use a mock allocation object to track that.
        # Instead, we have to use recognizable integers as "addresses" and check the validity of
        # allocations using a kind of a fuzzy match database.
        # Should work for testing purposes as long as we use small offsets,
        # and other integer parameters don't fall in the "address" range.
        self._allocation_start = 2**30
        self._allocation_step = 2**16
        self._allocation_idx = 0
        self._allocations: Dict[int, Tuple[int, weakref.ReferenceType[BaseContext], bytes]] = {}

        self.api_id = cuda_api_id()

    def add_devices(self, device_infos: Sequence[Union[str, "PyCUDADeviceInfo"]]) -> None:
        assert len(self.device_infos) == 0
        for device_info in device_infos:
            if isinstance(device_info, str):
                device_info = PyCUDADeviceInfo(name=device_info)
            self.device_infos.append(device_info)

    def push_context(self, context: "BaseContext") -> None:
        if self.is_stacked(context):
            raise ValueError("The given context is already in the context stack")
        self._context_stack.append(weakref.ref(context))

    def pop_context(self) -> None:
        self._context_stack.pop()

    def current_context(self) -> "BaseContext":
        context = self._context_stack[-1]()
        assert context is not None
        return context

    def is_stacked(self, context: "BaseContext") -> bool:
        for stacked_context_ref in self._context_stack:
            if stacked_context_ref() == context:
                return True
        return False

    def allocate(self, size: int) -> Tuple[int, int]:
        assert size <= self._allocation_step
        idx = self._allocation_idx
        self._allocation_idx += 1
        address = self._allocation_start + self._allocation_step * idx
        self._allocations[idx] = (size, self._context_stack[-1], b"\xef" * size)
        return idx, address

    def get_allocation_buffer(self, idx: int, offset: int, region_size: int) -> bytes:
        size, context, buf = self._allocations[idx]
        return buf[offset : offset + region_size]

    def set_allocation_buffer(self, idx: int, offset: int, data: bytes) -> None:
        size, context, buf = self._allocations[idx]
        self._allocations[idx] = size, context, buf[:offset] + data + buf[offset + len(data) :]

    def allocation_count(self) -> int:
        return len(self._allocations)

    def free_allocation(self, idx: int) -> None:
        del self._allocations[idx]

    def check_allocation(self, address: ConvertibleToInt) -> Tuple[int, int, int]:
        # will work as long as we don't have offsets larger than `_allocation_step`
        idx = (int(address) - self._allocation_start) // self._allocation_step
        offset = (int(address) - self._allocation_start) % self._allocation_step

        if idx not in self._allocations:
            raise RuntimeError("A previously freed allocation or an incorrect address is used")

        size, context_ref, _buffer = self._allocations[idx]

        context = context_ref()
        assert context is not None

        if context != self.current_context():
            raise RuntimeError(
                "Trying to access an allocation from a device different to where it was created"
            )

        return idx, offset, size - offset


class PycudaCompileError(Exception):
    pass


class Mock_pycuda_compiler:
    def __init__(self, backend: MockPyCUDA):
        self.SourceModule = make_source_module_class(backend)


class BaseSourceModule(ABC):

    _context: "BaseContext"

    @abstractmethod
    def __init__(
        self,
        src: MockSource,
        no_extern_c: bool = False,
        options: Optional[Sequence[str]] = None,
        keep: bool = False,
        cache_dir: Optional[str] = None,
    ):
        ...

    @abstractmethod
    def test_get_options(self) -> Optional[Sequence[str]]:
        ...

    @abstractmethod
    def get_function(self, name: str) -> "BaseFunction":
        ...

    @abstractmethod
    def get_global(self, name: str) -> Tuple["BaseDeviceAllocation", int]:
        ...


@lru_cache()
def make_source_module_class(backend: MockPyCUDA) -> Type[BaseSourceModule]:

    backend_ref = weakref.ref(backend)

    class SourceModule(BaseSourceModule):

        _backend_ref = backend_ref

        def __init__(
            self,
            src: MockSource,
            no_extern_c: bool = False,
            options: Optional[Sequence[str]] = None,
            keep: bool = False,
            cache_dir: Optional[str] = None,
        ):
            assert isinstance(src, MockSource)
            assert isinstance(no_extern_c, bool)
            assert options is None or all(isinstance(option, str) for option in options)
            assert isinstance(keep, bool)

            if src.should_fail:
                raise PycudaCompileError()

            backend = self._backend_ref()
            assert backend is not None
            self._context = backend.current_context()

            function_cls = backend.pycuda_driver.Function
            self._kernels = {kernel.name: function_cls(self, kernel) for kernel in src.kernels}
            self._constant_mem = src.constant_mem

            self._options = options

            # See the note in compile_single_device(). Apparently that's how PyCUDA operates.
            if keep and cache_dir is not None:
                temp_dir = mkdtemp()
                with open(os.path.join(temp_dir, "kernel.cu"), "w") as f:
                    f.write(str(src))
                print(f"*** compiler output in {temp_dir}")

        def test_get_options(self) -> Optional[Sequence[str]]:
            return self._options

        def get_function(self, name: str) -> "BaseFunction":
            return self._kernels[name]

        def get_global(self, name: str) -> Tuple["BaseDeviceAllocation", int]:
            size = self._constant_mem[name]
            backend = self._backend_ref()
            assert backend is not None
            alloc = backend.pycuda_driver.DeviceAllocation._allocate(size)
            return alloc, size

    return SourceModule


class MemAttachFlags(Enum):
    GLOBAL = 1


class FunctionAttribute(Enum):
    MAX_THREADS_PER_BLOCK = 0


class Mock_pycuda_driver:

    Function: Type["BaseFunction"]

    DeviceAllocation: Type["BaseDeviceAllocation"]

    def __init__(self, backend: MockPyCUDA, cuda_version: Tuple[int, int, int]):

        self._backend_ref = weakref.ref(backend)
        self._version = cuda_version

        self.Device = make_device_class(backend)
        self.Stream = make_stream_class(backend)
        self.Context = make_context_class(backend)
        self.DeviceAllocation = make_device_allocation_class(backend)
        self.Function = make_function_class(backend)

        self.CompileError = PycudaCompileError

        self.mem_attach_flags = MemAttachFlags
        self.function_attribute = FunctionAttribute

    def get_version(self) -> Tuple[int, int, int]:
        return self._version

    def mem_alloc(self, size: int) -> "BaseDeviceAllocation":
        return self.DeviceAllocation._allocate(size)

    def memcpy_htod(
        self, dest: "BaseDeviceAllocation", src: "numpy.ndarray[Any, numpy.dtype[Any]]"
    ) -> None:
        self.memcpy_htod_async(dest, src)

    def memcpy_htod_async(
        self,
        dest: "BaseDeviceAllocation",
        src: "numpy.ndarray[Any, numpy.dtype[Any]]",
        stream: Optional["BaseStream"] = None,
    ) -> None:
        backend = self._backend_ref()
        assert backend is not None
        current_context = backend.current_context()
        assert isinstance(src, numpy.ndarray)
        dest = self.DeviceAllocation._from_memcpy_arg(dest)
        if stream is not None:
            assert stream._context == current_context
        assert dest._size >= src.size * src.dtype.itemsize
        dest._set(src)

    def memcpy_dtoh(
        self, dest: "numpy.ndarray[Any, numpy.dtype[Any]]", src: "BaseDeviceAllocation"
    ) -> None:
        self.memcpy_dtoh_async(dest, src)

    def memcpy_dtoh_async(
        self,
        dest: "numpy.ndarray[Any, numpy.dtype[Any]]",
        src: "BaseDeviceAllocation",
        stream: Optional["BaseStream"] = None,
    ) -> None:
        backend = self._backend_ref()
        assert backend is not None
        current_context = backend.current_context()
        assert isinstance(dest, numpy.ndarray)
        src = self.DeviceAllocation._from_memcpy_arg(src)
        if stream is not None:
            assert stream._context == current_context
        assert src._size >= dest.size * dest.dtype.itemsize
        src._get(dest)

    def memcpy_dtod(
        self, dest: "BaseDeviceAllocation", src: "BaseDeviceAllocation", size: int
    ) -> None:
        self.memcpy_dtod_async(dest, src, size)

    def memcpy_dtod_async(
        self,
        dest: "BaseDeviceAllocation",
        src: "BaseDeviceAllocation",
        size: int,
        stream: Optional["BaseStream"] = None,
    ) -> None:
        backend = self._backend_ref()
        assert backend is not None
        current_context = backend.current_context()
        dest = self.DeviceAllocation._from_memcpy_arg(dest)
        src = self.DeviceAllocation._from_memcpy_arg(src)
        if stream is not None:
            assert stream._context == current_context
        assert dest._size >= size
        assert src._size >= size
        dest._set(src)

    def pagelocked_empty(
        self, shape: Sequence[int], dtype: "numpy.dtype[Any]"
    ) -> "numpy.ndarray[Any, numpy.dtype[Any]]":
        return numpy.empty(shape, dtype)


class PyCUDADeviceInfo:
    def __init__(
        self,
        name: str = "DefaultDeviceName",
        max_threads_per_block: int = 1024,
        max_block_dim_x: int = 1024,
        max_block_dim_y: int = 1024,
        max_block_dim_z: int = 64,
        max_grid_dim_x: int = 2**32 - 1,
        max_grid_dim_y: int = 2**32 - 1,
        max_grid_dim_z: int = 65536,
        warp_size: int = 32,
        max_shared_memory_per_block: int = 64 * 1024,
        multiprocessor_count: int = 8,
        compute_capability: int = 5,
    ):
        self.name = name
        self.max_threads_per_block = max_threads_per_block
        self.max_block_dim_x = max_block_dim_x
        self.max_block_dim_y = max_block_dim_y
        self.max_block_dim_z = max_block_dim_z
        self.max_grid_dim_x = max_grid_dim_x
        self.max_grid_dim_y = max_grid_dim_y
        self.max_grid_dim_z = max_grid_dim_z
        self.warp_size = warp_size
        self.max_shared_memory_per_block = max_shared_memory_per_block
        self.multiprocessor_count = multiprocessor_count
        self.compute_capability = compute_capability


class BaseDevice(ABC):

    max_grid_dim_x: int
    max_grid_dim_y: int
    max_grid_dim_z: int
    max_block_dim_x: int
    max_block_dim_y: int
    max_block_dim_z: int

    @abstractmethod
    def __init__(self, device_idx: int):
        ...

    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def compute_capability(self) -> Tuple[int, int]:
        ...

    @abstractmethod
    def make_context(self) -> "BaseContext":
        ...

    @staticmethod
    @abstractmethod
    def count() -> int:
        ...


# We need a device class that is an actual class
# (so that `type()` works on the results of its `__call__()`),
# but at the same time retains a reference to the backend object used to create it
# (so that we can control the properties of the mock).
@lru_cache()
def make_device_class(backend: MockPyCUDA) -> Type[BaseDevice]:

    backend_ref = weakref.ref(backend)

    class Device(BaseDevice):

        _backend_ref = backend_ref

        def __init__(self, device_idx: int):
            backend = Device._backend_ref()
            assert backend is not None
            device_info = backend.device_infos[device_idx]

            self._device_idx = device_idx
            self._name = device_info.name
            self.max_threads_per_block = device_info.max_threads_per_block

            self.max_block_dim_x = device_info.max_block_dim_x
            self.max_block_dim_y = device_info.max_block_dim_y
            self.max_block_dim_z = device_info.max_block_dim_z

            self.max_grid_dim_x = device_info.max_grid_dim_x
            self.max_grid_dim_y = device_info.max_grid_dim_y
            self.max_grid_dim_z = device_info.max_grid_dim_z

            self.warp_size = device_info.warp_size
            self.max_shared_memory_per_block = device_info.max_shared_memory_per_block
            self.multiprocessor_count = device_info.multiprocessor_count

            self._compute_capability = device_info.compute_capability

        def name(self) -> str:
            return self._name

        def compute_capability(self) -> Tuple[int, int]:
            return (self._compute_capability, 0)

        def make_context(self) -> BaseContext:
            backend = self._backend_ref()
            assert backend is not None
            context = backend.pycuda_driver.Context(self._device_idx)
            context.push()
            return context

        def __eq__(self, other: Any) -> bool:
            assert isinstance(other, Device)
            return self._device_idx == other._device_idx

        def __hash__(self) -> int:
            return hash((type(self), self._device_idx))

        @staticmethod
        def count() -> int:
            backend = Device._backend_ref()
            assert backend is not None
            return len(backend.device_infos)

    return Device


class BaseContext(ABC):

    _device_idx: int

    @abstractmethod
    def __init__(self, device_idx: int):
        ...

    @staticmethod
    @abstractmethod
    def pop() -> None:
        ...

    @staticmethod
    @abstractmethod
    def get_device() -> BaseDevice:
        ...

    @abstractmethod
    def push(self) -> None:
        ...

    @abstractmethod
    def is_stacked(self) -> bool:
        ...


@lru_cache()
def make_context_class(backend: MockPyCUDA) -> Type[BaseContext]:
    class Context(BaseContext):

        # Since we need the backend in __del__(),
        # we want to make sure that it is alive as long as a this object is alive.
        _backend = backend

        def __init__(self, device_idx: int):
            self._device_idx = device_idx

        @staticmethod
        def pop() -> None:
            Context._backend.pop_context()

        @staticmethod
        def get_device() -> BaseDevice:
            backend = Context._backend
            return backend.pycuda_driver.Device(backend.current_context()._device_idx)

        def push(self) -> None:
            self._backend.push_context(self)

        def is_stacked(self) -> bool:
            return self._backend.is_stacked(self)

        def __del__(self) -> None:
            # Sanity check
            message = "Context was deleted while still being in the context stack"
            assert not self.is_stacked(), message

    return Context


class BaseStream(ABC):

    _context: BaseContext

    @abstractmethod
    def synchronize(self) -> None:
        ...


@lru_cache()
def make_stream_class(backend: MockPyCUDA) -> Type[BaseStream]:

    backend_ref = weakref.ref(backend)

    class Stream(BaseStream):

        _backend_ref = backend_ref

        def __init__(self) -> None:
            backend = self._backend_ref()
            assert backend is not None
            self._context = backend.current_context()

        def synchronize(self) -> None:
            backend = self._backend_ref()
            assert backend is not None
            assert self._context == backend.current_context()

    return Stream


class BaseDeviceAllocation(ABC):

    _context: BaseContext
    _idx: int
    _offset: int
    _size: int

    @classmethod
    @abstractmethod
    def _allocate(cls, size: int) -> "BaseDeviceAllocation":
        ...

    @classmethod
    @abstractmethod
    def _from_memcpy_arg(cls, arg: Union["BaseDeviceAllocation", int]) -> "BaseDeviceAllocation":
        ...

    @classmethod
    @abstractmethod
    def _from_kernel_arg(
        cls, arg: Union["BaseDeviceAllocation", numpy.uintp]
    ) -> "BaseDeviceAllocation":
        ...

    @abstractmethod
    def __init__(self, idx: int, address: int, offset: int, size: int, owns_buffer: bool = False):
        ...

    @abstractmethod
    def _set(
        self, arr: Union["numpy.ndarray[Any, numpy.dtype[Any]]", "BaseDeviceAllocation"]
    ) -> None:
        ...

    @abstractmethod
    def _get(self, arr: "numpy.ndarray[Any, numpy.dtype[Any]]") -> None:
        ...

    @abstractmethod
    def __int__(self) -> int:
        ...


@lru_cache()
def make_device_allocation_class(backend: MockPyCUDA) -> Type[BaseDeviceAllocation]:

    backend_ref = weakref.ref(backend)

    class DeviceAllocation(BaseDeviceAllocation):

        _backend_ref = backend_ref

        @classmethod
        def _allocate(cls, size: int) -> "DeviceAllocation":
            idx, address = backend.allocate(size)
            return cls(idx, address, 0, size, owns_buffer=True)

        @classmethod
        def _from_memcpy_arg(cls, arg: Union["BaseDeviceAllocation", int]) -> "DeviceAllocation":
            # `memcpy` functions in PyCUDA require pointers to be `int`s,
            # and kernels require them to be `numpy.number`s. Go figure.
            assert isinstance(arg, (cls, int))
            if isinstance(arg, cls):
                return arg
            arg = cast(int, arg)  # mypy is not smart enough to derive that

            backend = cls._backend_ref()
            assert backend is not None
            # Unfortunately we can't keep track of subregion size in PyCUDA,
            # so for the size we just choose the maximum available.
            idx, offset, size = backend.check_allocation(arg)
            return cls(idx, arg, offset, size, owns_buffer=False)

        @classmethod
        def _from_kernel_arg(
            cls, arg: Union["BaseDeviceAllocation", numpy.uintp]
        ) -> "DeviceAllocation":
            assert isinstance(arg, (cls, numpy.uintp))
            return cls._from_memcpy_arg(int(arg))

        def __init__(
            self, idx: int, address: int, offset: int, size: int, owns_buffer: bool = False
        ):
            backend = self._backend_ref()
            assert backend is not None
            self._context = backend.current_context()
            self._address = address
            self._idx = idx
            self._offset = offset
            self._size = size
            self._owns_buffer = owns_buffer

        def _set(
            self, arr: Union["numpy.ndarray[Any, numpy.dtype[Any]]", "BaseDeviceAllocation"]
        ) -> None:
            backend = self._backend_ref()
            assert backend is not None
            if isinstance(arr, numpy.ndarray):
                data = arr.tobytes()
            else:
                data = backend.get_allocation_buffer(arr._idx, arr._offset, arr._size)
            assert len(data) <= self._size
            backend.set_allocation_buffer(self._idx, self._offset, data)

        def _get(self, arr: "numpy.ndarray[Any, numpy.dtype[Any]]") -> None:
            backend = self._backend_ref()
            assert backend is not None
            data = arr.tobytes()
            assert len(data) <= self._size
            buf = backend.get_allocation_buffer(self._idx, self._offset, len(data))
            buf_as_arr = numpy.frombuffer(buf, arr.dtype).reshape(arr.shape)
            numpy.copyto(arr, buf_as_arr)

        def __int__(self) -> int:
            return self._address

        def __del__(self) -> None:
            # Backend is held alive by the context object we reference.
            if self._owns_buffer:
                backend = self._backend_ref()
                assert backend is not None
                backend.free_allocation(self._idx)

    return DeviceAllocation


class BaseFunction(ABC):
    @abstractmethod
    def __init__(self, source_module: BaseSourceModule, kernel: MockKernel):
        ...

    @abstractmethod
    def __call__(
        self,
        *args: Union[BaseDeviceAllocation, numpy.generic],
        grid: Tuple[int, int, int],
        block: Tuple[int, int, int],
        stream: BaseStream,
        shared: Optional[int] = None,
    ) -> None:
        ...

    @abstractmethod
    def get_attribute(self, attribute: FunctionAttribute) -> Tuple[int, ...]:
        ...


@lru_cache()
def make_function_class(backend: MockPyCUDA) -> Type[BaseFunction]:

    backend_ref = weakref.ref(backend)

    class Function(BaseFunction):

        _backend_ref = backend_ref

        def __init__(self, source_module: BaseSourceModule, kernel: MockKernel):
            self._kernel = kernel
            self._source_module = source_module

        def __call__(
            self,
            *args: Union[BaseDeviceAllocation, numpy.generic],
            grid: Tuple[int, int, int],
            block: Tuple[int, int, int],
            stream: BaseStream,
            shared: Optional[int] = None,
        ) -> None:
            backend = self._backend_ref()
            assert backend is not None

            current_context = backend.current_context()

            assert self._source_module._context == current_context

            assert isinstance(stream, backend.pycuda_driver.Stream)
            assert stream._context == current_context

            for arg, param in zip(args, self._kernel.parameters):
                if param is None:
                    # Checks validity on creation
                    assert isinstance(arg, numpy.uintp)
                    buf = backend.pycuda_driver.DeviceAllocation._from_kernel_arg(arg)
                    assert buf._context == current_context
                elif isinstance(arg, numpy.generic):
                    assert arg.dtype == param
                else:
                    raise TypeError(f"Incorrect argument type: {type(arg)}")

            device = current_context.get_device()
            max_grid = [device.max_grid_dim_x, device.max_grid_dim_y, device.max_grid_dim_z]
            max_block = [device.max_block_dim_x, device.max_block_dim_y, device.max_block_dim_z]

            assert isinstance(grid, tuple)
            assert len(grid) == 3
            assert all(isinstance(x, int) for x in grid)
            assert all(g <= max_g for g, max_g in zip(grid, max_grid))

            assert isinstance(block, tuple)
            assert len(block) == 3
            assert all(isinstance(x, int) for x in block)
            assert all(b <= max_b for b, max_b in zip(block, max_block))

        def get_attribute(self, attribute: FunctionAttribute) -> Tuple[int, ...]:
            if attribute == FunctionAttribute.MAX_THREADS_PER_BLOCK:
                device_idx = self._source_module._context._device_idx
                return self._kernel.max_total_local_sizes[device_idx]
            else:  # pragma: no cover
                raise NotImplementedError(f"Unknown attribute: {attribute}")

    return Function
