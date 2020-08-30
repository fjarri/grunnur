from enum import Enum
from functools import lru_cache
from tempfile import mkdtemp
import os.path
import weakref

import numpy

from grunnur import cuda_api_id
from grunnur.dtypes import normalize_type
from grunnur.utils import prod, wrap_in_tuple

from .mock_base import MockSource


class MockPyCUDA:

    def __init__(self, cuda_version=(10, 0, 0)):
        self.pycuda_driver = Mock_pycuda_driver(self, cuda_version)
        self.pycuda_compiler = Mock_pycuda_compiler(self)

        self.device_infos = []
        self._context_stack = []

        # Since we need to cast DeviceAllocation objects to integers (to add offsets),
        # there is no way to use a mock allocation object to track that.
        # Instead, we have to use recognizable integers as "addresses" and check the validity of
        # allocations using a kind of a fuzzy match database.
        # Should work for testing purposes as long as we use small offsets,
        # and other integer parameters don't fall in the "address" range.
        self._allocation_start = 2**30
        self._allocation_step = 2**16
        self._allocation_idx = 0
        self._allocations = {}

        self.api_id = cuda_api_id()

    def add_devices(self, device_infos):
        assert len(self.device_infos) == 0
        for device_info in device_infos:
            if isinstance(device_info, str):
                device_info = PyCUDADeviceInfo(name=device_info)
            elif isinstance(device_info, PyCUDADeviceInfo):
                pass
            else:
                raise TypeError(type(device_info))
            self.device_infos.append(device_info)

    def push_context(self, context):
        if self.is_stacked(context):
            raise ValueError("The given context is already in the context stack")
        self._context_stack.append(weakref.ref(context))

    def pop_context(self):
        self._context_stack.pop()

    def current_context(self):
        return self._context_stack[-1]()

    def is_stacked(self, context):
        for stacked_context_ref in self._context_stack:
            if stacked_context_ref() == context:
                return True
        return False

    def allocate(self, size, managed=False):
        assert size <= self._allocation_step
        idx = self._allocation_idx
        self._allocation_idx += 1
        address = self._allocation_start + self._allocation_step * idx
        self._allocations[idx] = (size, None if managed else self._context_stack[-1], b"\xef" * size)
        return idx, address

    def get_allocation_buffer(self, idx, offset, region_size):
        size, context, buf = self._allocations[idx]
        return buf[offset:offset+region_size]

    def set_allocation_buffer(self, idx, offset, data):
        size, context, buf = self._allocations[idx]
        self._allocations[idx] = size, context, buf[:offset] + data + buf[offset+len(data):]

    def allocation_count(self):
        return len(self._allocations)

    def free_allocation(self, idx):
        del self._allocations[idx]

    def check_allocation(self, address):
        # will work as long as we don't have offsets larger than `_allocation_step`
        idx = (int(address) - self._allocation_start) // self._allocation_step
        offset = (int(address) - self._allocation_start) % self._allocation_step

        if idx not in self._allocations:
            raise RuntimeError("A previously freed allocation or an incorrect address is used")

        size, context_ref, _buffer = self._allocations[idx]

        if context_ref is not None:
            assert context_ref() == self.current_context()

        return idx, offset, size - offset


class PycudaCompileError(Exception):
    pass


class Mock_pycuda_compiler():

    def __init__(self, backend):
        self.SourceModule = make_source_module_class(backend)


@lru_cache()
def make_source_module_class(backend):

    backend_ref = weakref.ref(backend)

    class SourceModule:

        _backend_ref = backend_ref

        def __init__(self, src, no_extern_c=False, options=None, keep=False, cache_dir=None):
            assert isinstance(src, MockSource)
            assert isinstance(no_extern_c, bool)
            assert options is None or all(isinstance(option, str) for option in options)
            assert isinstance(keep, bool)

            if src.should_fail:
                raise PycudaCompileError()

            self._context = self._backend_ref().current_context()

            function_cls = self._backend_ref().pycuda_driver.Function
            self._kernels = {kernel.name: function_cls(self, kernel) for kernel in src.kernels}
            self._constant_mem = src.constant_mem

            # See the note in compile_single_device(). Apparently that's how PyCUDA operates.
            if keep and cache_dir is not None:
                temp_dir = mkdtemp()
                with open(os.path.join(temp_dir, 'kernel.cu'), 'w') as f:
                    f.write(str(src))
                print(f"*** compiler output in {temp_dir}")

        def get_function(self, name):
            return self._kernels[name]

        def get_global(self, name):
            size = self._constant_mem[name]
            alloc = self._backend_ref().pycuda_driver.DeviceAllocation._allocate(size)
            return alloc, size


    return SourceModule


class MemAttachFlags(Enum):
    GLOBAL = 1


class FunctionAttribute(Enum):
    MAX_THREADS_PER_BLOCK = 0


class Mock_pycuda_driver:

    def __init__(self, backend, cuda_version):

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

    def get_version(self):
        return self._version

    def init(self):
        pass

    def mem_alloc(self, size, _managed=False):
        return self.DeviceAllocation._allocate(size, _managed=_managed)

    def managed_empty(self, shape, dtype, mem_flags=None):
        size = prod(wrap_in_tuple(shape)) * normalize_type(dtype).itemsize
        class _mock_array:
            base = self.mem_alloc(size, _managed=True)
        return _mock_array()

    def memcpy_htod(self, dest, src):
        self.memcpy_htod_async(dest, src)

    def memcpy_htod_async(self, dest, src, stream=None):
        current_context = self._backend_ref().current_context()
        assert isinstance(src, numpy.ndarray)
        dest = self.DeviceAllocation._from_memcpy_arg(dest)
        if stream is not None:
            assert stream._context == current_context
        assert dest._size >= src.size * src.dtype.itemsize
        dest._set(src)

    def memcpy_dtoh(self, dest, src):
        self.memcpy_dtoh_async(dest, src)

    def memcpy_dtoh_async(self, dest, src, stream=None):
        current_context = self._backend_ref().current_context()
        assert isinstance(dest, numpy.ndarray)
        src = self.DeviceAllocation._from_memcpy_arg(src)
        if stream is not None:
            assert stream._context == current_context
        assert src._size >= dest.size * dest.dtype.itemsize
        src._get(dest)

    def memcpy_dtod(self, dest, src, size):
        self.memcpy_dtod_async(dest, src, size)

    def memcpy_dtod_async(self, dest, src, size, stream=None):
        current_context = self._backend_ref().current_context()
        dest = self.DeviceAllocation._from_memcpy_arg(dest)
        src = self.DeviceAllocation._from_memcpy_arg(src)
        if stream is not None:
            assert stream._context == current_context
        assert dest._size >= size
        assert src._size >= size
        dest._set(src)

    def pagelocked_empty(self, shape, dtype):
        return numpy.empty(shape, dtype)


class PyCUDADeviceInfo:

    def __init__(
            self,
            name="DefaultDeviceName",
            max_threads_per_block=1024,
            max_block_dim_x=1024,
            max_block_dim_y=1024,
            max_block_dim_z=64,
            max_grid_dim_x=2**32-1,
            max_grid_dim_y=2**32-1,
            max_grid_dim_z=65536,
            warp_size=32,
            max_shared_memory_per_block=64*1024,
            multiprocessor_count=8,
            compute_capability=5):
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


# We need a device class that is an actual class
# (so that `type()` works on the results of its `__call__()`),
# but at the same time retains a reference to the backend object used to create it
# (so that we can control the properties of the mock).
@lru_cache()
def make_device_class(backend):

    backend_ref = weakref.ref(backend)

    class Device:

        _backend_ref = backend_ref

        def __init__(self, device_idx):

            device_info = Device._backend_ref().device_infos[device_idx]

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

        def name(self):
            return self._name

        def compute_capability(self):
            return (self._compute_capability, 0)

        def make_context(self):
            context = self._backend_ref().pycuda_driver.Context(self._device_idx)
            context.push()
            return context

        def __eq__(self, other):
            return self._device_idx == other._device_idx

        def __hash__(self):
            return hash((type(self), self._device_idx))

        @staticmethod
        def count():
            return len(Device._backend_ref().device_infos)

    return Device


@lru_cache()
def make_context_class(backend):

    class Context:

        # Since we need the backend in __del__(),
        # we want to make sure that it alive as long as a this object is alive.
        _backend = backend

        def __init__(self, device_idx):
            self._device_idx = device_idx

        @staticmethod
        def pop():
            Context._backend.pop_context()

        @staticmethod
        def get_device():
            backend = Context._backend
            return backend.pycuda_driver.Device(backend.current_context()._device_idx)

        def push(self):
            self._backend.push_context(self)

        def __del__(self):
            if self._backend.is_stacked(self):
                raise RuntimeError("Context was deleted while still being in the context stack")

    return Context


@lru_cache()
def make_stream_class(backend):

    backend_ref = weakref.ref(backend)

    class Stream:

        _backend_ref = backend_ref

        def __init__(self):
            self._context = self._backend_ref().current_context()

        def synchronize(self):
            assert self._context == self._backend_ref().current_context()


    return Stream


@lru_cache()
def make_device_allocation_class(backend):

    backend_ref = weakref.ref(backend)

    class DeviceAllocation:

        _backend_ref = backend_ref

        @classmethod
        def _allocate(cls, size, _managed=True):
            idx, address = backend.allocate(size, managed=_managed)
            return cls(idx, address, 0, size, owns_buffer=True)

        @classmethod
        def _from_memcpy_arg(cls, arg):
            # `memcpy` functions in PyCUDA require pointers to be `int`s,
            # and kernels require them to be `numpy.number`s. Go figure.
            if isinstance(arg, cls):
                return arg
            elif isinstance(arg, int):
                # Unfortunately we can't keep track of subregion size in PyCUDA,
                # so for the size we just choose the maximum available.
                idx, offset, size = cls._backend_ref().check_allocation(arg)
                return cls(idx, arg, offset, size, owns_buffer=False)
            else:
                raise TypeError(type(arg))

        @classmethod
        def _from_kernel_arg(cls, arg):
            if isinstance(arg, (cls, numpy.uintp)):
                return cls._from_memcpy_arg(int(arg))
            else:
                raise TypeError(type(kernel_arg))

        def __init__(self, idx, address, offset, size, owns_buffer=False):
            self._context = self._backend_ref().current_context()
            self._address = address
            self._idx = idx
            self._offset = offset
            self._size = size
            self._owns_buffer = owns_buffer

        def _set(self, arr):
            if isinstance(arr, numpy.ndarray):
                data = arr.tobytes()
            else:
                data = self._backend_ref().get_allocation_buffer(arr._idx, arr._offset, arr._size)
            assert len(data) <= self._size
            self._backend_ref().set_allocation_buffer(self._idx, self._offset, data)

        def _get(self, arr):
            data = arr.tobytes()
            assert len(data) <= self._size
            buf = self._backend_ref().get_allocation_buffer(self._idx, self._offset, len(data))
            buf_as_arr = numpy.frombuffer(buf, arr.dtype).reshape(arr.shape)
            numpy.copyto(arr, buf_as_arr)

        def __int__(self):
            return self._address

        def __del__(self):
            # Backend is held alive by the context object we reference.
            if self._owns_buffer:
                self._backend_ref().free_allocation(self._idx)


    return DeviceAllocation


@lru_cache()
def make_function_class(backend):

    backend_ref = weakref.ref(backend)

    class Function:

        _backend_ref = backend_ref

        def __init__(self, source_module, kernel):
            self._kernel = kernel
            self._source_module = source_module

        def __call__(self, *args, grid=None, block=None, stream=None, shared=None):
            backend = self._backend_ref()

            current_context = backend.current_context()

            assert self._source_module._context == current_context

            if stream is not None:
                assert isinstance(stream, backend.pycuda_driver.Stream)
                assert stream._context == current_context

            for arg, param in zip(args, self._kernel.parameters):
                if param is None:
                    # Checks validity on creation
                    backend.pycuda_driver.DeviceAllocation._from_kernel_arg(arg)
                elif isinstance(arg, numpy.number):
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

        def get_attribute(self, attribute):
            if attribute == FunctionAttribute.MAX_THREADS_PER_BLOCK:
                device_idx = self._source_module._context._device_idx
                return self._kernel.max_total_local_sizes[device_idx]
            else:
                raise ValueError(f"Unknown attribute: {attribute}")

    return Function
