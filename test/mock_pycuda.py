from enum import Enum
from functools import lru_cache
import weakref

import numpy

from grunnur import CUDA_API_ID
from grunnur.dtypes import normalize_type
from grunnur.utils import prod, wrap_in_tuple

from .mock_base import MockSourceStr, DeviceInfo


class MockPyCUDA:

    def __init__(self, cuda_version="0.0"):
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
        self._allocations = []

        self.api_id = CUDA_API_ID

    def add_devices(self, device_infos):
        assert len(self.device_infos) == 0
        for device_info in device_infos:
            if isinstance(device_info, str):
                device_info = DeviceInfo(name=device_info)
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
        idx = len(self._allocations)
        address = self._allocation_start + self._allocation_step * idx
        self._allocations.append((size, None if managed else self._context_stack[-1]))
        return idx, address

    def free_allocation(self, idx):
        self._allocations[idx] = None

    def check_allocation(self, address):
        # will work as long as we don't have offsets larger than `_allocation_step`
        idx = (int(address) - self._allocation_start) // self._allocation_step

        if self._allocations[idx] is None:
            raise RuntimeError("A previously freed allocation is used")

        size, context_ref = self._allocations[idx]

        if context_ref is not None:
            assert context_ref() == self.current_context()


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

        def __init__(self, src, no_extern_c=False, options=None, keep=False):
            assert isinstance(src, MockSourceStr)
            assert isinstance(no_extern_c, bool)
            assert options is None or all(isinstance(option, str) for option in options)
            assert isinstance(keep, bool)

            mock_src = src.mock

            if mock_src.should_fail:
                raise PycudaCompileError()

            self._context = self._backend_ref().current_context()

            function_cls = self._backend_ref().pycuda_driver.Function
            self._kernels = {kernel.name: function_cls(self, kernel) for kernel in mock_src.kernels}
            self._constant_mem = mock_src.constant_mem

        def get_function(self, name):
            return self._kernels[name]

        def get_global(self, name):
            size = self._constant_mem[name]
            alloc = self._backend_ref().pycuda_driver.DeviceAllocation(size)
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
        return self.DeviceAllocation(size, _managed=_managed)

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
        assert isinstance(dest, self.DeviceAllocation)
        dest._check()
        if stream is not None:
            assert stream._context == current_context
        assert dest.size >= src.size * src.dtype.itemsize

    def memcpy_dtoh(self, dest, src):
        self.memcpy_dtoh_async(dest, src)

    def memcpy_dtoh_async(self, dest, src, stream=None):
        current_context = self._backend_ref().current_context()
        assert isinstance(src, self.DeviceAllocation)
        assert isinstance(dest, numpy.ndarray)
        src._check()
        if stream is not None:
            assert stream._context == current_context
        assert src.size >= dest.size * dest.dtype.itemsize

    def memcpy_dtod_async(self, dest, src, size, stream=None):
        current_context = self._backend_ref().current_context()
        assert isinstance(src, self.DeviceAllocation)
        assert isinstance(dest, self.DeviceAllocation)
        dest._check()
        src._check()
        if stream is not None:
            assert stream._context == current_context
        assert dest.size >= size
        assert src.size >= size


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
            self.max_threads_per_block = device_info.max_total_local_size

            self.max_block_dim_x = device_info.max_total_local_size
            self.max_block_dim_y = device_info.max_total_local_size
            self.max_block_dim_z = 64

            self.max_grid_dim_x = 2**32-1
            self.max_grid_dim_y = 2**32-1
            self.max_grid_dim_z = 65536

            self.warp_size = 32
            self.max_shared_memory_per_block = 48 * 1024
            self.multiprocessor_count = 8

        def name(self):
            return self._name

        def compute_capability(self):
            return (5, 0)

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

        def __init__(self, size, _managed=True):
            backend = self._backend_ref()

            idx, address = backend.allocate(size, managed=_managed)

            self._context = backend.current_context()
            self._address = address
            self._idx = idx

            self.size = size

        def _check(self):
            self._backend_ref().check_allocation(self._address)

        def __int__(self):
            return self._address

        def __del__(self):
            # Backend is held alive by the context object we reference.
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
                if isinstance(arg, backend.pycuda_driver.DeviceAllocation):
                    assert param is None
                    assert arg._context == current_context
                elif isinstance(arg, numpy.number):
                    if param is None:
                        backend.check_allocation(arg)
                    else:
                        assert arg == param
                else:
                    raise TypeError(f"Incorrect argument type: {type(arg)}")

            assert isinstance(grid, tuple)
            assert len(grid) == 3
            assert all(isinstance(x, int) for x in grid)

            assert isinstance(block, tuple)
            assert len(block) == 3
            assert all(isinstance(x, int) for x in block)

            # TODO: check that every element is smaller than the corresponding maximum for the device

        def get_attribute(self, attribute):
            if attribute == FunctionAttribute.MAX_THREADS_PER_BLOCK:
                device_idx = self._source_module._context._device_idx
                return self._kernel.max_total_local_sizes[device_idx]
            else:
                raise ValueError(f"Unknown attribute: {attribute}")

    return Function
