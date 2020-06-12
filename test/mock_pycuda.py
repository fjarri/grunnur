from functools import lru_cache
import weakref

import numpy

from grunnur import CUDA_API_ID

from .mock_base import MockSourceStr


class MockPyCUDA:

    def __init__(self, cuda_version="0.0"):
        self.pycuda_driver = Mock_pycuda_driver(self, cuda_version)
        self.pycuda_compiler = Mock_pycuda_compiler(self)

        self.device_names = []
        self._context_stack = []

        self.api_id = CUDA_API_ID

    def add_devices(self, device_names):
        assert len(self.device_names) == 0
        for device_name in device_names:
            self.device_names.append(device_name)

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

    def get_version(self):
        return self._version

    def init(self):
        pass

    def mem_alloc(self, size):
        return self.DeviceAllocation(size)

    def memcpy_htod(self, dest, src):
        self.memcpy_htod_async(dest, src)

    def memcpy_htod_async(self, dest, src, stream=None):
        current_context = self._backend_ref().current_context()
        assert isinstance(src, numpy.ndarray)
        assert isinstance(dest, self.DeviceAllocation)
        assert dest._context == current_context
        if stream is not None:
            assert stream._context == current_context
        assert dest.size >= src.size * src.dtype.itemsize

    def memcpy_dtoh(self, dest, src):
        self.memcpy_dtoh_async(dest, src)

    def memcpy_dtoh_async(self, dest, src, stream=None):
        current_context = self._backend_ref().current_context()
        assert isinstance(src, self.DeviceAllocation)
        assert isinstance(dest, numpy.ndarray)
        assert src._context == current_context
        if stream is not None:
            assert stream._context == current_context
        assert src.size >= dest.size * dest.dtype.itemsize

    def memcpy_dtod_async(self, dest, src, size, stream=None):
        current_context = self._backend_ref().current_context()
        assert isinstance(src, self.DeviceAllocation)
        assert isinstance(dest, self.DeviceAllocation)
        assert dest._context == current_context
        assert src._context == current_context
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
            self._device_idx = device_idx
            self._name = Device._backend_ref().device_names[device_idx]
            self.max_threads_per_block = 1024

            self.max_block_dim_x = 1024
            self.max_block_dim_y = 1024
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
            return len(Device._backend_ref().device_names)

    return Device


@lru_cache()
def make_context_class(backend):

    class Context:

        # Unlike in other classes, we want to make sure that the backend is still alive
        # as long as a Context object is alive.
        # This will allow us to check that a Context object is not deleted
        # without being popped off the stack.
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

    return Stream


@lru_cache()
def make_device_allocation_class(backend):

    backend_ref = weakref.ref(backend)

    class DeviceAllocation:

        _backend_ref = backend_ref

        def __init__(self, size):
            self._context = self._backend_ref().current_context()
            self.size = size

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

            for arg in args:
                if isinstance(arg, backend.pycuda_driver.DeviceAllocation):
                    assert arg._context == current_context
                elif isinstance(numpy.number):
                    pass
                else:
                    raise TypeError(f"Incorrect argument type: {type(arg)}")

            assert isinstance(grid, tuple)
            assert len(grid) == 3
            assert all(isinstance(x, int) for x in grid)

            assert isinstance(block, tuple)
            assert len(block) == 3
            assert all(isinstance(x, int) for x in block)

            # TODO: check that every element is smaller than the corresponding maximum for the device

    return Function
