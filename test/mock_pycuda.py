from functools import lru_cache
import weakref

from grunnur import CUDA_API_ID


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
        for stacked_context in self._context_stack:
            if stacked_context == context:
                raise ValueError("The given context is already in the context stack")
        self._context_stack.append(context)

    def pop_context(self):
        self._context_stack.pop()

    def current_context(self):
        return self._context_stack[-1]


class PycudaCompileError(Exception):
    pass


class Mock_pycuda_compiler():

    def __init__(self, backend):
        self.SourceModule = make_source_module_class(backend)


@lru_cache()
def make_source_module_class(backend):

    # Prevent the original object from being retained in the closure
    backend = weakref.ref(backend)

    class SourceModule:

        _backend = backend

        def __init__(self, src, no_extern_c=False, options=None, keep=False):
            assert isinstance(src, str)
            assert isinstance(no_extern_c, bool)
            assert options is None or all(isinstance(option, str) for option in options)
            assert isinstance(keep, bool)

    return SourceModule


class Mock_pycuda_driver:

    def __init__(self, backend, cuda_version):

        self._backend = weakref.ref(backend)
        self._version = cuda_version

        self.Device = make_device_class(backend)
        self.Stream = make_stream_class(backend)
        self.Context = make_context_class(backend)

        self.CompileError = PycudaCompileError

    def get_version(self):
        return self._version

    def init(self):
        pass


# We need a device class that is an actual class
# (so that `type()` works on the results of its `__call__()`),
# but at the same time retains a reference to the backend object used to create it
# (so that we can control the properties of the mock).
@lru_cache()
def make_device_class(backend):

    # Prevent the original object from being retained in the closure
    backend = weakref.ref(backend)

    class Device:

        _backend = backend

        def __init__(self, device_idx):
            self._device_idx = device_idx
            self._name = Device._backend().device_names[device_idx]
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
            context = self._backend().pycuda_driver.Context(self._device_idx)
            self._backend().push_context(context)
            return context

        def __eq__(self, other):
            return self._device_idx == other._device_idx

        def __hash__(self):
            return hash((type(self), self._device_idx))

        @staticmethod
        def count():
            return len(Device._backend().device_names)

    return Device


@lru_cache()
def make_context_class(backend):

    # Prevent the original object from being retained in the closure
    backend = weakref.ref(backend)

    class Context:

        _backend = backend

        def __init__(self, device_idx):
            self._device_idx = device_idx

        @staticmethod
        def pop():
            Context._backend().pop_context()

        @staticmethod
        def get_device():
            backend = Context._backend()
            return backend.pycuda_driver.Device(backend.current_context()._device_idx)

        def push(self):
            self._backend().push_context(self)

    return Context


@lru_cache()
def make_stream_class(backend):

    class Stream:

        _backend = backend

        def __init__(self):
            self._context = self._backend.pycuda_driver._current_context()

    return Stream
