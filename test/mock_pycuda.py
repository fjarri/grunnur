from functools import lru_cache
import weakref


class MockPyCUDA:

    def __init__(self, platforms_devices, version="0.0"):
        self.pycuda_drv = Mock_pycuda_drv(platforms_devices, version)


class Mock_pycuda_drv:

    def __init__(self, platforms_devices, version):

        assert len(platforms_devices) == 1 # for compatibility with MockPyOpenCL
        _, device_names = platforms_devices[0]

        self.device_names = device_names
        self._version = version

        self._context_stack = []

        # set the number of platforms, devices etc here
        self.Device = make_device_class(self)
        self.Stream = make_stream_class(self)
        self.Context = make_context_class(self)

    def get_version(self):
        return self._version

    def init(self):
        pass

    def _push_context(self, context):
        # avoiding circular references, since `context` has a reference to this object
        for stacked_context in self._context_stack:
            if stacked_context() == context:
                raise ValueError("The given context is already in the context stack")
        self._context_stack.append(weakref.ref(context))

    def _pop_context(self):
        self._context_stack.pop()

    def _current_context(self):
        return self._context_stack[-1]()


# We need a device class that is an actual class
# (so that `type()` works on the results of its `__call__()`),
# but at the same time retains a reference to the backend object used to create it
# (so that we can control the properties of the mock).
@lru_cache() # TODO: check that we don't need __eq__ for MockPyCUDA or something like that
def make_device_class(backend):

    class Device:

        _backend = backend

        def __init__(self, device_idx):
            self._device_idx = device_idx
            self._name = Device._backend.device_names[device_idx]
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
            context = self._backend.Context(self._device_idx)
            self._backend._push_context(context)
            return context

        def __eq__(self, other):
            return self._device_idx == other._device_idx

        def __hash__(self):
            return hash((type(self), self._device_idx))

        @staticmethod
        def count():
            return len(Device._backend.device_names)

    return Device


@lru_cache()
def make_context_class(backend):

    class Context:

        _backend = backend

        def __init__(self, device_idx):
            self._device_idx = device_idx

        @staticmethod
        def pop():
            Context._backend._pop_context()

        @staticmethod
        def get_device():
            return Context._backend.Device(Context._backend._current_context()._device_idx)

        def push(self):
            self._backend._push_context(self)

    return Context


@lru_cache()
def make_stream_class(backend):

    class Stream:

        _backend = backend

        def __init__(self):
            self._context = self._backend._current_context()

    return Stream
