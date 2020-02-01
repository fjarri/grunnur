from functools import lru_cache
import weakref


class MockPyCUDA:

    def __init__(self, device_names, version="0.0"):

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

        def __init__(self, device_num):
            self._device_num = device_num
            self._name = Device._backend.device_names[device_num]
            self.max_threads_per_block = 1024

        def name(self):
            return self._name

        def make_context(self):
            context = self._backend.Context(self._device_num)
            self._backend._push_context(context)
            return context

        def __eq__(self, other):
            return self._device_num == other._device_num

        def __hash__(self):
            return hash((type(self), self._device_num))

        @staticmethod
        def count():
            return len(Device._backend.device_names)

    return Device


@lru_cache()
def make_context_class(backend):

    class Context:

        _backend = backend

        def __init__(self, device_num):
            self._device_num = device_num

        @staticmethod
        def pop():
            Context._backend._pop_context()

        @staticmethod
        def get_device():
            return Context._backend.Device(Context._backend._current_context()._device_num)

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
