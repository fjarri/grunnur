from grunnur.modules import Snippet, RenderableSnippet


class MockKernel:

    def __init__(self, name, parameters=[], max_total_local_sizes={}):
        self.name = name
        self.parameters = parameters
        self.max_total_local_sizes = max_total_local_sizes


class MockSource:

    def __init__(self, kernels=[], prelude="", should_fail=False, constant_mem={}):
        self.kernels = kernels
        self.prelude = prelude
        self.should_fail = should_fail
        self.constant_mem = constant_mem


class MockSourceSnippet(Snippet):

    def __init__(self, *args, **kwds):
        self.mock = MockSource(*args, **kwds)

    def __process_modules__(self, process):
        return MockSourceRenderableSnippet(self.mock)


class MockSourceRenderableSnippet(RenderableSnippet):

    def __init__(self, mock):
        self.mock = mock

    def __call__(self):
        return MockSourceStr(self.mock)


class MockSourceStr:

    def __init__(self, mock):
        self.mock = mock

    def __radd__(self, other):
        assert isinstance(other, str)
        return MockSourceStr(MockSource(
            kernels=self.mock.kernels,
            prelude=other + self.mock.prelude,
            should_fail=self.mock.should_fail,
            constant_mem=self.mock.constant_mem))

    def split(self, delim):
        return self.mock.prelude.split(delim) + ["<<< mock source >>>"]


class DeviceInfo:

    def __init__(self, name="DefaultDeviceName", max_total_local_size=1024):
        self.name = name
        self.max_total_local_size = max_total_local_size
