from grunnur.modules import Snippet, RenderableSnippet


class MockKernel:

    def __init__(self, name):
        self.name = name


class MockSource:

    def __init__(self, kernels=[], prelude="", should_fail=False):
        self.kernels = kernels
        self.prelude = prelude
        self.should_fail = should_fail


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

        self.kernels = mock.kernels
        self.should_fail = self.mock.should_fail

    def __radd__(self, other):
        assert isinstance(other, str)
        return MockSourceStr(MockSource(
            kernels=self.mock.kernels,
            prelude=other + self.mock.prelude,
            should_fail=self.mock.should_fail))

    def split(self, delim):
        return self.mock.prelude.split(delim) + ["<<< mock source >>>"]
