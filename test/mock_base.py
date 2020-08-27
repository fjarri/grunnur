from grunnur.adapter_base import DeviceType
from grunnur.template import DefTemplate


class MockKernel:

    def __init__(self, name, parameters=[], max_total_local_sizes={}):
        self.name = name
        self.parameters = parameters
        self.max_total_local_sizes = max_total_local_sizes


class MockSource:

    def __init__(self, kernels=[], prelude="", should_fail=False, constant_mem={}, source=None, source_template=None):
        self.name = "mock_source"
        self.kernels = kernels
        self.prelude = prelude
        self.should_fail = should_fail
        self.constant_mem = constant_mem

        assert source is None or source_template is None

        self.source = source
        self.source_template = source_template

    def __radd__(self, other):
        assert isinstance(other, str)
        return MockSource(
            kernels=self.kernels,
            prelude=other + self.prelude,
            should_fail=self.should_fail,
            constant_mem=self.constant_mem,
            source=self.source,
            source_template=self.source_template)

    def split(self, delim):
        return self.prelude.split(delim) + ["<<< mock source >>>"]

    def __str__(self):
        return self.name

    def render(self, *args, **kwds):
        if self.source_template is None:
            return self
        else:
            return MockSource(
                kernels=self.kernels,
                prelude=self.prelude,
                should_fail=self.should_fail,
                constant_mem=self.constant_mem,
                source=self.source_template.render(*args, **kwds))


class MockDefTemplate(DefTemplate):

    def __init__(self, *args, **kwds):
        super().__init__(name="mock_source", mako_def_template=None, source="")
        self._mock_source = MockSource(*args, **kwds)

    def render(self, *args, **kwds):
        return self._mock_source.render(*args, **kwds)

    def __str__(self):
        return str(self._mock_source)
