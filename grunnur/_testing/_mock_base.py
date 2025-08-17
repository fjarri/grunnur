# This is a testing module, so it has a lot of assertions.
# ruff: noqa: S101

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import mako.template

from .._adapter_base import DeviceType
from .._template import DefTemplate

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping, Sequence


class MockKernel:
    def __init__(
        self,
        name: str,
        parameters: Sequence[Any] = (),
        max_total_local_sizes: Mapping[int, int] = {},
    ):
        self.name = name
        self.parameters = parameters
        self.max_total_local_sizes = max_total_local_sizes


class MockSource:
    def __init__(
        self,
        *,
        kernels: Sequence[MockKernel] = (),
        prelude: str = "",
        should_fail: bool = False,
        constant_mem: Mapping[str, int] = {},
        source: str | None = None,
        source_template: DefTemplate | None = None,
    ):
        self.name = "mock_source"
        self.kernels = kernels
        self.prelude = prelude
        self.should_fail = should_fail
        self.constant_mem = constant_mem

        # TODO: use two different classes to represent a template source and a template object.
        assert source is None or source_template is None

        self.source = source
        self.source_template = source_template

    def __radd__(self, other: Any) -> MockSource:
        assert isinstance(other, str)
        return MockSource(
            kernels=self.kernels,
            prelude=other + self.prelude,
            should_fail=self.should_fail,
            constant_mem=self.constant_mem,
            source=self.source,
            source_template=self.source_template,
        )

    def split(self, sep: str | None = None) -> list[str]:
        return [*self.prelude.split(sep), "<<< mock source >>>"]

    def __str__(self) -> str:
        return self.name

    def __contains__(self, substr: str) -> bool:
        return False if self.source is None else substr in self.source

    def render(self, *args: Any, **kwds: Any) -> MockSource:
        if self.source_template is None:
            return self
        return MockSource(
            kernels=self.kernels,
            prelude=self.prelude,
            should_fail=self.should_fail,
            constant_mem=self.constant_mem,
            source=self.source_template.render(*args, **kwds),
        )


class MockDefTemplate(DefTemplate):
    def __init__(self, *args: Any, **kwds: Any):
        dummy_template = mako.template.Template(text=r"""<%def name="mock_source()"></%def>""")
        super().__init__(
            name="mock_source", mako_def_template=dummy_template.get_def("mock_source"), source=""
        )
        self._mock_source = MockSource(*args, **kwds)

    def render(self, *args: Any, **kwds: Any) -> str:
        # We can't make `MockSource` a subclass of `str` since we need to add attributes to it.
        # So we have to intentionally ignore the return type mismatch.
        return cast(str, self._mock_source.render(*args, **kwds))

    def __str__(self) -> str:
        return str(self._mock_source)
