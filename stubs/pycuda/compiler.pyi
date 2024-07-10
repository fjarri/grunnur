from collections.abc import Sequence
from typing import Any, Literal, Optional, overload

import numpy

from . import driver

class SourceModule:
    def __init__(
        self,
        source: str,
        *,
        options: Sequence[str] | None = None,
        keep: bool = False,
        no_extern_c: bool = False,
        cache_dir: str | None = None,
        include_dirs: Sequence[str] = [],
    ): ...
    def get_function(self, name: str) -> Function: ...
    def get_global(self, name: str) -> tuple[int, int]: ...

class Function:
    @overload
    def get_attribute(
        self, attr: Literal[driver.function_attribute.MAX_THREADS_PER_BLOCK]
    ) -> int: ...
    @overload
    def get_attribute(self, attr: driver.function_attribute) -> Any: ...
    def __call__(
        self,
        *args: driver.DeviceAllocation | numpy.generic,
        block: tuple[int, ...],
        grid: tuple[int, ...],
        stream: driver.Stream | None = None,
        shared: int = 0,
    ) -> None: ...
