from typing import Optional, Sequence, Tuple, Any, Union, Literal, overload

import numpy
from . import driver

class SourceModule:
    def __init__(
        self,
        source: str,
        options: Optional[Sequence[str]] = None,
        keep: bool = False,
        no_extern_c: bool = False,
        cache_dir: Optional[str] = None,
        include_dirs: Sequence[str] = [],
    ): ...
    def get_function(self, name: str) -> Function: ...
    def get_global(self, name: str) -> Tuple[int, int]: ...

class Function:
    @overload
    def get_attribute(
        self, attr: Literal[driver.function_attribute.MAX_THREADS_PER_BLOCK]
    ) -> int: ...
    @overload
    def get_attribute(self, attr: driver.function_attribute) -> Any: ...
    def __call__(
        self,
        *args: Union[driver.DeviceAllocation, numpy.generic],
        block: Tuple[int, ...],
        grid: Tuple[int, ...],
        stream: Optional[driver.Stream] = None,
        shared: int = 0,
    ) -> None: ...
