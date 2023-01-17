from enum import Enum
from typing import Tuple, Union, Optional, Sequence, Any, TypeVar

import numpy

def init() -> None: ...
def get_version() -> Tuple[int, int, int]: ...

class Device:
    def __init__(self, idx: int): ...
    def make_context(self) -> Context: ...
    @staticmethod
    def count() -> int: ...
    def name(self) -> str: ...
    def compute_capability(self) -> Tuple[int, int]: ...
    max_threads_per_block: int
    max_block_dim_x: int
    max_block_dim_y: int
    max_block_dim_z: int
    max_grid_dim_x: int
    max_grid_dim_y: int
    max_grid_dim_z: int
    warp_size: int
    max_shared_memory_per_block: int
    multiprocessor_count: int

class Context:
    @staticmethod
    def pop() -> None: ...
    def push(self) -> None: ...
    @staticmethod
    def get_device() -> Device: ...

class CompileError(Exception): ...

class Stream:
    def synchronize(self) -> None: ...

class DeviceAllocation:
    def __int__(self) -> int: ...

class function_attribute(Enum):
    MAX_THREADS_PER_BLOCK = ...

def mem_alloc(bytes: int) -> DeviceAllocation: ...
def memcpy_htod(dest: Union[int, DeviceAllocation], src: Any) -> None: ...
def memcpy_htod_async(
    dest: Union[int, DeviceAllocation], src: Any, stream: Optional[Stream] = None
) -> None: ...
def memcpy_dtoh(dest: Any, src: Union[int, DeviceAllocation]) -> None: ...
def memcpy_dtoh_async(
    dest: Any, src: Union[int, DeviceAllocation], stream: Optional[Stream] = None
) -> None: ...
def memcpy_dtod(
    dest: Union[int, DeviceAllocation], src: Union[int, DeviceAllocation], size: int
) -> None: ...
def memcpy_dtod_async(
    dest: Union[int, DeviceAllocation],
    src: Union[int, DeviceAllocation],
    size: int,
    stream: Optional[Stream] = None,
) -> None: ...

_T = TypeVar("_T", bound=numpy.dtype[Any])

def pagelocked_empty(shape: Sequence[int], dtype: _T) -> numpy.ndarray[Any, _T]: ...
