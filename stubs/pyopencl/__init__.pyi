from enum import Enum, Flag
from typing import Any, List, Optional, Sequence, Union

import numpy

class Platform:
    name: str
    vendor: str
    version: str

    def get_devices(self) -> list[Device]: ...

class device_type(Enum):
    CPU = ...
    GPU = ...

class Device:
    platform: Platform
    name: str
    type: device_type
    max_work_group_size: int
    max_work_item_sizes: list[int]
    address_bits: int
    extensions: str
    compute_capability_major_nv: int
    warp_size_nv: int
    vendor: str
    local_mem_size: int
    max_compute_units: int

class Context:
    devices: list[Device]

    def __init__(self, devices: Sequence[Device]): ...

class Program:
    def __init__(self, context: Context, src: str): ...
    def build(
        self,
        options: Sequence[str] = [],
        devices: Sequence[Device] | None = None,
        cache_dir: str | None = None,
    ) -> None: ...

class kernel_work_group_info:
    WORK_GROUP_SIZE: Any

class Kernel:
    def __call__(
        self,
        queue: CommandQueue,
        global_size: Sequence[int],
        local_size: Sequence[int] | None,
        *args: Buffer | numpy.generic,
    ) -> Event: ...
    def get_work_group_info(self, param: kernel_work_group_info, device: Device) -> Any: ...

class Event: ...

def get_platforms() -> list[Platform]: ...

class RuntimeError(Exception): ...

class mem_flags(Flag):
    _NONE = ...
    READ_WRITE = ...
    ALLOC_HOST_PTR = ...

class Buffer:
    size: int
    offset: int
    def __init__(self, context: Context, flags: mem_flags = mem_flags._NONE, size: int = 0): ...
    def get_sub_region(
        self, origin: int, size: int, flags: mem_flags = mem_flags._NONE
    ) -> Buffer: ...

class CommandQueue:
    def __init__(self, context: Context, device: Device | None = None): ...
    def finish(self) -> None: ...

def enqueue_copy(
    queue: CommandQueue,
    dest: "Buffer | numpy.ndarray[Any, numpy.dtype[Any]]",
    src: "Buffer | numpy.ndarray[Any, numpy.dtype[Any]]",
    is_blocking: bool = True,
) -> None: ...
