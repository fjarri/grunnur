from . import dtypes, functions
from ._adapter_base import DeviceParameters
from ._api import (
    API,
    all_api_ids,
    cuda_api_id,
    opencl_api_id,
)
from ._array import Array, ArrayLike, MultiArray
from ._array_metadata import ArrayMetadata, AsArrayMetadata
from ._buffer import Buffer
from ._context import BoundDevice, Context
from ._device import Device, DeviceFilter
from ._device_discovery import (
    platforms_and_devices_by_mask,
    select_devices,
)
from ._modules import Module, Snippet
from ._platform import Platform, PlatformFilter
from ._program import CompilationError, Program
from ._queue import MultiQueue, Queue
from ._static import StaticKernel
from ._template import DefTemplate, RenderError, Template
from ._virtual_alloc import (
    TrivialManager,
    VirtualAllocationStatistics,
    VirtualAllocator,
    VirtualManager,
    ZeroOffsetManager,
)
from ._vsize import VirtualSizeError
