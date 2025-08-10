from . import dtypes, functions
from .adapter_base import DeviceParameters
from .api import (
    API,
    all_api_ids,
    cuda_api_id,
    opencl_api_id,
)
from .array import Array, ArrayLike, MultiArray
from .array_metadata import ArrayMetadata, AsArrayMetadata
from .buffer import Buffer
from .context import BoundDevice, Context
from .device import Device, DeviceFilter
from .device_discovery import (
    platforms_and_devices_by_mask,
    select_devices,
)
from .modules import Module, Snippet
from .platform import Platform, PlatformFilter
from .program import CompilationError, Program
from .queue import MultiQueue, Queue
from .static import StaticKernel
from .template import DefTemplate, RenderError, Template
from .virtual_alloc import VirtualManager
from .vsize import VirtualSizeError


def __getattr__(name: str) -> API:
    if name == "cuda_api":
        return API.from_api_id(cuda_api_id())
    if name == "opencl_api":
        return API.from_api_id(opencl_api_id())
    if name == "any_api":
        apis = API.all_available()
        if len(apis) == 0:
            raise ImportError("No APIs are available. Please install either PyCUDA or PyOpenCL")
        return apis[0]

    raise ImportError(f"Cannot import name '{name}' from '{__name__}'")
