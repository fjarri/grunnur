from .api import (
    API,
    cuda_api_id,
    opencl_api_id,
    all_api_ids,
)
from .program import Program, CompilationError
from .platform import Platform, PlatformFilter
from .device import Device, DeviceFilter
from .device_discovery import (
    platforms_and_devices_by_mask,
    select_devices,
)
from .queue import Queue, MultiQueue
from .array_metadata import ArrayMetadataLike
from .array import Array, ArrayLike, MultiArray
from .context import Context
from .buffer import Buffer
from .static import StaticKernel
from .template import Template, DefTemplate, RenderError
from .modules import Module, Snippet
from .vsize import VirtualSizeError
from .virtual_alloc import VirtualManager


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
