from .__version__ import __version__

from .adapter_base import ArrayMetadataLike
from .api import (
    API,
    cuda_api_id,
    opencl_api_id,
    all_api_ids,
    )
from .program import Program, CompilationError
from .platform import Platform
from .device import Device
from .device_discovery import (
    platforms_and_devices_by_mask,
    select_devices,
    )
from .queue import Queue, MultiQueue
from .array import Array, MultiArray
from .context import Context
from .buffer import Buffer
from .static import StaticKernel
from .template import Template, DefTemplate, RenderError
from .modules import Module, Snippet
from .vsize import VirtualSizeError
from .virtual_alloc import VirtualManager


def __getattr__(name):
    if name == 'cuda_api':
        return API.from_api_id(cuda_api_id())
    if name == 'opencl_api':
        return API.from_api_id(opencl_api_id())
    if name == 'any_api':
        apis = API.all_available()
        if len(apis) == 0:
            raise ImportError("No APIs are available. Please install either PyCUDA or PyOpenCL")
        return apis[0]
    if name == '__mro__':
        return ()

    raise ImportError(f"Cannot import name '{name}' from '{__name__}'")
