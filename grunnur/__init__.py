from .__version__ import __version__

from .api import (
    API,
    cuda_api_id,
    opencl_api_id,
    all_api_ids,
    )
from .program import Program, CompilationError, MultiDevice
from .platform import Platform
from .device import Device
from .queue import Queue
from .array import Array
from .context import Context
from .buffer import Buffer
from .static import StaticKernel


def __getattr__(name):
    if name == 'cuda_api':
        return API.from_api_id(cuda_api_id())
    elif name == 'opencl_api':
        return API.from_api_id(opencl_api_id())
    elif name == 'any_api':
        apis = API.all()
        if len(apis) == 0:
            raise ImportError("No APIs are available. Please install either PyCUDA or PyOpenCL")
        else:
            return apis[0]
    elif name == '__annotations__':
        return {}
    elif name == '__mro__':
        return ()
    else:
        raise ImportError(f"Cannot import name '{name}' from '{__name__}'")
