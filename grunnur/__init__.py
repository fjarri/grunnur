from .__version__ import __version__

from .api_discovery import (
    available_apis,
    find_apis,
    )
from .cuda import make_cuda_api, CUDA_API_ID
from .opencl import make_opencl_api, OPENCL_API_ID
from .static import StaticKernel


def __getattr__(name):
    if name == 'cuda_api':
        return make_cuda_api()
    elif name == 'opencl_api':
        return make_opencl_api()
    elif name == 'any_api':
        apis = available_apis()
        if len(apis) == 0:
            raise ImportError("No APIs are available. Please install either PyCUDA or PyOpenCL")
        else:
            return apis[0]
    else:
        raise ImportError(f"Cannot import name '{name}' from '{__name__}'")
