"""
Utilities for testing and mocking Grunnur internals.
Brittle, handle with care.
"""

from .mock_base import MockKernel, MockSource, MockDefTemplate
from .mock_pycuda import MockPyCUDA, PyCUDADeviceInfo
from .mock_pyopencl import MockPyOpenCL, PyOpenCLDeviceInfo
from .mock_factory import MockBackendFactory
