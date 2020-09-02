import pytest

from grunnur import StaticKernel, VirtualSizeError, API, Context, Queue
from grunnur.template import DefTemplate

from ..mock_base import MockKernel, MockDefTemplate, MockDefTemplate
from ..test_on_device.test_static import (
    _test_compile_static,
    _test_compile_static_multi_device,
    )
from ..test_on_device.test_program import _test_constant_memory
from ..mock_pycuda import PyCUDADeviceInfo
from ..mock_pyopencl import PyOpenCLDeviceInfo


def test_compile_static(mock_context):
    _test_compile_static(context=mock_context, is_mocked=True)


def test_find_local_size(mock_context):
    queue = Queue.on_all_devices(mock_context)
    kernel = MockKernel('multiply', [None], max_total_local_sizes={0: 64})
    src = MockDefTemplate(kernels=[kernel])
    multiply = StaticKernel(queue, src, 'multiply', (11, 15))
    assert multiply._vs_metadata[0].real_global_size == (16, 12)
    assert multiply._vs_metadata[0].real_local_size == (16, 4)


def test_compile_static_multi_device(mock_4_device_context):
    _test_compile_static_multi_device(context=mock_4_device_context, is_mocked=True)


def test_constant_memory(mock_context):
    _test_constant_memory(context=mock_context, is_mocked=True, is_static=True)


def test_reserved_names(mock_context):
    queue = Queue.on_all_devices(mock_context)
    kernel = MockKernel('test', [None])
    src = MockDefTemplate(kernels=[kernel])
    with pytest.raises(ValueError, match="The global name 'static' is reserved in static kernels"):
        multiply = StaticKernel(queue, src, 'test', (1024,), render_globals=dict(static=1))


def test_zero_max_total_local_size(mock_context):
    queue = Queue.on_all_devices(mock_context)
    kernel = MockKernel('test', [None], max_total_local_sizes={0: 0})
    src = MockDefTemplate(kernels=[kernel])
    with pytest.raises(
            VirtualSizeError,
            match="The kernel requires too much resourses to be executed with any local size"):
        multiply = StaticKernel(queue, src, 'test', (1024,))


def test_virtual_sizes_error_propagated(mock_backend_pycuda):

    # Testing for PyCUDA backend only since mocked PyOpenCL backend does not have a way
    # to set maximum global sizes (PyOpenCL devices don't have a corresponding parameter),
    # and PyCUDA is enough to test the required code path.

    device_info = PyCUDADeviceInfo(
        max_threads_per_block=2**4,
        max_block_dim_x=2**4,
        max_block_dim_y=2**4,
        max_block_dim_z=2**4,
        max_grid_dim_x=2**10,
        max_grid_dim_y=2**10,
        max_grid_dim_z=2**8)

    mock_backend_pycuda.add_devices([device_info])
    api = API.from_api_id(mock_backend_pycuda.api_id)
    device = api.platforms[0].devices[0]
    context = Context.from_devices([device])
    kernel = MockKernel('test', [None], max_total_local_sizes={0: 16})
    src = MockDefTemplate(kernels=[kernel])

    queue = Queue.on_all_devices(context)

    # Just enough to fit in the grid limits
    multiply = StaticKernel(queue, src, 'test', (2**14, 2**10, 2**8), (2**4, 1, 1))

    # Global size is too large to fit on the device,
    # so virtual size finding fails and the error is propagated to the user.
    with pytest.raises(
            VirtualSizeError,
            match="Bounding global size \\(16384, 2048, 256\\) is too large"):
        multiply = StaticKernel(queue, src, 'test', (2**14, 2**11, 2**8), (2**4, 1, 1))


def test_builtin_globals(mock_backend_pycuda):
    mock_backend_pycuda.add_devices([
        PyCUDADeviceInfo(max_threads_per_block=1024),
        PyCUDADeviceInfo(max_threads_per_block=512)])

    source_template = DefTemplate.from_string(
        'mock_source', [],
        """
        KERNEL void test()
        {
            int max_total_local_size = ${device_params.max_total_local_size};
        }
        """)

    api = API.from_api_id(mock_backend_pycuda.api_id)
    context = Context.from_devices([api.platforms[0].devices[0], api.platforms[0].devices[1]])

    src = MockDefTemplate(
        kernels=[MockKernel('test', [None], max_total_local_sizes={0: 1024, 1: 512})],
        source_template=source_template)

    queue = Queue.on_all_devices(context)
    kernel = StaticKernel(queue, src, 'test', (1024,))

    assert 'max_total_local_size = 1024' in kernel.sources[0].source
    assert 'max_total_local_size = 512' in kernel.sources[1].source
