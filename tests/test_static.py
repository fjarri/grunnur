import pytest
import numpy

from grunnur import (
    cuda_api_id,
    opencl_api_id,
    StaticKernel,
    VirtualSizeError,
    API,
    Context,
    Queue,
    MultiQueue,
    Array,
    MultiArray,
)
from grunnur.template import DefTemplate
from grunnur.testing import MockKernel, MockDefTemplate, PyCUDADeviceInfo, PyOpenCLDeviceInfo

from test_program import _test_constant_memory


SRC = """
KERNEL void multiply(GLOBAL_MEM int *dest, GLOBAL_MEM int *a, GLOBAL_MEM int *b)
{
    if (${static.skip}()) return;
    const int i = ${static.global_id}(0);
    const int j = ${static.global_id}(1);
    const int idx = ${static.global_flat_id}();
    dest[idx] = a[i] * b[j];
}
"""


def test_compile_static(mock_or_real_context):

    context, mocked = mock_or_real_context

    if mocked:
        kernel = MockKernel("multiply", [None, None, None], max_total_local_sizes={0: 1024})
        src = MockDefTemplate(kernels=[kernel])
    else:
        src = SRC

    a = numpy.arange(11).astype(numpy.int32)
    b = numpy.arange(15).astype(numpy.int32)
    ref = numpy.outer(a, b)

    queue = Queue(context.device)

    a_dev = Array.from_host(queue, a)
    b_dev = Array.from_host(queue, b)

    res_dev = Array.empty(context.device, (11, 15), numpy.int32)

    multiply = StaticKernel([context.device], src, "multiply", (11, 15))
    multiply(queue, res_dev, a_dev, b_dev)

    res = res_dev.get(queue)

    if not mocked:
        assert (res == ref).all()


def test_compile_static_multi_device(mock_or_real_multi_device_context):

    context, mocked = mock_or_real_multi_device_context

    if mocked:
        kernel = MockKernel("multiply", [None, None, None], max_total_local_sizes={0: 1024, 1: 512})
        src = MockDefTemplate(kernels=[kernel])
    else:
        src = SRC

    a = numpy.arange(22).astype(numpy.int32)
    b = numpy.arange(15).astype(numpy.int32)
    ref = numpy.outer(a, b)

    mqueue = MultiQueue.on_devices(context.devices[[0, 1]])

    a_dev = MultiArray.from_host(mqueue, a)
    b_dev = MultiArray.from_host(mqueue, b, splay=MultiArray.CloneSplay())
    res_dev = MultiArray.empty(mqueue.devices, (22, 15), ref.dtype)

    multiply = StaticKernel(mqueue.devices, src, "multiply", res_dev.shapes)
    multiply(mqueue, res_dev, a_dev, b_dev)

    res = res_dev.get(mqueue)

    if not mocked:
        assert (res == ref).all()


def test_constant_memory(mock_or_real_context):
    context, mocked = mock_or_real_context
    _test_constant_memory(context=context, mocked=mocked, is_static=True)


def test_find_local_size(mock_context):
    kernel = MockKernel("multiply", [None], max_total_local_sizes={0: 64})
    src = MockDefTemplate(kernels=[kernel])
    multiply = StaticKernel([mock_context.device], src, "multiply", (11, 15))
    assert multiply._vs_metadata[mock_context.devices[0]].real_global_size == (16, 12)
    assert multiply._vs_metadata[mock_context.devices[0]].real_local_size == (16, 4)


def test_reserved_names(mock_context):
    kernel = MockKernel("test", [None])
    src = MockDefTemplate(kernels=[kernel])
    with pytest.raises(ValueError, match="The global name 'static' is reserved in static kernels"):
        multiply = StaticKernel(
            [mock_context.device], src, "test", (1024,), render_globals=dict(static=1)
        )


def test_zero_max_total_local_size(mock_context):
    kernel = MockKernel("test", [None], max_total_local_sizes={0: 0})
    src = MockDefTemplate(kernels=[kernel])
    with pytest.raises(
        VirtualSizeError,
        match="The kernel requires too much resourses to be executed with any local size",
    ):
        multiply = StaticKernel([mock_context.device], src, "test", (1024,))


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
        max_grid_dim_z=2**8,
    )

    mock_backend_pycuda.add_devices([device_info])
    api = API.from_api_id(mock_backend_pycuda.api_id)
    device = api.platforms[0].devices[0]
    context = Context.from_devices([device])
    kernel = MockKernel("test", [None], max_total_local_sizes={0: 16})
    src = MockDefTemplate(kernels=[kernel])

    # Just enough to fit in the grid limits
    multiply = StaticKernel(
        [context.device], src, "test", (2**14, 2**10, 2**8), (2**4, 1, 1)
    )

    # Global size is too large to fit on the device,
    # so virtual size finding fails and the error is propagated to the user.
    with pytest.raises(
        VirtualSizeError, match="Bounding global size \\[16384, 2048, 256\\] is too large"
    ):
        multiply = StaticKernel(
            [context.device], src, "test", (2**14, 2**11, 2**8), (2**4, 1, 1)
        )


def test_builtin_globals(mock_backend_pycuda):
    mock_backend_pycuda.add_devices(
        [PyCUDADeviceInfo(max_threads_per_block=1024), PyCUDADeviceInfo(max_threads_per_block=512)]
    )

    source_template = DefTemplate.from_string(
        "mock_source",
        [],
        """
        KERNEL void test()
        {
            int max_total_local_size = ${device_params.max_total_local_size};
        }
        """,
    )

    api = API.from_api_id(mock_backend_pycuda.api_id)
    context = Context.from_devices([api.platforms[0].devices[0], api.platforms[0].devices[1]])

    src = MockDefTemplate(
        kernels=[MockKernel("test", [None], max_total_local_sizes={0: 1024, 1: 512})],
        source_template=source_template,
    )

    kernel = StaticKernel(context.devices, src, "test", (1024,))

    assert "max_total_local_size = 1024" in kernel.sources[context.devices[0]].source
    assert "max_total_local_size = 512" in kernel.sources[context.devices[1]].source
