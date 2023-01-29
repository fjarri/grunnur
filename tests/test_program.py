from typing import NamedTuple, Tuple
import os.path
import re

import numpy
import pytest

from grunnur import (
    cuda_api_id,
    opencl_api_id,
    cuda_api_id,
    Program,
    Queue,
    MultiQueue,
    Buffer,
    Array,
    MultiArray,
    CompilationError,
    StaticKernel,
    API,
    Context,
    DeviceFilter,
)
from grunnur.template import Template, DefTemplate
from grunnur.testing import MockKernel, MockDefTemplate, PyCUDADeviceInfo


SRC_OPENCL = """
__kernel void multiply(__global int *dest, __global int *a, __global int *b, int c)
{
    const int i = get_global_id(0);
    dest[i] = a[i] * b[i] + c;
}
"""

SRC_CUDA = """
extern "C" __global__ void multiply(int *dest, int *a, int *b, int c)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    dest[i] = a[i] * b[i] + c;
}
"""

SRC_GENERIC = """
KERNEL void multiply(GLOBAL_MEM int *dest, GLOBAL_MEM int *a, GLOBAL_MEM int *b, int c)
{
    const int i = get_global_id(0);
    dest[i] = a[i] * b[i] + c;
}
"""


@pytest.mark.parametrize("no_prelude", [False, True], ids=["with_prelude", "no_prelude"])
def test_compile(mock_or_real_context, no_prelude):

    context, mocked = mock_or_real_context

    if mocked:
        src = MockDefTemplate(kernels=[MockKernel("multiply", [None, None, None, numpy.int32])])
    else:
        if no_prelude:
            src = SRC_CUDA if context.api.id == cuda_api_id() else SRC_OPENCL
        else:
            src = SRC_GENERIC

    program = Program([context.device], src, no_prelude=no_prelude)

    if mocked and no_prelude:
        assert program.sources[context.device].prelude.strip() == ""

    length = 64

    a = numpy.arange(length).astype(numpy.int32)
    b = numpy.arange(length).astype(numpy.int32) + 1
    c = numpy.int32(3)
    ref = a * b + c

    queue = Queue(context.device)

    a_dev = Array.from_host(queue, a)
    b_dev = Array.from_host(queue, b)

    res_dev = Array.empty(context.device, [length], numpy.int32)
    # Check that passing both Arrays and Buffers is supported
    # Pass one of the buffers as a subregion, too.
    a_dev_view = a_dev.data.get_sub_region(0, a_dev.data.size)
    program.kernel.multiply(queue, [length], None, res_dev, a_dev_view, b_dev.data, c)
    res = res_dev.get(queue)
    if not mocked:
        assert (res == ref).all()

    # Explicit local_size
    res2_dev = Array.from_host(queue, a)  # Array.empty(queue, length, numpy.int32)
    program.kernel.multiply(queue, [length], [length // 2], res2_dev, a_dev, b_dev, c)
    res2 = res2_dev.get(queue)
    if not mocked:
        assert (res2 == ref).all()


def test_compile_multi_device(mock_or_real_multi_device_context):

    context, mocked = mock_or_real_multi_device_context
    devices = context.devices[[1, 0]]

    if mocked:
        src = MockDefTemplate(kernels=[MockKernel("multiply", [None, None, None, numpy.int32])])
    else:
        src = SRC_GENERIC

    length = 64

    program = Program(devices, src)
    a = numpy.arange(length).astype(numpy.int32)
    b = numpy.arange(length).astype(numpy.int32) + 1
    c = numpy.int32(3)
    ref = a * b + c

    mqueue = MultiQueue.on_devices(devices)

    a_dev = MultiArray.from_host(mqueue, a)
    b_dev = MultiArray.from_host(mqueue, b)

    res_dev = MultiArray.empty(devices, [length], numpy.int32)
    program.kernel.multiply(mqueue, a_dev.shapes, None, res_dev, a_dev, b_dev, c)
    res = res_dev.get(mqueue)
    if not mocked:
        assert (res == ref).all()

    # Test argument unpacking from dictionaries
    res_dev = MultiArray.empty(devices, [length], numpy.int32)
    program.kernel.multiply(
        mqueue,
        a_dev.shapes,
        {device: None for device in devices},
        res_dev,
        a_dev.subarrays,
        b_dev,
        c,
    )
    res = res_dev.get(mqueue)
    if not mocked:
        assert (res == ref).all()


def test_mismatched_devices(mock_4_device_context):
    context = mock_4_device_context
    src = MockDefTemplate(kernels=[MockKernel("multiply", [None, None, None, numpy.int32])])
    program = Program(context.devices, src)
    with pytest.raises(ValueError, match="Mismatched device sets for global and local sizes"):
        program.kernel.multiply.prepare(
            {context.devices[0]: [10], context.devices[2]: [20]},
            {context.devices[0]: None, context.devices[1]: None},
        )


SRC_CONSTANT_MEM = """
KERNEL void copy_from_cm(
    GLOBAL_MEM int *dest
#ifdef GRUNNUR_OPENCL_API
    , CONSTANT_MEM int *cm1
    , CONSTANT_MEM int *cm2
    , CONSTANT_MEM int *cm3
#endif
    )
{
    const int i = get_global_id(0);
    dest[i] = cm1[i] + cm2[i] + cm3[i];
}
"""


SRC_CONSTANT_MEM_STATIC = """
KERNEL void copy_from_cm(
    GLOBAL_MEM int *dest
#ifdef GRUNNUR_OPENCL_API
    , CONSTANT_MEM int *cm1
    , CONSTANT_MEM int *cm2
    , CONSTANT_MEM int *cm3
#endif
    )
{
    if (${static.skip}()) return;
    const int i = ${static.global_id}(0);
    dest[i] = cm1[i] + cm2[i] + cm3[i];
}
"""


def _test_constant_memory(context, mocked, is_static):

    cm1 = numpy.arange(16).astype(numpy.int32)
    cm2 = numpy.arange(16).astype(numpy.int32) * 2 + 1
    cm3 = numpy.arange(16).astype(numpy.int32) * 3 + 2

    if mocked:
        kernel = MockKernel(
            "copy_from_cm",
            [None] if context.api.id == cuda_api_id() else [None, None, None, None],
            max_total_local_sizes={0: 1024},
        )
        src = MockDefTemplate(
            constant_mem={
                "cm1": cm1.size * cm1.dtype.itemsize,
                "cm2": cm2.size * cm2.dtype.itemsize,
                "cm3": cm3.size * cm3.dtype.itemsize,
            },
            kernels=[kernel],
        )
    else:
        src = SRC_CONSTANT_MEM_STATIC if is_static else SRC_CONSTANT_MEM

    queue = Queue(context.device)

    cm1_dev = Array.from_host(queue, cm1)
    cm2_dev = Array.from_host(queue, cm2)
    cm3_dev = Array.from_host(queue, cm3)
    res_dev = Array.empty(context.device, [16], numpy.int32)

    class Metadata(NamedTuple):
        shape: Tuple[int, ...]
        dtype: numpy.dtype

    if context.api.id == cuda_api_id():

        # Use different forms of constant array representation
        constant_arrays = dict(
            cm1=cm1,  # as an array(-like) object
            cm2=Metadata(cm2.shape, cm2.dtype),  # as a metadata-like
            cm3=cm3_dev,
        )  # as a device array

        if is_static:
            copy_from_cm = StaticKernel(
                [context.device],
                src,
                "copy_from_cm",
                global_size=[16],
                constant_arrays=constant_arrays,
            )
            copy_from_cm.set_constant_array(queue, "cm1", cm1_dev)  # setting from a device array
            copy_from_cm.set_constant_array(queue, "cm2", cm2)  # setting from a host array
            copy_from_cm.set_constant_array(
                queue, "cm3", cm3_dev.data
            )  # setting from a host buffer
        else:
            program = Program([context.device], src, constant_arrays=constant_arrays)
            program.set_constant_array(queue, "cm1", cm1_dev)  # setting from a device array
            program.set_constant_array(queue, "cm2", cm2)  # setting from a host array
            program.set_constant_array(queue, "cm3", cm3_dev.data)  # setting from a host buffer
            copy_from_cm = lambda queue, *args: program.kernel.copy_from_cm(
                queue, [16], None, *args
            )

        copy_from_cm(queue, res_dev)
    else:

        if is_static:
            copy_from_cm = StaticKernel([context.device], src, "copy_from_cm", global_size=[16])
        else:
            program = Program([context.device], src)
            copy_from_cm = lambda queue, *args: program.kernel.copy_from_cm(
                queue, [16], None, *args
            )

        copy_from_cm(queue, res_dev, cm1_dev, cm2_dev, cm3_dev)

    res = res_dev.get(queue)

    if not mocked:
        assert (res == cm1 + cm2 + cm3).all()


def test_constant_memory(mock_or_real_context):
    context, mocked = mock_or_real_context
    _test_constant_memory(context=context, mocked=mocked, is_static=False)


SRC_COMPILE_ERROR = """
KERNEL void compile_error(GLOBAL_MEM int *dest)
{
    const int i = get_global_id(0);
    dest[i] = 1;
    zzz
}
"""


def test_compilation_error(mock_or_real_context, capsys):

    context, mocked = mock_or_real_context

    if mocked:
        src = MockDefTemplate(should_fail=True)
    else:
        src = SRC_COMPILE_ERROR

    with pytest.raises(CompilationError):
        Program([context.device], src)

    captured = capsys.readouterr()
    assert "Failed to compile on device" in captured.out

    # check that the full source is shown (including the prelude)
    assert "#define GRUNNUR_" in captured.out

    if mocked:
        assert "<<< mock source >>>" in captured.out
    else:
        assert "KERNEL void compile_error(GLOBAL_MEM int *dest)" in captured.out


def test_keep(mock_or_real_context, capsys):

    context, mocked = mock_or_real_context

    if mocked:
        src = MockDefTemplate(kernels=[MockKernel("multiply", [None, None, None, numpy.int32])])
    else:
        src = SRC_GENERIC

    program = Program([context.device], src, keep=True)
    captured = capsys.readouterr()
    path = re.match(r"\*\*\* compiler output in (.*)", captured.out).group(1)
    assert os.path.isdir(path)

    if context.api.id == opencl_api_id():
        srcfile = os.path.join(path, "kernel.cl")
    elif context.api.id == cuda_api_id():
        srcfile = os.path.join(path, "kernel.cu")

    with open(srcfile) as f:
        source = f.read()

    assert str(src) in source


def test_compiler_options(mock_context):

    src = MockDefTemplate(kernels=[MockKernel("multiply", [None, None, None, numpy.int32])])
    program = Program([mock_context.device], src, compiler_options=["--my_option"])

    adapter = program._sd_programs[mock_context.device]._sd_program_adapter

    if mock_context.api.id == opencl_api_id():
        assert "--my_option" in adapter._pyopencl_program.test_get_options()
    elif mock_context.api.id == cuda_api_id():
        assert "--my_option" in adapter._pycuda_program.test_get_options()


def test_wrong_kernel_argument(mock_context):
    src = MockDefTemplate(kernels=[MockKernel("multiply", [None, None, None, numpy.int32])])
    program = Program([mock_context.device], src)
    queue = Queue(mock_context.device)

    buf = Buffer.allocate(mock_context.device, 100)

    with pytest.raises(TypeError, match="Incorrect argument type: <class 'int'>"):
        program.kernel.multiply(queue, (100,), None, buf, buf, buf, 4)

    if mock_context.api.id == cuda_api_id():
        with pytest.raises(
            RuntimeError, match="A previously freed allocation or an incorrect address is used"
        ):
            program.kernel.multiply(queue, (100,), None, numpy.uintp(0), buf, buf, 4)


def test_wrong_device_idxs(mock_4_device_context):
    src = MockDefTemplate(kernels=[MockKernel("multiply", [None])])

    context = mock_4_device_context
    program = Program(context.devices[[0, 1]], src)
    mqueue = MultiQueue.on_devices(context.devices[[2, 1]])
    res_dev = MultiArray.empty(context.devices[[2, 1]], [16], numpy.int32)

    # Using all the queue's devices (1, 2)
    with pytest.raises(ValueError, match="Requested execution on devices"):
        program.kernel.multiply(mqueue, [8], None, res_dev)


def test_wrong_context(mock_backend):

    mock_backend.add_devices(["Device0"])

    src = MockDefTemplate(kernels=[MockKernel("multiply", [None])])

    api = API.from_api_id(mock_backend.api_id)
    context = Context.from_devices([api.platforms[0].devices[0]])
    context2 = Context.from_devices([api.platforms[0].devices[0]])

    res_dev = Array.empty(context.device, [16], numpy.int32)

    program = Program([context.device], src)
    queue = Queue(context2.device)

    with pytest.raises(
        ValueError, match="The provided queue must belong to the same context this program uses"
    ):
        program.kernel.multiply(queue, [8], None, res_dev)

    # If we don't do anything here, contexts may be deactivated in an incorrect order
    # (on drop of the corresponding `Context` objects).
    # Grunnur cannot resolve this, so we need to deactivate the currently active context
    # (the second one) manually, and the first one will be deactivated on drop.

    context2.deactivate()


def test_set_constant_array_errors(mock_4_device_context):

    context = mock_4_device_context

    api = API.from_api_id(mock_4_device_context.api.id)
    other_context = Context.from_criteria(api)
    other_queue = Queue(other_context.devices[0])
    # Contexts don't know about each other and can't interact with stack in a consistent manner.
    # So we deactivate the other context if we're on CUDA API.
    if api.id == cuda_api_id():
        other_context.deactivate()

    cm1 = numpy.arange(16).astype(numpy.int32)
    src = MockDefTemplate(
        kernels=[
            MockKernel("kernel", [], max_total_local_sizes={0: 1024, 1: 1024, 2: 1024, 3: 1024})
        ],
        constant_mem={"cm1": cm1.size * cm1.dtype.itemsize},
    )
    queue = Queue(context.devices[0])

    if context.api.id == cuda_api_id():
        program = Program(context.devices, src, constant_arrays=dict(cm1=cm1))

        with pytest.raises(
            ValueError,
            match="The provided queue must belong to the same context as this program uses",
        ):
            program.set_constant_array(other_queue, "cm1", cm1)

        with pytest.raises(TypeError, match="Unsupported array type"):
            program.set_constant_array(queue, "cm1", [1])

        with pytest.raises(ValueError, match="Incorrect size of the constant buffer;"):
            program.set_constant_array(queue, "cm1", cm1[:8])

        with pytest.raises(TypeError, match="Unknown constant array metadata type"):
            program = Program(context.devices[[0, 1, 2]], src, constant_arrays=dict(cm1=1))

        program = Program(context.devices[[0, 1, 2]], src, constant_arrays=dict(cm1=cm1))
        queue3 = Queue(context.devices[3])

        with pytest.raises(
            ValueError, match="The program was not compiled for the device this queue uses"
        ):
            program.set_constant_array(queue3, "cm1", cm1)

    else:
        with pytest.raises(
            ValueError, match="Compile-time constant arrays are only supported for CUDA API"
        ):
            program = Program(context.devices, src, constant_arrays=dict(cm1=cm1))

        program = Program(context.devices, src)
        with pytest.raises(
            RuntimeError, match="OpenCL does not allow setting constant arrays externally"
        ):
            program.set_constant_array(queue, "cm1", cm1)

        with pytest.raises(
            ValueError, match="Compile-time constant arrays are only supported for CUDA API"
        ):
            sk = StaticKernel(context.devices, src, "kernel", [1024], constant_arrays=dict(cm1=cm1))

        sk = StaticKernel(context.devices, src, "kernel", [1024])
        with pytest.raises(
            RuntimeError, match="OpenCL does not allow setting constant arrays externally"
        ):
            sk.set_constant_array(queue, "cm1", cm1)


def test_max_total_local_sizes(mock_backend):
    mock_backend.add_devices(["Device1", "Device2 - tag", "Device3 - tag", "Device4"])
    api = API.from_api_id(mock_backend.api_id)
    context = Context.from_criteria(
        api, devices_num=2, device_filter=DeviceFilter(include_masks=["tag"])
    )

    # Providing max_total_local_sizes for all possible devices to make sure
    # only the ones corresponding to the context will get picked up
    kernel = MockKernel("test", max_total_local_sizes={0: 64, 1: 1024, 2: 512, 3: 128})

    src = MockDefTemplate(kernels=[kernel])
    program = Program(context.devices, src)

    # The indices here correspond to the devices in the context, not in the platform
    assert program.kernel.test.max_total_local_sizes == {
        context.devices[0]: 1024,
        context.devices[1]: 512,
    }


def test_cannot_override_builtin_globals(mock_context):
    with pytest.raises(
        ValueError, match="'device_params' is a reserved global name and cannot be used"
    ):
        Program(
            [mock_context.device],
            MockDefTemplate(kernels=[MockKernel("test", [None])]),
            render_globals=dict(device_params=None),
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

    src = MockDefTemplate(kernels=[MockKernel("test", [None])], source_template=source_template)

    program = Program(context.devices, src)

    assert "max_total_local_size = 1024" in program.sources[context.devices[0]].source
    assert "max_total_local_size = 512" in program.sources[context.devices[1]].source
