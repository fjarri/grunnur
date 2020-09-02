import os.path
import re

import numpy
import pytest

from grunnur import cuda_api_id, opencl_api_id, Program, Queue, Array, CompilationError, MultiDevice, StaticKernel
from grunnur.template import Template

from ..mock_base import MockKernel, MockDefTemplate


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


def _test_compile(context, no_prelude, is_mocked):

    if is_mocked:
        src = MockDefTemplate(kernels=[MockKernel('multiply', [None, None, None, numpy.int32])])
    else:
        if no_prelude:
            src = SRC_CUDA if context.api.id == cuda_api_id() else SRC_OPENCL
        else:
            src = SRC_GENERIC

    program = Program(context, src, no_prelude=no_prelude)

    if is_mocked and no_prelude:
        assert program.sources[0].prelude.strip() == ""

    length = 64

    a = numpy.arange(length).astype(numpy.int32)
    b = numpy.arange(length).astype(numpy.int32) + 1
    c = numpy.int32(3)
    ref = a * b + c

    queue = Queue.on_all_devices(context)

    a_dev = Array.from_host(queue, a)
    b_dev = Array.from_host(queue, b)

    res_dev = Array.empty(queue, length, numpy.int32)
    # Check that passing both Arrays and Buffers is supported
    # Pass one of the buffers as a subregion, too.
    a_dev_view = a_dev.data.get_sub_region(0, a_dev.data.size)
    program.kernel.multiply(queue, length, None, res_dev, a_dev_view, b_dev.data, c)
    res = res_dev.get()
    if not is_mocked:
        assert (res == ref).all()

    # Explicit local_size
    res2_dev = Array.from_host(queue, a) # Array.empty(queue, length, numpy.int32)
    program.kernel.multiply(queue, length, length // 2, res2_dev, a_dev, b_dev, c)
    res2 = res2_dev.get()
    if not is_mocked:
        assert (res2 == ref).all()

    # Explicit device_idxs
    res3_dev = Array.from_host(queue, a) # Array.empty(queue, length, numpy.int32)
    program.kernel.multiply(queue, length, None, res3_dev, a_dev, b_dev, c, device_idxs=[0])
    res3 = res3_dev.get()
    if not is_mocked:
        assert (res3 == ref).all()


@pytest.mark.parametrize('no_prelude', [False, True], ids=["with_prelude", "no_prelude"])
def test_compile(context, no_prelude):
    _test_compile(
        context=context,
        no_prelude=no_prelude,
        is_mocked=False)


def _test_compile_multi_device(context, device_idxs, is_mocked):

    if is_mocked:
        src = MockDefTemplate(kernels=[MockKernel('multiply', [None, None, None, numpy.int32])])
    else:
        src = SRC_GENERIC

    length = 64

    program = Program(context, src)
    a = numpy.arange(length).astype(numpy.int32)
    b = numpy.arange(length).astype(numpy.int32) + 1
    c = numpy.int32(3)
    ref = a * b + c

    queue = Queue.on_device_idxs(context, device_idxs=device_idxs)

    a_dev = Array.from_host(queue, a)
    b_dev = Array.from_host(queue, b)
    res_dev = Array.empty(queue, length, numpy.int32)

    d1, d2 = device_idxs

    a_dev_1 = a_dev.single_device_view(d1)[:length//2]
    a_dev_2 = a_dev.single_device_view(d2)[length//2:]

    b_dev_1 = b_dev.single_device_view(d1)[:length//2]
    b_dev_2 = b_dev.single_device_view(d2)[length//2:]

    res_dev_1 = res_dev.single_device_view(d1)[:length//2]
    res_dev_2 = res_dev.single_device_view(d2)[length//2:]

    program.kernel.multiply(
        queue, length // 2, None,
        MultiDevice(res_dev_1, res_dev_2),
        MultiDevice(a_dev_1, a_dev_2),
        MultiDevice(b_dev_1, b_dev_2),
        c)

    queue.synchronize()

    res = res_dev.get()

    device_names = [device.name for device in queue.devices.values()]

    if not is_mocked:

        correct_result = (res == ref).all()

        expected_to_fail = (
            context.api.id == opencl_api_id() and
            'Apple' in context.platform.name and
            any('GeForce' in name for name in device_names) and
            not all('GeForce' in name for name in device_names))

        if expected_to_fail:
            if correct_result:
                raise Exception("This test was expected to fail on this configuration.")
            else:
                pytest.xfail(
                    "Multi-device OpenCL contexts on an Apple platform with one device being a GeForce "
                    "don't work correctly (the kernel invocation on GeForce is ignored).")

        assert correct_result


def test_compile_multi_device(multi_device_context):
    _test_compile_multi_device(
        context=multi_device_context,
        device_idxs=[0, 1],
        is_mocked=False)


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
    ${static.begin};
    const int i = ${static.global_id}(0);
    dest[i] = cm1[i] + cm2[i] + cm3[i];
}
"""


def _test_constant_memory(context, is_mocked, is_static):

    cm1 = numpy.arange(16).astype(numpy.int32)
    cm2 = numpy.arange(16).astype(numpy.int32) * 2 + 1
    cm3 = numpy.arange(16).astype(numpy.int32) * 3 + 2

    if is_mocked:
        kernel = MockKernel(
            'copy_from_cm',
            [None] if context.api.id == cuda_api_id() else [None, None, None, None],
            max_total_local_sizes={0: 1024})
        src = MockDefTemplate(
            constant_mem={
                'cm1': cm1.size * cm1.dtype.itemsize,
                'cm2': cm2.size * cm2.dtype.itemsize,
                'cm3': cm3.size * cm3.dtype.itemsize},
            kernels=[kernel])
    else:
        src = SRC_CONSTANT_MEM_STATIC if is_static else SRC_CONSTANT_MEM

    queue = Queue.on_all_devices(context)

    cm1_dev = Array.from_host(queue, cm1)
    cm2_dev = Array.from_host(queue, cm2)
    cm3_dev = Array.from_host(queue, cm3)
    res_dev = Array.empty(queue, 16, numpy.int32)

    if context.api.id == cuda_api_id():

        # Use different forms of constant array representation
        constant_arrays=dict(
            cm1=cm1, # as an array(-like) object
            cm2=(cm2.shape, cm2.dtype), # as a tuple of shape and dtype
            cm3=cm3_dev) # as a device array

        if is_static:
            copy_from_cm = StaticKernel(
                queue, src, 'copy_from_cm',
                global_size=16, constant_arrays=constant_arrays)
            copy_from_cm.set_constant_array(queue, 'cm1', cm1_dev) # setting from a device array
            copy_from_cm.set_constant_array(queue, 'cm2', cm2) # setting from a host array
            copy_from_cm.set_constant_array(queue, 'cm3', cm3_dev.data) # setting from a host buffer
        else:
            program = Program(context, src, constant_arrays=constant_arrays)
            program.set_constant_array(queue, 'cm1', cm1_dev) # setting from a device array
            program.set_constant_array(queue, 'cm2', cm2) # setting from a host array
            program.set_constant_array(queue, 'cm3', cm3_dev.data) # setting from a host buffer
            copy_from_cm = lambda *args: program.kernel.copy_from_cm(queue, 16, None, *args)

        copy_from_cm(res_dev)
    else:

        if is_static:
            copy_from_cm = StaticKernel(queue, src, 'copy_from_cm', global_size=16)
        else:
            program = Program(context, src)
            copy_from_cm = lambda *args: program.kernel.copy_from_cm(queue, 16, None, *args)

        copy_from_cm(res_dev, cm1_dev, cm2_dev, cm3_dev)

    res = res_dev.get()

    if not is_mocked:
        assert (res == cm1 + cm2 + cm3).all()


def test_constant_memory(context):
    _test_constant_memory(context=context, is_mocked=False, is_static=False)


SRC_COMPILE_ERROR = """
KERNEL void compile_error(GLOBAL_MEM int *dest)
{
    const int i = get_global_id(0);
    dest[i] = 1;
    zzz
}
"""


def _test_compilation_error(context, capsys, is_mocked):

    if is_mocked:
        src = MockDefTemplate(should_fail=True)
    else:
        src = SRC_COMPILE_ERROR

    with pytest.raises(CompilationError):
        Program(context, src)

    captured = capsys.readouterr()
    assert "Failed to compile on device 0" in captured.out

    # check that the full source is shown (including the prelude)
    assert "#define GRUNNUR_" in captured.out

    if is_mocked:
        assert "<<< mock source >>>" in captured.out
    else:
        assert "KERNEL void compile_error(GLOBAL_MEM int *dest)" in captured.out


def test_compilation_error(context, capsys):
    _test_compilation_error(context=context, capsys=capsys, is_mocked=False)


def _test_keep(context, capsys, is_mocked):
    if is_mocked:
        src = MockDefTemplate(kernels=[MockKernel('multiply', [None, None, None, numpy.int32])])
    else:
        src = SRC_GENERIC

    program = Program(context, src, keep=True)
    captured = capsys.readouterr()
    path = re.match(r'\*\*\* compiler output in (.*)', captured.out).group(1)
    assert os.path.isdir(path)

    if context.api.id == opencl_api_id():
        srcfile = os.path.join(path, 'kernel.cl')
    elif context.api.id == cuda_api_id():
        srcfile = os.path.join(path, 'kernel.cu')

    with open(srcfile) as f:
        source = f.read()

    assert str(src) in source


def test_keep(context, capsys):
    _test_keep(context=context, capsys=capsys, is_mocked=False)
