import numpy

import pytest

from grunnur import CUDA_API_ID, OPENCL_API_ID, Program, Queue, Array, CompilationError
from grunnur.template import Template

from ..mock_base import MockKernel, MockSourceSnippet


SRC_OPENCL = """
__kernel void multiply(__global int *dest, __global int *a, __global int *b)
{
    const int i = get_global_id(0);
    dest[i] = a[i] * b[i];
}
"""

SRC_CUDA = """
extern "C" __global__ void multiply(int *dest, int *a, int *b)
{
    const int i = threadIdx.x;
    dest[i] = a[i] * b[i];
}
"""

SRC_GENERIC = """
KERNEL void multiply(GLOBAL_MEM int *dest, GLOBAL_MEM int *a, GLOBAL_MEM int *b)
{
    const int i = get_global_id(0);
    dest[i] = a[i] * b[i];
}
"""


def _test_compile(context, no_prelude, is_mocked):

    if is_mocked:
        src = MockSourceSnippet(kernels=[MockKernel('multiply')])
    else:
        if no_prelude:
            src = SRC_CUDA if context.api.id == CUDA_API_ID else SRC_OPENCL
        else:
            src = SRC_GENERIC

    program = Program(context, src, no_prelude=no_prelude)

    if is_mocked and no_prelude:
        assert program.sources[0].mock.prelude.strip() == ""

    a = numpy.arange(16).astype(numpy.int32)
    b = numpy.arange(16).astype(numpy.int32) + 1
    ref = a * b

    queue = Queue.from_device_idxs(context)

    a_dev = Array.from_host(queue, a)
    b_dev = Array.from_host(queue, b)

    res_dev = Array.empty(queue, 16, numpy.int32)

    program.multiply(queue, 16, None, res_dev, a_dev, b_dev)

    res = res_dev.get()

    if not is_mocked:
        assert (res == ref).all()


@pytest.mark.parametrize('no_prelude', [False, True], ids=["with_prelude", "no_prelude"])
def test_compile(context, no_prelude):
    _test_compile(
        context=context,
        no_prelude=no_prelude,
        is_mocked=False)


def test_compile_multi_device(multi_device_context):

    context = multi_device_context

    program = Program(context, SRC_GENERIC)
    a = numpy.arange(16).astype(numpy.int32)
    b = numpy.arange(16).astype(numpy.int32) + 1
    ref = a * b

    queue = Queue.from_device_idxs(context, device_idxs=[0, 1])

    a_dev = Array.from_host(queue, a)
    b_dev = Array.from_host(queue, b)
    res_dev = Array.empty(queue, 16, numpy.int32)

    a_dev_1 = a_dev.single_device_view(0)[:8]
    a_dev_2 = a_dev.single_device_view(1)[8:]

    b_dev_1 = b_dev.single_device_view(0)[:8]
    b_dev_2 = b_dev.single_device_view(1)[8:]

    res_dev_1 = res_dev.single_device_view(0)[:8]
    res_dev_2 = res_dev.single_device_view(1)[8:]

    program.multiply(
        queue, 8, None, [res_dev_1, res_dev_2], [a_dev_1, a_dev_2], [b_dev_1, b_dev_2])

    queue.synchronize()

    res = res_dev.get()
    correct_result = (res == ref).all()

    device_names = [device.name for device in queue.devices.values()]
    expected_to_fail = (
        context.api.id == OPENCL_API_ID and
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


SRC_CONSTANT_MEM = """
KERNEL void copy_from_cm(
    GLOBAL_MEM int *dest
#ifdef GRUNNUR_OPENCL_API
    , CONSTANT_MEM_ARG int *cm1
    , CONSTANT_MEM_ARG int *cm2
#endif
    )
{
    const int i = get_global_id(0);
    dest[i] = cm1[i] + cm2[i];
}
"""


def _test_constant_memory(context, is_mocked):

    cm1 = numpy.arange(16).astype(numpy.int32)
    cm2 = numpy.arange(16).astype(numpy.int32) * 2

    if is_mocked:
        src = MockSourceSnippet(
            constant_mem={
                'cm1': cm1.size * cm1.dtype.itemsize,
                'cm2': cm2.size * cm2.dtype.itemsize},
            kernels=[MockKernel('copy_from_cm')])
    else:
        src = SRC_CONSTANT_MEM

    queue = Queue.from_device_idxs(context)

    cm1_dev = Array.from_host(queue, cm1)
    cm2_dev = Array.from_host(queue, cm2)
    res_dev = Array.empty(queue, 16, numpy.int32)

    if context.api.id == CUDA_API_ID:
        program = Program(context, src, constant_arrays=dict(cm1=cm1, cm2=cm2))
        program.set_constant_array('cm1', cm1_dev) # setting from a device array
        program.set_constant_array('cm2', cm2, queue=queue) # setting from a host array
        program.copy_from_cm(queue, 16, None, res_dev)
    else:
        program = Program(context, src)
        program.copy_from_cm(queue, 16, None, res_dev, cm1_dev, cm2_dev)

    res = res_dev.get()

    if not is_mocked:
        assert (res == cm1 + cm2).all()


def test_constant_memory(context):
    _test_constant_memory(
        context=context,
        is_mocked=False)


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
        src = MockSourceSnippet(should_fail=True)
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
