from grunnur import StaticKernel

from ..mock_base import MockKernel, MockDefTemplate, MockDefTemplate
from ..test_on_device.test_static import (
    _test_compile_static,
    _test_compile_static_multi_device,
    )
from ..test_on_device.test_program import _test_constant_memory


def test_compile_static(mock_context):
    _test_compile_static(context=mock_context, is_mocked=True)


def test_find_local_size(mock_context):
    kernel = MockKernel('multiply', [None], max_total_local_sizes={0: 64})
    src = MockDefTemplate(kernels=[kernel])
    multiply = StaticKernel(mock_context, src, 'multiply', (11, 15))
    assert multiply._vs_metadata[0].real_global_size == (16, 12)
    assert multiply._vs_metadata[0].real_local_size == (16, 4)


def test_compile_static_multi_device(mock_4_device_context):
    _test_compile_static_multi_device(context=mock_4_device_context, is_mocked=True)


def test_constant_memory(mock_context):
    _test_constant_memory(context=mock_context, is_mocked=True, is_static=True)
