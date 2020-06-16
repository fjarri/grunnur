from ..test_on_device.test_static import (
    _test_compile_static,
    _test_compile_static_multi_device,
    )


def test_compile_static(mock_context):
    _test_compile_static(context=mock_context, is_mocked=True)


def test_compile_static_multi_device(mock_4_device_context):
    _test_compile_static_multi_device(context=mock_4_device_context, is_mocked=True)
