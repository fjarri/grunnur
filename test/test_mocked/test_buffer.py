from ..test_on_device.test_buffer import _test_allocate, _test_migrate


def test_allocate(mock_context):
    _test_allocate(context=mock_context)


def test_migrate(mock_4_device_context):
    _test_migrate(context=mock_4_device_context)
