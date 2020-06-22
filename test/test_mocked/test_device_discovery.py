import pytest

from ..utils import check_select_devices
from ..mock_pyopencl import PyOpenCLDeviceInfo


def test_platform_take_single(mock_stdin, mock_backend_factory, capsys):
    # Test that if only one platform is available, it is selected automatically

    platforms_devices = [
        ('Platform1', ['Device1', 'Device2'])]
    inputs = ['1']
    devices = check_select_devices(
        mock_stdin, mock_backend_factory, capsys, platforms_devices,
        inputs=inputs, interactive=True)

    assert len(devices) == 1
    assert devices[0].name == 'Device2'


def test_platform_choose(mock_stdin, mock_backend_factory, capsys):
    # Test selecting one of several available platforms

    platforms_devices = [
        ('Platform1', ['Device1', 'Device2']),
        ('Platform2', ['Device3', 'Device4'])]
    inputs = ['1', '0']
    devices = check_select_devices(
        mock_stdin, mock_backend_factory, capsys, platforms_devices,
        inputs=inputs, interactive=True)

    assert len(devices) == 1
    assert devices[0].name == 'Device3'


def test_platform_take_default(mock_stdin, mock_backend_factory, capsys):
    # Test that just pressing enter gives the default (first) platform

    platforms_devices = [
        ('Platform1', ['Device1', 'Device2']),
        ('Platform2', ['Device3', 'Device4'])]
    inputs = ['', '0']
    devices = check_select_devices(
        mock_stdin, mock_backend_factory, capsys, platforms_devices,
        inputs=inputs, interactive=True)

    assert len(devices) == 1
    assert devices[0].name == 'Device1'


def test_device_take_single(mock_stdin, mock_backend_factory, capsys):
    # Test that if only one device is available, it is selected automatically

    platforms_devices = [
        ('Platform1', ['Device1', 'Device2']),
        ('Platform2', ['Device3'])]
    inputs = ['1']
    devices = check_select_devices(
        mock_stdin, mock_backend_factory, capsys, platforms_devices,
        inputs=inputs, interactive=True)

    assert len(devices) == 1
    assert devices[0].name == 'Device3'


def test_device_take_multiple(mock_stdin, mock_backend_factory, capsys):
    # Test the case when several devices are requested,
    # and that's exactly how many the chosen platform has.
    # Technically it's the same as a single device and quantity=1, but testing just in case.

    platforms_devices = [
        ('Platform1', ['Device1', 'Device2']),
        ('Platform2', ['Device3', 'Device4', 'Device5'])]
    devices = check_select_devices(
        mock_stdin, mock_backend_factory, capsys, platforms_devices,
        interactive=True, quantity=3)

    assert len(devices) == 3
    assert [device.name for device in devices] == ['Device3', 'Device4', 'Device5']


def test_device_choose_single(mock_stdin, mock_backend_factory, capsys):
    # Test that if only one device is available, it is selected automatically

    platforms_devices = [
        ('Platform1', ['Device1', 'Device2']),
        ('Platform2', ['Device3', 'Device4'])]
    inputs = ['1', '1']
    devices = check_select_devices(
        mock_stdin, mock_backend_factory, capsys, platforms_devices,
        inputs=inputs, interactive=True)

    assert len(devices) == 1
    assert devices[0].name == 'Device4'


def test_device_choose_multiple(mock_stdin, mock_backend_factory, capsys):
    # Test that if only one device is available, it is selected automatically

    platforms_devices = [
        ('Platform1', ['Device1', 'Device2']),
        ('Platform2', ['Device3', 'Device4', 'Device5'])]
    inputs = ['1', '0, 2']
    devices = check_select_devices(
        mock_stdin, mock_backend_factory, capsys, platforms_devices,
        inputs=inputs, interactive=True, quantity=2)

    assert len(devices) == 2
    assert [device.name for device in devices] == ['Device3', 'Device5']


def test_device_choose_multiple_wrong_quantity(mock_stdin, mock_backend_factory, capsys):
    # Test error in the case when the user selects a number of devices not equal to `quantity`

    platforms_devices = [
        ('Platform1', ['Device1', 'Device2']),
        ('Platform2', ['Device3', 'Device4', 'Device5'])]
    inputs = ['1', '0, 1, 2']

    with pytest.raises(ValueError):
        devices = check_select_devices(
            mock_stdin, mock_backend_factory, capsys, platforms_devices,
            inputs=inputs, interactive=True, quantity=2)


def test_device_take_default_single(mock_stdin, mock_backend_factory, capsys):
    # Test that just pressing enter gives the default (first) device

    platforms_devices = [
        ('Platform1', ['Device1', 'Device2']),
        ('Platform2', ['Device3', 'Device4'])]
    inputs = ['1', '']
    devices = check_select_devices(
        mock_stdin, mock_backend_factory, capsys, platforms_devices,
        inputs=inputs, interactive=True)

    assert len(devices) == 1
    assert devices[0].name == 'Device3'


def test_device_take_default_multiple(mock_stdin, mock_backend_factory, capsys):
    # Test that just pressing enter gives the first `quantity` devices

    platforms_devices = [
        ('Platform1', ['Device1', 'Device2']),
        ('Platform2', ['Device3', 'Device4', 'Device5'])]
    inputs = ['1', '']
    devices = check_select_devices(
        mock_stdin, mock_backend_factory, capsys, platforms_devices,
        inputs=inputs, interactive=True, quantity=2)

    assert len(devices) == 2
    assert [device.name for device in devices] == ['Device3', 'Device4']


def test_device_take_all_available(mock_stdin, mock_backend_factory, capsys):
    # Test that `quantity=None` returns all suitable devices.

    platforms_devices = [
        ('Platform1', ['Device1', 'Device2']),
        ('Platform2', ['Device3', 'Device4', 'Device5'])]
    inputs = ['1']
    devices = check_select_devices(
        mock_stdin, mock_backend_factory, capsys, platforms_devices,
        inputs=inputs, interactive=True, quantity=None)

    assert len(devices) == 3
    assert [device.name for device in devices] == ['Device3', 'Device4', 'Device5']


def test_filter_include_platforms(mock_stdin, mock_backend_factory, capsys):

    platforms_devices = [
        ('PlatformFoo', ['Device1', 'Device2']),
        ('PlatformBar', ['Device3', 'Device4', 'Device5'])]
    devices = check_select_devices(
        mock_stdin, mock_backend_factory, capsys, platforms_devices,
        platform_include_masks=['Bar'])

    assert len(devices) == 1
    assert devices[0].name == 'Device3'


def test_filter_exclude_platforms(mock_stdin, mock_backend_factory, capsys):

    platforms_devices = [
        ('PlatformFoo', ['Device1', 'Device2']),
        ('PlatformBar', ['Device3', 'Device4', 'Device5'])]
    devices = check_select_devices(
        mock_stdin, mock_backend_factory, capsys, platforms_devices,
        platform_exclude_masks=['Foo'])

    assert len(devices) == 1
    assert devices[0].name == 'Device3'


def test_filter_include_devices(mock_stdin, mock_backend_factory, capsys):

    platforms_devices = [
        ('PlatformFoo', ['DeviceFoo', 'DeviceBar']),
        ('PlatformBar', ['DeviceFoo', 'DeviceBar', 'DeviceBaz'])]
    devices = check_select_devices(
        mock_stdin, mock_backend_factory, capsys, platforms_devices,
        inputs=['1'], interactive=True, device_include_masks=['Foo'])

    assert len(devices) == 1
    assert devices[0].name == 'DeviceFoo'
    assert devices[0].platform.name == 'PlatformBar'


def test_filter_exclude_devices(mock_stdin, mock_backend_factory, capsys):

    platforms_devices = [
        ('PlatformFoo', ['DeviceFoo', 'DeviceBar']),
        ('PlatformBar', ['DeviceFoo', 'DeviceBar', 'DeviceBaz'])]
    devices = check_select_devices(
        mock_stdin, mock_backend_factory, capsys, platforms_devices,
        inputs=['1', '1'], interactive=True, device_exclude_masks=['Bar'])

    assert len(devices) == 1
    assert devices[0].name == 'DeviceBaz'


def test_filter_exclude_all_devices(mock_stdin, mock_backend_factory, capsys):

    platforms_devices = [
        ('PlatformFoo', ['DeviceFoo', 'DeviceBar']),
        ('PlatformBar', ['DeviceFoo', 'DeviceBar', 'DeviceBaz'])]

    with pytest.raises(ValueError):
        devices = check_select_devices(
            mock_stdin, mock_backend_factory, capsys, platforms_devices,
            device_exclude_masks=['Device'])


@pytest.mark.parametrize('unique_only', [False, True], ids=["unique_only=False", "unique_only=True"])
def test_unique_devices_only(mock_stdin, mock_backend_factory, capsys, unique_only):

    platforms_devices = [
        ('PlatformFoo', ['DeviceFoo', 'DeviceFoo']),
        ('PlatformBar', ['DeviceBar', 'DeviceBar', 'DeviceBaz'])]

    devices = check_select_devices(
        mock_stdin, mock_backend_factory, capsys, platforms_devices,
        inputs=['1', '1'], interactive=True, unique_devices_only=unique_only)

    assert len(devices) == 1
    assert devices[0].name == 'DeviceBaz' if unique_only else 'DeviceBar'


@pytest.mark.parametrize('include_pp', [False, True], ids=["include_pp=False", "include_pp=True"])
def test_include_pure_parallel_devices(mock_stdin, mock_backend_factory, capsys, include_pp):

    # `check_select_devices()` mocks OpenCL, so we can use multiple platforms
    # and OpenCL-specific device info
    platforms_devices = [
        ('PlatformFoo', ['Device1', 'Device2']),
        ('PlatformBar', [
            PyOpenCLDeviceInfo(name='Device3', max_work_group_size=1), 'Device4', 'Device5'])]

    devices = check_select_devices(
        mock_stdin, mock_backend_factory, capsys, platforms_devices,
        inputs=['1', '1'], interactive=True, include_pure_parallel_devices=include_pp)

    assert len(devices) == 1
    assert devices[0].name == 'Device4' if include_pp else 'Device5'
