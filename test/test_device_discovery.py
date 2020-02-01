import pytest

from grunnur.api_discovery import all_api_factories
from grunnur.opencl import OPENCL_API_FACTORY

from .utils import mock_backend, mock_input


def check_select_devices(monkeypatch, capsys, platforms_devices, inputs=None, **kwds):

    # CUDA API has a single fixed platform, so using the OpenCL one
    api_factory = OPENCL_API_FACTORY

    mock_backend(monkeypatch, api_factory, platforms_devices)

    if inputs is not None:
        mock_input(monkeypatch, inputs)

    api = api_factory.make_api()

    try:
        devices = api.select_devices(**kwds)
    finally:
        if inputs is not None:
            # Otherwise the output will be shown in the console
            captured = capsys.readouterr()

    return devices


def test_platform_take_single(monkeypatch, capsys):
    # Test that if only one platform is available, it is selected automatically

    platforms_devices = [
        ('Platform1', ['Device1', 'Device2'])]
    inputs = ['1']
    devices = check_select_devices(
        monkeypatch, capsys, platforms_devices, inputs=inputs, interactive=True)

    assert len(devices) == 1
    assert devices[0].name == 'Device2'


def test_platform_choose(monkeypatch, capsys):
    # Test selecting one of several available platforms

    platforms_devices = [
        ('Platform1', ['Device1', 'Device2']),
        ('Platform2', ['Device3', 'Device4'])]
    inputs = ['1', '0']
    devices = check_select_devices(
        monkeypatch, capsys, platforms_devices, inputs=inputs, interactive=True)

    assert len(devices) == 1
    assert devices[0].name == 'Device3'


def test_platform_take_default(monkeypatch, capsys):
    # Test that just pressing enter gives the default (first) platform

    platforms_devices = [
        ('Platform1', ['Device1', 'Device2']),
        ('Platform2', ['Device3', 'Device4'])]
    inputs = ['', '0']
    devices = check_select_devices(
        monkeypatch, capsys, platforms_devices, inputs=inputs, interactive=True)

    assert len(devices) == 1
    assert devices[0].name == 'Device1'


def test_device_take_single(monkeypatch, capsys):
    # Test that if only one device is available, it is selected automatically

    platforms_devices = [
        ('Platform1', ['Device1', 'Device2']),
        ('Platform2', ['Device3'])]
    inputs = ['1']
    devices = check_select_devices(
        monkeypatch, capsys, platforms_devices, inputs=inputs, interactive=True)

    assert len(devices) == 1
    assert devices[0].name == 'Device3'


def test_device_take_multiple(monkeypatch, capsys):
    # Test the case when several devices are requested,
    # and that's exactly how many the chosen platform has.
    # Technically it's the same as a single device and quantity=1, but testing just in case.

    platforms_devices = [
        ('Platform1', ['Device1', 'Device2']),
        ('Platform2', ['Device3', 'Device4', 'Device5'])]
    inputs = ['1']
    devices = check_select_devices(
        monkeypatch, capsys, platforms_devices, inputs=inputs, interactive=True, quantity=3)

    assert len(devices) == 3
    assert [device.name for device in devices] == ['Device3', 'Device4', 'Device5']


def test_device_choose_single(monkeypatch, capsys):
    # Test that if only one device is available, it is selected automatically

    platforms_devices = [
        ('Platform1', ['Device1', 'Device2']),
        ('Platform2', ['Device3', 'Device4'])]
    inputs = ['1', '1']
    devices = check_select_devices(
        monkeypatch, capsys, platforms_devices, inputs=inputs, interactive=True)

    assert len(devices) == 1
    assert devices[0].name == 'Device4'


def test_device_choose_multiple(monkeypatch, capsys):
    # Test that if only one device is available, it is selected automatically

    platforms_devices = [
        ('Platform1', ['Device1', 'Device2']),
        ('Platform2', ['Device3', 'Device4', 'Device5'])]
    inputs = ['1', '0, 2']
    devices = check_select_devices(
        monkeypatch, capsys, platforms_devices, inputs=inputs, interactive=True, quantity=2)

    assert len(devices) == 2
    assert [device.name for device in devices] == ['Device3', 'Device5']


def test_device_choose_multiple_wrong_quantity(monkeypatch, capsys):
    # Test error in the case when the user selects a number of devices not equal to `quantity`

    platforms_devices = [
        ('Platform1', ['Device1', 'Device2']),
        ('Platform2', ['Device3', 'Device4', 'Device5'])]
    inputs = ['1', '0, 1, 2']

    with pytest.raises(ValueError):
        devices = check_select_devices(
            monkeypatch, capsys, platforms_devices, inputs=inputs, interactive=True, quantity=2)


def test_device_take_default_single(monkeypatch, capsys):
    # Test that just pressing enter gives the default (first) device

    platforms_devices = [
        ('Platform1', ['Device1', 'Device2']),
        ('Platform2', ['Device3', 'Device4'])]
    inputs = ['1', '']
    devices = check_select_devices(
        monkeypatch, capsys, platforms_devices, inputs=inputs, interactive=True)

    assert len(devices) == 1
    assert devices[0].name == 'Device3'


def test_device_take_default_multiple(monkeypatch, capsys):
    # Test that just pressing enter gives the first `quantity` devices

    platforms_devices = [
        ('Platform1', ['Device1', 'Device2']),
        ('Platform2', ['Device3', 'Device4', 'Device5'])]
    inputs = ['1', '']
    devices = check_select_devices(
        monkeypatch, capsys, platforms_devices, inputs=inputs, interactive=True, quantity=2)

    assert len(devices) == 2
    assert [device.name for device in devices] == ['Device3', 'Device4']


def test_device_take_all_available(monkeypatch, capsys):
    # Test that `quantity=None` returns all suitable devices.

    platforms_devices = [
        ('Platform1', ['Device1', 'Device2']),
        ('Platform2', ['Device3', 'Device4', 'Device5'])]
    inputs = ['1']
    devices = check_select_devices(
        monkeypatch, capsys, platforms_devices, inputs=inputs, interactive=True, quantity=None)

    assert len(devices) == 3
    assert [device.name for device in devices] == ['Device3', 'Device4', 'Device5']


def test_filter_include_platforms(monkeypatch, capsys):

    platforms_devices = [
        ('PlatformFoo', ['Device1', 'Device2']),
        ('PlatformBar', ['Device3', 'Device4', 'Device5'])]
    devices = check_select_devices(
        monkeypatch, capsys, platforms_devices,
        platform_include_masks=['Bar'])

    assert len(devices) == 1
    assert devices[0].name == 'Device3'


def test_filter_exclude_platforms(monkeypatch, capsys):

    platforms_devices = [
        ('PlatformFoo', ['Device1', 'Device2']),
        ('PlatformBar', ['Device3', 'Device4', 'Device5'])]
    devices = check_select_devices(
        monkeypatch, capsys, platforms_devices,
        platform_exclude_masks=['Foo'])

    assert len(devices) == 1
    assert devices[0].name == 'Device3'


def test_filter_include_devices(monkeypatch, capsys):

    platforms_devices = [
        ('PlatformFoo', ['DeviceFoo', 'DeviceBar']),
        ('PlatformBar', ['DeviceFoo', 'DeviceBar', 'DeviceBaz'])]
    devices = check_select_devices(
        monkeypatch, capsys, platforms_devices,
        inputs=['1'], interactive=True, device_include_masks=['Foo'])

    assert len(devices) == 1
    assert devices[0].name == 'DeviceFoo'
    assert devices[0].platform.name == 'PlatformBar'


def test_filter_exclude_devices(monkeypatch, capsys):

    platforms_devices = [
        ('PlatformFoo', ['DeviceFoo', 'DeviceBar']),
        ('PlatformBar', ['DeviceFoo', 'DeviceBar', 'DeviceBaz'])]
    devices = check_select_devices(
        monkeypatch, capsys, platforms_devices,
        inputs=['1', '1'], interactive=True, device_exclude_masks=['Bar'])

    assert len(devices) == 1
    assert devices[0].name == 'DeviceBaz'


def test_filter_exclude_all_devices(monkeypatch, capsys):

    platforms_devices = [
        ('PlatformFoo', ['DeviceFoo', 'DeviceBar']),
        ('PlatformBar', ['DeviceFoo', 'DeviceBar', 'DeviceBaz'])]

    with pytest.raises(ValueError):
        devices = check_select_devices(
            monkeypatch, capsys, platforms_devices,
            device_exclude_masks=['Device'])


def test_unique_devices_only(monkeypatch, capsys):

    platforms_devices = [
        ('PlatformFoo', ['DeviceFoo', 'DeviceFoo']),
        ('PlatformBar', ['DeviceBar', 'DeviceBar', 'DeviceBaz'])]

    devices = check_select_devices(
        monkeypatch, capsys, platforms_devices,
        inputs=['1', '1'], interactive=True, unique_devices_only=True)

    assert len(devices) == 1
    assert devices[0].name == 'DeviceBaz'

    devices = check_select_devices(
        monkeypatch, capsys, platforms_devices,
        inputs=['1', '1'], interactive=True, unique_devices_only=False)

    assert len(devices) == 1
    assert devices[0].name == 'DeviceBar'


def test_include_pure_parallel_devices(monkeypatch, capsys):

    platforms_devices = [
        ('PlatformFoo', ['Device1', 'Device2']),
        ('PlatformBar', [
            dict(name='Device3', max_work_group_size=1), 'Device4', 'Device5'])]

    devices = check_select_devices(
        monkeypatch, capsys, platforms_devices,
        inputs=['1', '1'], interactive=True, include_pure_parallel_devices=True)

    assert len(devices) == 1
    assert devices[0].name == 'Device4'

    devices = check_select_devices(
        monkeypatch, capsys, platforms_devices,
        inputs=['1', '1'], interactive=True, include_pure_parallel_devices=False)

    assert len(devices) == 1
    assert devices[0].name == 'Device5'
