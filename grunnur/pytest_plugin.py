from functools import lru_cache

import pytest

from .api import API, all_api_ids
from .platform import Platform
from .device import Device
from .context import Context


def pytest_addoption(parser):

    api_shortcuts = [api_id.shortcut for api_id in all_api_ids()]
    parser.addoption(
        "--api",
        action="store",
        help="GPGPU API: " + "/".join(api_shortcuts) + " (or all available if not given)",
        default=None, choices=api_shortcuts)

    parser.addoption(
        "--platform-include-mask",
        action="append",
        help="Run tests on matching platforms only",
        default=[])
    parser.addoption(
        "--platform-exclude-mask",
        action="append",
        help="Run tests on matching platforms only",
        default=[])

    parser.addoption(
        "--device-include-mask",
        action="append",
        help="Run tests on matching devices only",
        default=[])
    parser.addoption(
        "--device-exclude-mask",
        action="append",
        help="Run tests on matching devices only",
        default=[])

    parser.addoption(
        "--include-duplicate-devices",
        action="store_true",
        help="Run tests on all available devices and not only on uniquely named ones",
        default=False)


@lru_cache()
def get_apis(config):
    return API.all_by_shortcut(config.option.api)


def concatenate(lists):
    return sum(lists, [])


@lru_cache()
def get_platforms(config):
    apis = get_apis(config)
    return concatenate(
        Platform.all_by_masks(
            api,
            include_masks=config.option.platform_include_mask,
            exclude_masks=config.option.platform_exclude_mask)
        for api in apis)


@lru_cache()
def get_device_sets(config, unique_devices_only_override=None):

    if unique_devices_only_override is not None:
        unique_devices_only = unique_devices_only_override
    else:
        unique_devices_only = not config.option.include_duplicate_devices

    platforms = get_platforms(config)
    return [
        Device.all_by_masks(
            platform,
            include_masks=config.option.device_include_mask,
            exclude_masks=config.option.device_exclude_mask,
            unique_only=unique_devices_only)
        for platform in platforms]


@lru_cache()
def get_devices(config):
    return concatenate(get_device_sets(config))


@lru_cache()
def get_multi_device_sets(config):
    device_sets = get_device_sets(config, unique_devices_only_override=False)
    return [device_set for device_set in device_sets if len(device_set) > 1]


@pytest.fixture(scope='function')
def context(request):
    device = request.param
    yield Context.from_devices(device)


@pytest.fixture(scope='function')
def multi_device_context(request):
    devices = request.param
    yield Context.from_devices(devices)


def pytest_generate_tests(metafunc):

    apis = get_apis(metafunc.config)
    platforms = get_platforms(metafunc.config)
    devices = get_devices(metafunc.config)

    api_ids = [api.id for api in apis]

    fixtures = [
        ('api', apis),
        ('platform', platforms),
        ('device', devices),
        ]

    for name, vals in fixtures:
        if name in metafunc.fixturenames:
            metafunc.parametrize(name, vals, ids=['no_' + name] if len(vals) == 0 else str)

    if 'context' in metafunc.fixturenames:
        metafunc.parametrize(
            'context', devices,
            ids=['no_device'] if len(devices) == 0 else lambda device: device.shortcut,
            indirect=True)

    if 'multi_device_context' in metafunc.fixturenames:
        device_sets = get_multi_device_sets(metafunc.config)
        ids = ["+".join(device.shortcut for device in device_set) for device_set in device_sets]
        metafunc.parametrize(
            'multi_device_context', device_sets,
            ids=['no_multi_device'] if len(device_sets) == 0 else ids,
            indirect=True)


def pytest_report_header(config):
    devices = get_devices(config)

    if len(devices) == 0:
        print("No GPGPU devices available")
    else:
        print("Running tests on:")
        for device in sorted(devices, key=lambda device: str(device)):
            platform = device.platform
            print(f"  {device}: {platform.name}, {device.name}")
