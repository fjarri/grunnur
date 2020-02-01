from functools import lru_cache

import pytest

from .api_discovery import all_api_factories, find_apis


def addoption(parser):

    api_shortcuts = [api_factory.api_id.shortcut for api_factory in all_api_factories()]
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
        action="store_false",
        help="Run tests on all available devices and not only on uniquely named ones",
        default=False)


@lru_cache()
def get_apis(config):
    return find_apis(config.option.api)


def concatenate(lists):
    return sum(lists, [])


@lru_cache()
def get_platforms(config):
    apis = get_apis(config)
    return concatenate(
        api.find_platforms(
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
        platform.find_devices(
            include_masks=config.option.device_include_mask,
            exclude_masks=config.option.device_exclude_mask,
            unique_devices_only=unique_devices_only)
        for platform in platforms]


@lru_cache()
def get_devices(config):
    return concatenate(get_device_sets(config))


@lru_cache()
def get_multi_device_sets(config):
    device_sets = get_device_sets(config, unique_devices_only_override=False)
    return [device_set for device_set in device_sets if len(device_set) > 1]


@pytest.fixture
def context(request):
    device = request.param
    api = device.platform.api
    context = api.create_context([device])
    yield context


@pytest.fixture
def multi_device_context(request):
    devices = request.param
    api = devices[0].platform.api
    context = api.create_context(devices)
    yield context


def generate_tests(metafunc):

    api_factories = all_api_factories()
    apis = get_apis(metafunc.config)
    platforms = get_platforms(metafunc.config)
    devices = get_devices(metafunc.config)

    api_ids = [api.id for api in apis]
    platform_ids = [platform.id for platform in platforms]
    device_ids = [device.id for device in devices]

    idgen = lambda val: val.short_name

    fixtures = [
        ('api_factory', api_factories),
        ('api_id', api_ids),
        ('api', apis),
        ('platform_id', platform_ids),
        ('platform', platforms),
        ('device_id', device_ids),
        ('device', devices)]

    for name, vals in fixtures:
        if name in metafunc.fixturenames:
            if len(vals) == 0:
                metafunc.parametrize(name, [])
            else:
                metafunc.parametrize(name, vals, ids=idgen)

    if 'context' in metafunc.fixturenames:
        if len(devices) == 0:
            metafunc.parametrize('context', [])
        else:
            metafunc.parametrize(
                'context', devices, ids=lambda device: device.id.shortcut, indirect=True)

    if 'multi_device_context' in metafunc.fixturenames:
        device_sets = get_multi_device_sets(metafunc.config)
        ids = ["+".join(device.id.shortcut for device in device_set) for device_set in device_sets]
        metafunc.parametrize('multi_device_context', device_sets, ids=ids, indirect=True)


def report_header(config):
    devices = get_devices(config)

    if len(devices) == 0:
        print("No GPGPU devices available")
    else:
        print("Running tests on:")
        for device in sorted(devices, key=lambda device: device.short_name):
            platform = device.platform
            print(f"  {device.short_name}: {platform.name}, {device.name}")
