from functools import lru_cache
from typing import Iterable, TypeVar, Optional, List, Iterator, Any, Tuple, cast

import pytest

from .api import API, all_api_ids
from .platform import Platform, PlatformFilter
from .device import Device, DeviceFilter
from .context import Context


def pytest_addoption(parser: pytest.Parser) -> None:

    api_shortcuts = [api_id.shortcut for api_id in all_api_ids()]
    parser.addoption(
        "--api",
        action="store",
        help="GPGPU API: " + "/".join(api_shortcuts) + " (or all available if not given)",
        default=None,
        choices=api_shortcuts,
    )

    parser.addoption(
        "--platform-include-mask",
        action="append",
        help="Run tests on matching platforms only",
        default=[],
    )
    parser.addoption(
        "--platform-exclude-mask",
        action="append",
        help="Run tests on matching platforms only",
        default=[],
    )

    parser.addoption(
        "--device-include-mask",
        action="append",
        help="Run tests on matching devices only",
        default=[],
    )
    parser.addoption(
        "--device-exclude-mask",
        action="append",
        help="Run tests on matching devices only",
        default=[],
    )

    parser.addoption(
        "--include-duplicate-devices",
        action="store_true",
        help="Run tests on all available devices and not only on uniquely named ones",
        default=False,
    )

    parser.addoption(
        "--include-pure-parallel-devices",
        action="store_true",
        help="Include pure parallel devices (not supporting synchronization within a work group)",
        default=False,
    )


@lru_cache()
def get_apis(config: pytest.Config) -> List[API]:
    return API.all_by_shortcut(config.option.api)


_T = TypeVar("_T")


def concatenate(lists: Iterable[List[_T]]) -> List[_T]:
    return sum(lists, [])


@lru_cache()
def get_platforms(config: pytest.Config) -> List[Platform]:
    apis = get_apis(config)
    return concatenate(
        Platform.all_filtered(
            api,
            PlatformFilter(
                include_masks=config.option.platform_include_mask,
                exclude_masks=config.option.platform_exclude_mask,
            ),
        )
        for api in apis
    )


@lru_cache()
def get_device_sets(
    config: pytest.Config, unique_devices_only_override: Optional[bool] = None
) -> List[List[Device]]:

    if unique_devices_only_override is not None:
        unique_devices_only = unique_devices_only_override
    else:
        unique_devices_only = not config.option.include_duplicate_devices

    platforms = get_platforms(config)
    return [
        Device.all_filtered(
            platform,
            DeviceFilter(
                include_masks=config.option.device_include_mask,
                exclude_masks=config.option.device_exclude_mask,
                unique_only=unique_devices_only,
                exclude_pure_parallel=not config.option.include_pure_parallel_devices,
            ),
        )
        for platform in platforms
    ]


@lru_cache()
def get_devices(config: pytest.Config) -> List[Device]:
    return concatenate(get_device_sets(config))


@lru_cache()
def get_multi_device_sets(config: pytest.Config) -> List[List[Device]]:
    device_sets = get_device_sets(config, unique_devices_only_override=False)
    return [device_set for device_set in device_sets if len(device_set) > 1]


@pytest.fixture
def context(device: Device) -> Iterator[Context]:
    yield Context.from_devices([device])


@pytest.fixture
def some_context(some_device: Device) -> Iterator[Context]:
    yield Context.from_devices([some_device])


@pytest.fixture
def multi_device_context(device_set: List[Device]) -> Iterator[Context]:
    yield Context.from_devices(device_set)


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:

    apis = get_apis(metafunc.config)
    platforms = get_platforms(metafunc.config)
    devices = get_devices(metafunc.config)

    api_ids = [api.id for api in apis]

    fixtures: List[Tuple[str, List[Any]]] = [
        ("api", apis),
        ("platform", platforms),
        ("device", devices),
    ]

    for name, vals in fixtures:
        if name in metafunc.fixturenames:
            metafunc.parametrize(
                name,
                vals,
                ids=["no_" + name] if len(vals) == 0 else lambda obj: cast(str, obj.shortcut),
            )

    if "some_device" in metafunc.fixturenames:
        metafunc.parametrize(
            "some_device",
            devices if len(devices) == 0 else [devices[0]],
            ids=["no_device"] if len(devices) == 0 else lambda device: cast(str, device.shortcut),
        )

    if "device_set" in metafunc.fixturenames:
        device_sets = get_multi_device_sets(metafunc.config)
        ids = ["+".join(device.shortcut for device in device_set) for device_set in device_sets]
        metafunc.parametrize(
            "device_set", device_sets, ids=["no_multi_device"] if len(device_sets) == 0 else ids
        )


def pytest_report_header(config: pytest.Config) -> None:
    devices = get_devices(config)

    if len(devices) == 0:
        print("No GPGPU devices available")
    else:
        print("Running tests on:")
        for device in sorted(devices, key=lambda device: str(device)):
            platform = device.platform
            print(f"  {device}: {platform.name}, {device.name}")
