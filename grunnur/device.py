from __future__ import annotations

from typing import Any, NamedTuple, Optional, List, Sequence

from .api import API
from .adapter_base import DeviceParameters, DeviceAdapter
from .platform import Platform
from .utils import string_matches_masks


class DeviceFilter(NamedTuple):
    """
    A set of filters for device discovery.
    """

    include_masks: Optional[List[str]] = None
    """A list of strings (treated as regexes), one of which must match the device name."""

    exclude_masks: Optional[List[str]] = None
    """A list of strings (treated as regexes), neither of which must match the device name."""

    unique_only: bool = False
    """If ``True``, only return devices with unique names."""

    exclude_pure_parallel: bool = False
    """
    If ``True``, exclude devices with
    :py:attr:`params.max_total_local_size <grunnur.adapter_base.DeviceParameters.max_total_local_size>`
    equal to 1.
    """


class Device:
    """
    A generalized GPGPU device.
    """

    platform: "Platform"
    """The :py:class:`~grunnur.Platform` object this device belongs to."""

    name: str
    """This device's name."""

    @classmethod
    def all(cls, platform: "Platform") -> List["Device"]:
        """
        Returns a list of devices available for the given platform.

        :param platform: the platform to search in.
        """
        return [
            Device.from_index(platform, device_idx)
            for device_idx in range(platform._platform_adapter.device_count)
        ]

    @classmethod
    def all_filtered(
        cls,
        platform: "Platform",
        filter: Optional[DeviceFilter] = None,
    ) -> List["Device"]:
        """
        Returns a list of all devices satisfying the given criteria in the given platform.
        If ``filter`` is not provided, returns all the devices.
        """
        if filter is None:
            return cls.all(platform)

        seen_devices = set()
        devices = []

        for device in cls.all(platform):
            if not string_matches_masks(
                device.name, include_masks=filter.include_masks, exclude_masks=filter.exclude_masks
            ):
                continue

            if filter.unique_only and device.name in seen_devices:
                continue

            if filter.exclude_pure_parallel and device.params.max_total_local_size == 1:
                continue

            seen_devices.add(device.name)
            devices.append(device)

        return devices

    @classmethod
    def from_backend_device(cls, obj: Any) -> "Device":
        """
        Wraps a backend device object into a Grunnur device object.
        """
        for api in API.all_available():
            if api._api_adapter.isa_backend_device(obj):
                device_adapter = api._api_adapter.make_device_adapter(obj)
                return cls(device_adapter)

        raise TypeError(f"{obj} was not recognized as a device object by any available API")

    @classmethod
    def from_index(cls, platform: "Platform", device_idx: int) -> "Device":
        """
        Creates a device based on its index in the list returned by the API.

        :param platform: the API to search in.
        :param device_idx: the target device's index.
        """
        device_adapter = platform._platform_adapter.get_device_adapters()[device_idx]
        return cls(device_adapter)

    def __init__(self, device_adapter: DeviceAdapter):
        self.platform = Platform(device_adapter.platform_adapter)
        self._device_adapter = device_adapter
        self.name = self._device_adapter.name

        self.shortcut = f"{self.platform.shortcut},{device_adapter.device_idx}"

        self._params = None

    def __eq__(self, other: Any) -> bool:
        return (
            type(self) == type(other)
            and isinstance(other, Device)
            and self._device_adapter == other._device_adapter
        )

    def __hash__(self) -> int:
        return hash((type(self), self._device_adapter))

    @property
    def params(self) -> DeviceParameters:
        """
        Returns a :py:class:`~grunnur.adapter_base.DeviceParameters` object
        associated with this device.
        """
        # Already cached in the adapters
        return self._device_adapter.params

    def __str__(self) -> str:
        return f"device({self.shortcut})"
