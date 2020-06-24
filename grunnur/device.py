from __future__ import annotations

from typing import Optional, Iterable, List, Sequence

from .api import API
from .platform import Platform
from .utils import string_matches_masks


class Device:
    """
    A generalized GPGPU device.

    .. py:attribute:: platform

        The :py:class:`Platform` object this device belongs to.

    .. py:attribute:: id

        This device's ID, a :py:class:`DeviceID` object.

    .. py:attribute:: short_name

        This device's short name.
    """

    @classmethod
    def all(cls, platform):
        return [
            Device.from_index(platform, device_idx)
            for device_idx in range(platform._platform_adapter.device_count)]

    @classmethod
    def all_by_masks(
            cls,
            platform,
            include_masks: Optional[Sequence[str]]=None,
            exclude_masks: Optional[Sequence[str]]=None,
            unique_only: bool=False,
            include_pure_parallel_devices: bool=False) \
            -> List[Device]:
        """
        Returns a list of all devices satisfying the given criteria.

        :param include_masks: a list of strings (treated as regexes),
            one of which must match with the device name.
        :param exclude_masks: a list of strings (treated as regexes),
            neither of which must match with the device name.
        :param unique_only: if ``True``, only return devices with unique names.
        :param include_pure_parallel_devices: if ``True``, include devices with
            :py:meth:`~Device.max_total_local_size` equal to 1.
        """

        seen_devices = set()
        devices = []

        for device in cls.all(platform):
            if not string_matches_masks(
                    device.name, include_masks=include_masks, exclude_masks=exclude_masks):
                continue

            if unique_only and device.name in seen_devices:
                continue

            if not include_pure_parallel_devices and device.params.max_total_local_size == 1:
                continue

            seen_devices.add(device.name)
            devices.append(device)

        return devices

    @classmethod
    def from_backend_device(cls, obj):
        for api in API.all():
            if api._api_adapter.isa_backend_device(obj):
                device_adapter = api._api_adapter.make_device_adapter(obj)
                return cls(device_adapter)

        raise TypeError(f"{obj} was not recognized as a device object by any available API")

    @classmethod
    def from_index(cls, platform, device_idx):
        device_adapter = platform._platform_adapter.get_device_adapters()[device_idx]
        return cls(device_adapter)

    def __init__(self, device_adapter):
        self.platform = Platform(device_adapter.platform_adapter)
        self._device_adapter = device_adapter
        self.name = self._device_adapter.name

        self.shortcut = f"{self.platform.shortcut},{device_adapter.device_idx}"
        self.short_name = f"device({self.shortcut})"

        self._params = None

    def __eq__(self, other):
        return isinstance(other, Device) and self._device_adapter == other._device_adapter

    def __hash__(self):
        return hash((type(self), self._device_adapter))

    @property
    def params(self):
        # Already cached in the adapters
        return self._device_adapter.params
