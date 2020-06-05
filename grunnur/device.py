from __future__ import annotations

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
            Device.from_number(platform, device_num)
            for device_num in range(platform._platform_adapter.num_devices)]

    @classmethod
    def all_by_masks(
            cls,
            platform,
            include_masks: Optional[Iterable[str]]=None,
            exclude_masks: Optional[Iterable[str]]=None,
            unique_devices_only: bool=False,
            include_pure_parallel_devices: bool=False) \
            -> List[Device]:
        """
        Returns a list of all devices satisfying the given criteria.

        :param include_masks: a list of strings (treated as regexes),
            one of which must match with the device name.
        :param exclude_masks: a list of strings (treated as regexes),
            neither of which must match with the device name.
        :param unique_devices_only: if ``True``, only return devices with unique names.
        :param include_pure_parallel_devices: if ``True``, include devices with
            :py:meth:`~Device.max_total_local_size` equal to 1.
        """

        seen_devices = set()
        devices = []

        for device in cls.all(platform):
            if not string_matches_masks(
                    device.name, include_masks=include_masks, exclude_masks=exclude_masks):
                continue

            if unique_devices_only and device.name in seen_devices:
                continue

            if not include_pure_parallel_devices and device.params.max_total_local_size == 1:
                continue

            seen_devices.add(device.name)
            devices.append(device)

        return devices

    @classmethod
    def from_number(cls, platform, device_num):
        device_adapter = platform._platform_adapter.get_device_adapters()[device_num]
        return cls(device_adapter)

    def __init__(self, device_adapter):
        self.platform = Platform(device_adapter.platform_adapter)
        self._device_adapter = device_adapter
        self.name = self._device_adapter.name

        self.shortcut = f"{self.platform.shortcut},{device_adapter.device_num}"
        self.short_name = f"device({self.shortcut})"

        self._params = None

    @property
    def params(self):
        if self._params is None:
            self._params = self._device_adapter.params
        return self._params


def platforms_and_devices_by_mask(
        api,
        quantity: Optional[int]=1,
        platform_include_masks: Optional[Iterable[str]]=None,
        platform_exclude_masks: Optional[Iterable[str]]=None,
        device_include_masks: Optional[Iterable[str]]=None,
        device_exclude_masks: Optional[Iterable[str]]=None,
        unique_devices_only: bool=False,
        include_pure_parallel_devices: bool=False) \
        -> List[Tuple[Platform, List[Device]]]:
    """
    Returns all tuples (platform, list of devices) where the platform name and device names
    satisfy the given criteria, and there are at least ``quantity`` devices in the list.

    :param quantity: the number of devices to find. If ``None``,
        find all matching devices belonging to a single platform.
    :param platform_include_masks: passed to :py:meth:`find_platforms`.
    :param platform_exclude_masks: passed to :py:meth:`find_platforms`.
    :param device_include_masks: passed to :py:meth:`Platform.find_devices`.
    :param device_exclude_masks: passed to :py:meth:`Platform.find_devices`.
    :param unique_devices_only: passed to :py:meth:`Platform.find_devices`.
    :param include_pure_parallel_devices: passed to :py:meth:`Platform.find_devices`.
    """

    results = []

    suitable_platforms = Platform.all_by_masks(
        api, include_masks=platform_include_masks, exclude_masks=platform_exclude_masks)

    for platform in suitable_platforms:

        suitable_devices = Device.all_by_masks(
            platform,
            include_masks=device_include_masks, exclude_masks=device_exclude_masks,
            unique_devices_only=unique_devices_only,
            include_pure_parallel_devices=include_pure_parallel_devices)

        if ((quantity is None and len(suitable_devices) > 0) or
                (quantity is not None and len(suitable_devices) >= quantity)):
            results.append((platform, suitable_devices))

    return results


def _select_devices_interactive(suitable_pds, quantity=1):

    if len(suitable_pds) == 1:
        platform, devices = suitable_pds[0]
        print(f"Platform: {platform.name}")
    else:
        print("Platforms:")
        for pnum, pd in enumerate(suitable_pds):
            platform, _ = pd
            print(f"[{pnum}]: {platform.name}")

        default_pnum = 0
        print(f"Choose the platform [{default_pnum}]: ", end='')
        selected_pnum = input()
        if selected_pnum == '':
            selected_pnum = default_pnum
        else:
            selected_pnum = int(selected_pnum)

        platform, devices = suitable_pds[selected_pnum]

    if quantity is None or (quantity is not None and len(devices) == quantity):
        selected_devices = devices
        print(f"Devices: {[device.name for device in selected_devices]}")
    else:
        print("Devices:")
        default_dnums = list(range(quantity))
        for dnum, device in enumerate(devices):
            print(f"[{dnum}]: {device.name}")
        default_dnums_str = ', '.join(str(dnum) for dnum in default_dnums)
        print(f"Choose the device(s), comma-separated [{default_dnums_str}]: ", end='')
        selected_dnums = input()
        if selected_dnums == '':
            selected_dnums = default_dnums
        else:
            selected_dnums = [int(dnum) for dnum in selected_dnums.split(',')]
            if len(selected_dnums) != quantity:
                raise ValueError(f"Exactly {quantity} devices must be selected")

        selected_devices = [devices[dnum] for dnum in selected_dnums]

    return selected_devices


def select_devices(
        api, interactive: bool=False, quantity: Optional[int]=1,
        **device_filters) -> List[Device]:
    """
    Using the results from :py:meth:`find_devices`, either lets the user
    select the devices (from the ones matching the criteria) interactively,
    or takes the first matching list of ``quantity`` devices.

    :param interactive: if ``True``, shows a dialog to select the devices.
        If ``False``, selects the first matching ones.
    :param quantity: passed to :py:meth:`find_devices`.
    :param device_filters: passed to :py:meth:`find_devices`.
    """
    suitable_pds = platforms_and_devices_by_mask(api, quantity, **device_filters)

    if len(suitable_pds) == 0:
        quantity_val = "any" if quantity is None else quantity
        raise ValueError(
            f"Could not find {quantity_val} devices on a single platform "
            "matching the given criteria")

    if interactive:
        return _select_devices_interactive(suitable_pds, quantity=quantity)
    else:
        _, devices = suitable_pds[0]
        return devices if quantity is None else devices[:quantity]
