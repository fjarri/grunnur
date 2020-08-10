from typing import Optional, Tuple, List, Sequence

from .platform import Platform
from .device import Device


def platforms_and_devices_by_mask(
        api,
        quantity: Optional[int]=1,
        platform_include_masks: Optional[Sequence[str]]=None,
        platform_exclude_masks: Optional[Sequence[str]]=None,
        device_include_masks: Optional[Sequence[str]]=None,
        device_exclude_masks: Optional[Sequence[str]]=None,
        unique_devices_only: bool=False,
        include_pure_parallel_devices: bool=False) \
        -> List[Tuple[Platform, List[Device]]]:
    """
    Returns all tuples (platform, list of devices) where the platform name and device names
    satisfy the given criteria, and there are at least ``quantity`` devices in the list.

    :param quantity: the number of devices to find. If ``None``,
        find all matching devices belonging to a single platform.
    :param platform_include_masks: passed to :py:meth:`Platform.all_by_masks`.
    :param platform_exclude_masks: passed to :py:meth:`Platform.all_by_masks`.
    :param device_include_masks: passed to :py:meth:`Device.all_by_masks`.
    :param device_exclude_masks: passed to :py:meth:`Device.all_by_masks`.
    :param unique_devices_only: passed to :py:meth:`Device.all_by_masks`.
    :param include_pure_parallel_devices: passed to :py:meth:`Device.all_by_masks`.
    """

    results = []

    suitable_platforms = Platform.all_by_masks(
        api, include_masks=platform_include_masks, exclude_masks=platform_exclude_masks)

    for platform in suitable_platforms:

        suitable_devices = Device.all_by_masks(
            platform,
            include_masks=device_include_masks, exclude_masks=device_exclude_masks,
            unique_only=unique_devices_only,
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
    Using the results from :py:func:`platforms_and_devices_by_mask`, either lets the user
    select the devices (from the ones matching the criteria) interactively,
    or takes the first matching list of ``quantity`` devices.

    :param interactive: if ``True``, shows a dialog to select the devices.
        If ``False``, selects the first matching ones.
    :param quantity: passed to :py:func:`platforms_and_devices_by_mask`.
    :param device_filters: passed to :py:func:`platforms_and_devices_by_mask`.
    """
    suitable_pds = platforms_and_devices_by_mask(api, quantity, **device_filters)

    if len(suitable_pds) == 0:
        quantity_val = "any" if quantity is None else quantity
        raise ValueError(
            f"Could not find {quantity_val} devices on a single platform "
            "matching the given criteria")

    if interactive:
        return _select_devices_interactive(suitable_pds, quantity=quantity)

    _, devices = suitable_pds[0]
    return devices if quantity is None else devices[:quantity]
