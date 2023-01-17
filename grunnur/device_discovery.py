from typing import Optional, Tuple, List, Sequence

from .api import API
from .platform import Platform, PlatformFilter
from .device import Device, DeviceFilter


def platforms_and_devices_by_mask(
    api: API,
    quantity: Optional[int] = 1,
    device_filter: Optional[DeviceFilter] = None,
    platform_filter: Optional[PlatformFilter] = None,
) -> List[Tuple["Platform", List["Device"]]]:
    """
    Returns all tuples (platform, list of devices) where the platform name and device names
    satisfy the given criteria, and there are at least ``quantity`` devices in the list.
    """

    results = []

    suitable_platforms = Platform.all_filtered(api, platform_filter)

    for platform in suitable_platforms:

        suitable_devices = Device.all_filtered(platform, device_filter)

        if (quantity is None and len(suitable_devices) > 0) or (
            quantity is not None and len(suitable_devices) >= quantity
        ):
            results.append((platform, suitable_devices))

    return results


def _select_devices_interactive(
    suitable_pds: Sequence[Tuple[Platform, Sequence[Device]]], quantity: Optional[int] = 1
) -> List[Device]:

    if len(suitable_pds) == 1:
        platform, devices = suitable_pds[0]
        print(f"Platform: {platform.name}")
    else:
        print("Platforms:")
        for pnum, pd in enumerate(suitable_pds):
            platform, _ = pd
            print(f"[{pnum}]: {platform.name}")

        default_pnum = 0
        selected_pnum = default_pnum
        print(f"Choose the platform [{default_pnum}]: ", end="")
        pnum_input = input()
        if pnum_input != "":
            selected_pnum = int(pnum_input)

        platform, devices = suitable_pds[selected_pnum]

    if quantity is None or (quantity is not None and len(devices) == quantity):
        selected_devices = list(devices)
        print(f"Devices: {[device.name for device in selected_devices]}")
    else:
        print("Devices:")
        default_dnums = list(range(quantity))
        for dnum, device in enumerate(devices):
            print(f"[{dnum}]: {device.name}")
        default_dnums_str = ", ".join(str(dnum) for dnum in default_dnums)
        print(f"Choose the device(s), comma-separated [{default_dnums_str}]: ", end="")
        selected_dnums = default_dnums
        dnums_input = input()
        if dnums_input != "":
            selected_dnums = [int(dnum) for dnum in dnums_input.split(",")]
            if len(selected_dnums) != quantity:
                raise ValueError(f"Exactly {quantity} devices must be selected")

        selected_devices = [devices[dnum] for dnum in selected_dnums]

    return selected_devices


def select_devices(
    api: API,
    interactive: bool = False,
    quantity: Optional[int] = 1,
    device_filter: Optional[DeviceFilter] = None,
    platform_filter: Optional[PlatformFilter] = None,
) -> List["Device"]:
    """
    Using the results from :py:func:`platforms_and_devices_by_mask`, either lets the user
    select the devices (from the ones matching the criteria) interactively,
    or takes the first matching list of ``quantity`` devices.

    :param interactive: if ``True``, shows a dialog to select the devices.
        If ``False``, selects the first matching ones.
    :param quantity: passed to :py:func:`platforms_and_devices_by_mask`.
    :param device_filters: passed to :py:func:`platforms_and_devices_by_mask`.
    """
    suitable_pds = platforms_and_devices_by_mask(
        api, quantity, device_filter=device_filter, platform_filter=platform_filter
    )

    if len(suitable_pds) == 0:
        quantity_val = "any" if quantity is None else quantity
        raise ValueError(
            f"Could not find {quantity_val} devices on a single platform "
            "matching the given criteria"
        )

    if interactive:
        return _select_devices_interactive(suitable_pds, quantity=quantity)

    _, devices = suitable_pds[0]
    return devices if quantity is None else devices[:quantity]
