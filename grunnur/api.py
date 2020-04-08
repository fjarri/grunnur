from __future__ import annotations

from typing import Optional, Iterable, List

from .adapter_cuda import CuAPIAdapterFactory
from .adapter_opencl import OclAPIAdapterFactory


CUDA_API_ID = CuAPIAdapterFactory().api_id


OPENCL_API_ID = OclAPIAdapterFactory().api_id


_ALL_API_ADAPTER_FACTORIES = {
    factory.api_id: factory for factory in [
        CuAPIAdapterFactory(),
        OclAPIAdapterFactory(),
        ]
    }


def all_api_ids() -> List[APIID]:
    return list(_ALL_API_ADAPTER_FACTORIES.keys())


class API:
    """
    A generalized GPGPU API.

    .. py:attribute:: id

        This API's ID, an :py:class:`APIID` object.

    .. py:attribute:: short_name

        This API's short name.

    .. py:attribute:: shortcut

        A shortcut for this API (to use in :py:func:`~grunnur.find_apis`).
    """

    @classmethod
    def all(cls) -> List[API]:
        """
        Returns a list of :py:class:`~grunnur.base_classes.API` objects
        for which backends are available.
        """
        # TODO: rename to `all_available()`?
        return [
            cls.from_api_id(api_id)
            for api_id, api_factory in _ALL_API_ADAPTER_FACTORIES.items()
            if api_factory.available]

    @classmethod
    def all_by_shortcut(cls, shortcut: Optional[str]=None) -> List[API]:
        """
        If ``shortcut`` is a string, returns a list of one :py:class:`~grunnur.base_classes.API` object
        whose :py:attr:`~grunnur.base_classes.API.id` attribute has its
        :py:attr:`~grunnur.base_classes.APIID.shortcut` attribute equal to it
        (or raises an error if it was not found, or its backend is not available).

        If ``shortcut`` is ``None``, returns a list of all available
        py:class:`~grunnur.base_classes.API` objects.

        :param shortcut: an API shortcut to match.
        """
        if shortcut is None:
            apis = cls.all()
        else:
            for api_id, api_factory in _ALL_API_ADAPTER_FACTORIES.items():
                if shortcut == api_id.shortcut:
                    if not api_factory.available:
                        raise ValueError(str(shortcut) + " API is not available")
                    apis = [cls.from_api_id(api_id)]
                    break
            else:
                raise ValueError("Invalid API shortcut: " + str(shortcut))

        return apis

    @classmethod
    def from_api_id(cls, api_id):
        api_adapter = _ALL_API_ADAPTER_FACTORIES[api_id].make_api_adapter()
        return cls(api_adapter)

    def __init__(self, api_adapter):
        self.id = api_adapter.id
        self._api_adapter = api_adapter
        self.shortcut = self.id.shortcut
        self.short_name = f"api({self.shortcut})"

    def find_devices(
            self,
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

        suitable_platforms = self.find_platforms(
            include_masks=platform_include_masks, exclude_masks=platform_exclude_masks)

        for platform in suitable_platforms:

            suitable_devices = platform.find_devices(
                include_masks=device_include_masks, exclude_masks=device_exclude_masks,
                unique_devices_only=unique_devices_only,
                include_pure_parallel_devices=include_pure_parallel_devices)

            if ((quantity is None and len(suitable_devices) > 0) or
                    (quantity is not None and len(suitable_devices) >= quantity)):
                results.append((platform, suitable_devices))

        return results

    def select_devices(
            self, interactive: bool=False, quantity: Optional[int]=1,
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
        suitable_pds = self.find_devices(quantity, **device_filters)

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
