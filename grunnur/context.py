from __future__ import annotations

from typing import Optional

from .utils import wrap_in_tuple, normalize_object_sequence, all_same
from .api import API
from .device import Device
from .device_discovery import select_devices


class Context:
    """
    GPGPU context.
    """

    @classmethod
    def from_devices(cls, devices):
        devices = wrap_in_tuple(devices)
        devices = normalize_object_sequence(devices, Device)

        platforms = [device.platform for device in devices]
        if not all_same(platforms):
            raise ValueError("All devices must belong to the same platform")
        platform = platforms[0]

        device_adapters = [device._device_adapter for device in devices]
        api_adapter = platform.api._api_adapter
        context_adapter = api_adapter.make_context_adapter_from_device_adapters(device_adapters)

        return cls(context_adapter)

    @classmethod
    def from_backend_devices(cls, backend_devices):
        backend_devices = wrap_in_tuple(backend_devices)
        devices = [Device.from_backend_device(backend_device) for backend_device in backend_devices]
        return cls.from_devices(devices)

    @classmethod
    def from_backend_contexts(cls, backend_contexts, take_ownership=False):
        backend_contexts = wrap_in_tuple(backend_contexts)
        for api in API.all():
            if api._api_adapter.isa_backend_context(backend_contexts[0]):
                context_adapter = api._api_adapter.make_context_adapter_from_backend_contexts(
                    backend_contexts, take_ownership=take_ownership)
                return cls(context_adapter)
        raise TypeError(
            f"{type(backend_contexts[0])} objects were not recognized as contexts by any API")

    @classmethod
    def from_criteria(
            cls, api, interactive: bool=False, devices_num: Optional[int]=1, **device_filters) -> Context:
        """
        Finds devices matching the given criteria and creates a
        :py:class:`Context` object out of them.

        :param interactive: passed to :py:meth:`select_devices`.
        :param devices_num: passed to :py:meth:`select_devices` as ``quantity``.
        :param device_filters: passed to :py:meth:`select_devices`.
        """
        devices = select_devices(api, interactive=interactive, quantity=devices_num, **device_filters)
        return cls.from_devices(devices)

    def __init__(self, context_adapter):
        self._context_adapter = context_adapter
        self.devices = [
            Device(device_adapter) for device_adapter in context_adapter.device_adapters]
        self.platform = self.devices[0].platform
        self.api = self.platform.api

    def deactivate(self):
        self._context_adapter.deactivate()
