from __future__ import annotations

from typing import Optional

from .utils import wrap_in_tuple, normalize_object_sequence
from .api import API
from .device import Device, select_devices


class Context:
    """
    GPGPU context.
    """

    @classmethod
    def from_devices(cls, devices):
        devices = wrap_in_tuple(devices)
        platform = devices[0].platform

        device_adapters = [device._device_adapter for device in devices]
        context_adapter = platform._platform_adapter.make_context(device_adapters)

        return cls(context_adapter)

    @classmethod
    def from_backend_devices(cls, backend_devices):
        #devices = normalize_object_sequence(devices)
        for api in API.all():
            if api._api_adapter.isa_backend_device(backend_devices[0]):
                context_adapter = api._api_adapter.make_context_from_backend_devices(backend_devices)
                return cls(context_adapter)
        raise TypeError(
            f"{type(backend_devices[0])} objects were not recognized as devices by any API")

    @classmethod
    def from_backend_contexts(cls, backend_contexts):
        #contexts = normalize_object_sequence(contexts)
        for api in API.all():
            if api._api_adapter.isa_backend_context(backend_contexts[0]):
                context_adapter = api._api_adapter.make_context_from_backend_contexts(backend_contexts)
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
            Device.from_device_adapter(device_adapter)
            for device_adapter in context_adapter.device_adapters]
        self.platform = self.devices[0].platform
        self.api = self.platform.api
