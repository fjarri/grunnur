from __future__ import annotations

from typing import Optional, Union, Iterable, Tuple

from .utils import wrap_in_tuple, normalize_object_sequence, all_same
from .api import API
from .device import Device
from .device_discovery import select_devices
from .platform import Platform


class Context:
    """
    GPGPU context.
    """

    devices: Tuple[Device]
    """Devices in this context."""

    platform: Platform
    """The platform this context is based on."""

    api: API
    """The API this context is based on."""

    @classmethod
    def from_devices(cls, devices: Union[Device, Iterable[Device]]) -> Context:
        """
        Creates a context from a device or an iterable of devices.

        :param devices: one or several devices to use.
        """
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
    def from_backend_devices(cls, backend_devices) -> Context:
        """
        Creates a context from a single or several backend device objects.
        """
        backend_devices = wrap_in_tuple(backend_devices)
        devices = [Device.from_backend_device(backend_device) for backend_device in backend_devices]
        return cls.from_devices(devices)

    @classmethod
    def from_backend_contexts(cls, backend_contexts, take_ownership: bool=False) -> Context:
        """
        Creates a context from a single or several backend device contexts.
        If ``take_ownership`` is ``True``, this object will be responsible for the lifetime
        of backend context objects (important for CUDA backend).
        """
        backend_contexts = wrap_in_tuple(backend_contexts)
        for api in API.all_available():
            if api._api_adapter.isa_backend_context(backend_contexts[0]):
                context_adapter = api._api_adapter.make_context_adapter_from_backend_contexts(
                    backend_contexts, take_ownership=take_ownership)
                return cls(context_adapter)
        raise TypeError(
            f"{type(backend_contexts[0])} objects were not recognized as contexts by any API")

    @classmethod
    def from_criteria(
            cls, api: API, interactive: bool=False,
            devices_num: Optional[int]=1, **device_filters) -> Context:
        """
        Finds devices matching the given criteria and creates a
        :py:class:`Context` object out of them.

        :param interactive: passed to :py:func:`select_devices`.
        :param devices_num: passed to :py:func:`select_devices` as ``quantity``.
        :param device_filters: passed to :py:func:`select_devices`.
        """
        devices = select_devices(api, interactive=interactive, quantity=devices_num, **device_filters)
        return cls.from_devices(devices)

    def __init__(self, context_adapter):
        self._context_adapter = context_adapter
        self.devices = tuple(
            Device(device_adapter) for device_adapter in context_adapter.device_adapters)
        self.platform = self.devices[0].platform
        self.api = self.platform.api

    def deactivate(self):
        """
        CUDA API only: deactivates this context, popping all the CUDA context objects from the stack.
        """
        self._context_adapter.deactivate()
