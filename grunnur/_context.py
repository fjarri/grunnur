from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from typing import TYPE_CHECKING, Any, overload

from ._api import API, cuda_api_id
from ._device import Device, DeviceFilter
from ._device_discovery import select_devices
from ._platform import Platform, PlatformFilter
from ._utils import all_same, normalize_object_sequence

if TYPE_CHECKING:  # pragma: no cover
    from ._adapter_base import ContextAdapter, DeviceAdapter


class BoundDevice(Device):
    """A :py:class:`~grunnur.Device` object in a :py:class:`~grunnur.Context`."""

    context: Context
    """The context this device belongs to."""

    def __init__(self, context: Context, device_adapter: DeviceAdapter):
        super().__init__(device_adapter)
        self.context = context

        # A proper hashing would require `Context` to be hashable too,
        # but `BoundDevice` objects are only ever used in small collections
        # and with all the device indices being different.
        # If somehow there's a hash collision, it will be taken care of by ``__eq__``.
        self._hash = hash(device_adapter)

    def as_unbound(self) -> Device:
        """
        :meta private:
        Returns the unbound :py:class:`Device` object.
        """
        return Device(self._device_adapter)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, BoundDevice)
            and self.context == other.context
            and super().__eq__(other)
        )

    def __hash__(self) -> int:
        return self._hash

    def __str__(self) -> str:
        return super().__str__() + " in " + str(self.context)


class BoundMultiDevice(Sequence[BoundDevice]):
    """A sequence of bound devices belonging to the same context."""

    context: Context
    """The context these devices belong to."""

    @classmethod
    def from_bound_devices(cls, devices: Sequence[BoundDevice]) -> BoundMultiDevice:
        """
        Creates this object from a sequence of bound devices
        (note that a ``BoundMultiDevice`` object itself can serve as such a sequence).
        """
        if not all_same(device.context for device in devices):
            raise ValueError("All devices in a multi-device must belong to the same context")

        if len(set(devices)) != len(devices):
            raise ValueError("All devices in a multi-device must be distinct")

        return cls(devices[0].context, [device._device_adapter for device in devices])  # noqa: SLF001

    def __init__(self, context: Context, device_adapters: Sequence[DeviceAdapter]):
        self.context = context
        self._devices = [BoundDevice(context, device_adapter) for device_adapter in device_adapters]
        self._devices_as_set = set(self._devices)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, BoundMultiDevice)
            and self.context == other.context
            and self._devices == other._devices
        )

    def issubset(self, devices: BoundMultiDevice) -> bool:
        return self._devices_as_set.issubset(devices._devices_as_set)

    def __iter__(self) -> Iterator[BoundDevice]:
        return iter(self._devices)

    @overload
    def __getitem__(self, idx: int) -> BoundDevice: ...

    @overload
    def __getitem__(self, idx: slice | Iterable[int]) -> BoundMultiDevice: ...

    def __getitem__(self, idx: int | slice | Iterable[int]) -> BoundDevice | BoundMultiDevice:
        """
        Given a single index, returns a single :py:class:`BoundDevice`.
        Given a sequence of indices, returns a :py:class:`BoundMultiDevice` object
        containing respective devices.

        The indices correspond to the list of devices used to create this context.
        """
        if isinstance(idx, Iterable):
            return BoundMultiDevice.from_bound_devices([self._devices[i] for i in idx])
        if isinstance(idx, slice):
            return BoundMultiDevice.from_bound_devices(self._devices[idx])
        return self._devices[idx]

    def __len__(self) -> int:
        return len(self._devices)


class Context:
    """GPGPU context."""

    platform: Platform
    """The platform this context is based on."""

    api: API
    """The API this context is based on."""

    @classmethod
    def from_devices(cls, devices: Sequence[Device]) -> Context:
        """
        Creates a context from a device or an iterable of devices.

        :param devices: one or several devices to use.
        """
        devices = normalize_object_sequence(devices, Device)

        platforms = [device.platform for device in devices]
        if not all_same(platforms):
            raise ValueError("All devices must belong to the same platform")
        platform = platforms[0]

        device_adapters = [device._device_adapter for device in devices]  # noqa: SLF001
        api_adapter = platform.api._api_adapter  # noqa: SLF001
        context_adapter = api_adapter.make_context_adapter_from_device_adapters(device_adapters)

        return cls(context_adapter)

    @classmethod
    def from_backend_devices(cls, backend_devices: Sequence[Any]) -> Context:
        """Creates a context from a single or several backend device objects."""
        devices = [Device.from_backend_device(backend_device) for backend_device in backend_devices]
        return cls.from_devices(devices)

    @classmethod
    def from_backend_contexts(
        cls, backend_contexts: Sequence[Any], *, take_ownership: bool = False
    ) -> Context:
        """
        Creates a context from a single or several backend device contexts.
        If ``take_ownership`` is ``True``, this object will be responsible for the lifetime
        of backend context objects (only important for the CUDA backend).
        """
        for api in API.all_available():
            if api._api_adapter.isa_backend_context(backend_contexts[0]):  # noqa: SLF001
                context_adapter = api._api_adapter.make_context_adapter_from_backend_contexts(  # noqa: SLF001
                    backend_contexts, take_ownership=take_ownership
                )
                return cls(context_adapter)
        raise TypeError(
            f"{type(backend_contexts[0])} objects were not recognized as contexts by any API"
        )

    @classmethod
    def from_criteria(
        cls,
        api: API,
        *,
        interactive: bool = False,
        devices_num: int | None = 1,
        device_filter: DeviceFilter | None = None,
        platform_filter: PlatformFilter | None = None,
    ) -> Context:
        """
        Finds devices matching the given criteria and creates a
        :py:class:`Context` object out of them.

        :param interactive: passed to :py:func:`select_devices`.
        :param devices_num: passed to :py:func:`select_devices` as ``quantity``.
        :param device_filters: passed to :py:func:`select_devices`.
        """
        devices = select_devices(
            api,
            interactive=interactive,
            quantity=devices_num,
            device_filter=device_filter,
            platform_filter=platform_filter,
        )
        return cls.from_devices(devices)

    def __init__(self, context_adapter: ContextAdapter):
        self._context_adapter = context_adapter
        self._device_adapters = context_adapter.device_adapters
        self.platform = Platform(next(iter(self._device_adapters.values())).platform_adapter)
        self.api = self.platform.api

    @property
    def devices(self) -> BoundMultiDevice:
        """
        Returns the :py:class:`~grunnur._context.BoundMultiDevice`
        encompassing all the devices in this context.
        """
        # Need to create it on-demand to avoid a circular reference.
        device_adapters = [
            self._device_adapters[device_idx] for device_idx in self._context_adapter.device_order
        ]
        return BoundMultiDevice(self, device_adapters)

    @property
    def device(self) -> BoundDevice:
        if len(self._device_adapters) > 1:
            raise RuntimeError("The `device` shortcut only works for single-device contexts")

        return self.devices[0]

    def deactivate(self) -> None:
        """
        For CUDA API: deactivates this context, popping all the CUDA context objects from the stack.
        Other APIs: no effect.

        Only call it if you need to manage CUDA contexts manually,
        and created this object with `take_ownership = False`.
        If `take_ownership = True` contexts will be deactivated automatically in the destructor.
        """
        self._context_adapter.deactivate()
