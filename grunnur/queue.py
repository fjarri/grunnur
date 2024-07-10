from __future__ import annotations

import weakref
from abc import ABC
from typing import TYPE_CHECKING

from .adapter_base import QueueAdapter
from .context import BoundDevice, BoundMultiDevice
from .device import Device

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterable, Sequence


class MultiQueue:
    """A queue on multiple devices."""

    devices: BoundMultiDevice
    """Multi-device on which this queue operates."""

    queues: dict[BoundDevice, Queue]
    """Single-device queues associated with device indices."""

    @classmethod
    def on_devices(cls, devices: Iterable[BoundDevice]) -> MultiQueue:
        """Creates a queue from provided devices (belonging to the same context)."""
        return cls([Queue(device) for device in devices])

    def __init__(self, queues: Sequence[Queue]):
        """
        :param queues: single-device queues
            (must belong to distinct devices and the same context).
        """
        self.devices = BoundMultiDevice.from_bound_devices([queue.device for queue in queues])
        self.queues = {queue.device: queue for queue in queues}

    def synchronize(self) -> None:
        """Blocks until queues on all devices are empty."""
        for queue in self.queues.values():
            queue.synchronize()


class Queue:
    """A queue on a single device."""

    device: BoundDevice
    """Device on which this queue operates."""

    def __init__(self, device: BoundDevice):
        """:param device: a device on which to create a queue."""
        self.device = device
        self._queue_adapter = device.context._context_adapter.make_queue_adapter(  # noqa: SLF001
            device._device_adapter  # noqa: SLF001
        )

    def synchronize(self) -> None:
        """Blocks until sub-queues on all devices are empty."""
        self._queue_adapter.synchronize()
