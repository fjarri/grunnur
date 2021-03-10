from __future__ import annotations

from abc import ABC
from typing import Optional, Iterable, Tuple, Set, Dict
import weakref

from .adapter_base import QueueAdapter
from .context import Context
from .device import Device


class MultiQueue:
    """
    A queue on multiple devices.
    """

    context: 'Context'
    """This queue's context."""

    device_idxs: Set[int]
    """Device indices (in the context) this queue operates on."""

    queues: Dict[int, 'Queue']
    """Single-device queues associated with device indices."""

    devices: Dict[int, 'Device']
    """Device objects associated with device indices."""

    @classmethod
    def on_device_idxs(cls, context: 'Context', device_idxs: Iterable[int]) -> 'MultiQueue':
        """
        Creates a queue from provided device indexes (in the context).
        """
        return cls(context, [Queue(context, device_idx) for device_idx in device_idxs])

    def __init__(self, context: 'Context', queues: Optional[Iterable['Queue']]=None):
        """
        :param context: a context on which to create a queue.
        :param queues: single-device queues (must belong to distinct devices).
        """

        if queues is None:
            queues = [Queue(context, device_idx) for device_idx in range(len(context.devices))]
        else:
            queues = list(queues)
            device_idxs = [queue.device_idx for queue in queues]
            if len(set(device_idxs)) != len(device_idxs):
                raise ValueError("Queues in a MultiQueue must belong to distinct devices")

        self.context = context
        self.queues = {queue.device_idx: queue for queue in queues}

        # Preserving the order of given device indices
        self.device_idxs = {queue.device_idx for queue in queues}

        self.devices = {device_idx: context.devices[device_idx] for device_idx in self.device_idxs}

    def synchronize(self):
        """
        Blocks until queues on all devices are empty.
        """
        for queue in self.queues.values():
            queue.synchronize()


class Queue:
    """
    A queue on a single device.
    """

    context: 'Context'
    """This queue's context."""

    device_idx: int
    """Device index this queue operates on."""

    device: 'Device'
    """Device object this queue operates on."""

    def __init__(self, context: 'Context', device_idx: Optional[int]=None):
        """
        :param context: a context on which to create a queue.
        :param device_idx: device index in the context on which to create a queue.
            If there is more than one device in the context, it must be specified.
        """

        if device_idx is None:
            if len(context.devices) > 1:
                raise ValueError("For multi-device contexts, device_idx must be set explicitly")
            else:
                device_idx = 0

        self.context = context
        self._queue_adapter = context._context_adapter.make_queue_adapter(device_idx)
        self.device_idx = device_idx
        self.device = context.devices[device_idx]

    def synchronize(self):
        """
        Blocks until sub-queues on all devices are empty.
        """
        self._queue_adapter.synchronize()
