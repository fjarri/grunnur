from typing import Optional, Iterable, Tuple

from .adapter_base import QueueAdapter
from .context import Context
from .device import Device


class Queue:
    """
    A queue on multiple devices.
    """

    context: Context
    """This queue's context."""

    device_idxs: Tuple[int, ...]
    """Device indices (in the context) this queue operates on."""

    @classmethod
    def on_all_devices(cls, context: Context) -> 'Queue':
        return cls.on_device_idxs(context, list(range(len(context.devices))))

    @classmethod
    def on_device_idxs(cls, context: Context, device_idxs: Iterable[int]) -> 'Queue':
        """
        Creates a queue from provided device indexes (in the context).

        :param context: the context to create a queue in.
        :param device_idxs: the indices of devices (in the context) to use.
        """
        device_idxs = tuple(sorted(device_idxs))
        queue_adapter = context._context_adapter.make_queue_adapter(device_idxs)
        return cls(context, queue_adapter, device_idxs)

    def __init__(self, context: Context, queue_adapter: QueueAdapter, device_idxs: Tuple[int, ...]):
        self.context = context
        self._queue_adapter = queue_adapter
        self.default_device_idx = device_idxs[0]
        self.device_idxs = device_idxs # Preserving the order of given device indices
        self.devices = {
            device_idx: Device(queue_adapter.device_adapters[device_idx])
            for device_idx in device_idxs}

    def synchronize(self):
        """
        Blocks until sub-queues on all devices are empty.
        """
        self._queue_adapter.synchronize()
