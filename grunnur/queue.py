from .device import Device


class Queue:
    """
    A queue on multiple devices.
    """

    @classmethod
    def from_device_idxs(cls, context, device_idxs=None):
        # TODO: need a better method name for the case of device_idxs=None

        if device_idxs is None:
            device_idxs = tuple(range(len(context.devices)))
        else:
            device_idxs = tuple(sorted(device_idxs))

        queue_adapter = context._context_adapter.make_queue_adapter(device_idxs)
        return cls(context, queue_adapter, device_idxs)

    def __init__(self, context, queue_adapter, device_idxs):
        self.context = context
        self._queue_adapter = queue_adapter
        self.devices = {
            device_idx: Device(queue_adapter.device_adapters[device_idx])
            for device_idx in device_idxs}

    def synchronize(self):
        """
        Blocks until sub-queues on all devices are empty.
        """
        self._queue_adapter.synchronize()
