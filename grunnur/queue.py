from .device import Device


class Queue:
    """
    A queue on multiple devices.
    """

    @classmethod
    def from_device_nums(cls, context, device_nums=None):
        # TODO: need a better method name for the case of device_nums=None

        if device_nums is None:
            device_nums = tuple(range(len(context.devices)))
        else:
            device_nums = tuple(sorted(device_nums))

        queue_adapter = context._context_adapter.make_multi_queue(device_nums)
        return cls(context, queue_adapter, device_nums)

    def __init__(self, context, queue_adapter, device_nums):
        self.context = context
        self._queue_adapter = queue_adapter
        self.device_nums = device_nums
        self.devices = [
            Device.from_backend_device(device_adapter) for device_adapter in queue_adapter.devices]

    def synchronize(self):
        """
        Blocks until sub-queues on all devices are empty.
        """
        self._queue_adapter.synchronize()
