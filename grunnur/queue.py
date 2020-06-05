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

        queue_adapter = context._context_adapter.make_queue_adapter(device_nums)
        return cls(context, queue_adapter, device_nums)

    def __init__(self, context, queue_adapter, device_nums):
        self.context = context
        self._queue_adapter = queue_adapter
        self.devices = {
            device_num: Device(queue_adapter.device_adapters[device_num])
            for device_num in device_nums}

    def synchronize(self):
        """
        Blocks until sub-queues on all devices are empty.
        """
        self._queue_adapter.synchronize()
