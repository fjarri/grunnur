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

        backend_queue = context._backend_context.make_multi_queue(device_nums)
        return cls(context, backend_queue, device_nums)

    def __init__(self, context, backend_queue, device_nums):
        self.context = context
        self.backend_queue = backend_queue
        self.device_nums = device_nums
        self.devices = [
            Device.from_backend_device(backend_device) for backend_device in backend_queue.devices]

    def synchronize(self):
        """
        Blocks until sub-queues on all devices are empty.
        """
        self.backend_queue.synchronize()
