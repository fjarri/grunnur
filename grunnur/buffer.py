class Buffer:

    @classmethod
    def allocate(cls, context, size):
        backend_buffer = context._backend_context.allocate(size)
        return cls(context, backend_buffer)

    def __init__(self, context, backend_buffer):
        self.context = context
        self.backend_buffer = backend_buffer

    @property
    def kernel_arg(self):
        # Has to be a property since backend_buffer can be externally updated
        # (e.g. if it's a virtual buffer)
        return self.backend_buffer.kernel_arg

    @property
    def offset(self):
        return self.backend_buffer.offset

    @property
    def size(self):
        return self.backend_buffer.size

    def set(self, queue, device_num, host_array, async_=False, dont_sync_other_devices=False):
        self.backend_buffer.set(
            queue.backend_queue, device_num, host_array,
            async_=async_, dont_sync_other_devices=dont_sync_other_devices)

    def get(self, queue, device_num, host_array, async_=False, dont_sync_other_devices=False):
        self.backend_buffer.get(
            queue.backend_queue, device_num, host_array,
            async_=async_, dont_sync_other_devices=dont_sync_other_devices)

    def get_sub_region(self, origin, size):
        return Buffer(self.context, self.backend_buffer.get_sub_region(origin, size))

    def migrate(self, queue, device_num):
        self.backend_buffer.migrate(queue.backend_queue, device_num)
