class Buffer:

    @classmethod
    def allocate(cls, context, size):
        buffer_adapter = context._context_adapter.allocate(size)
        return cls(context, buffer_adapter)

    def __init__(self, context, buffer_adapter):
        self.context = context
        self._buffer_adapter = buffer_adapter

    @property
    def kernel_arg(self):
        # Has to be a property since buffer_adapter can be externally updated
        # (e.g. if it's a virtual buffer)
        return self._buffer_adapter.kernel_arg

    @property
    def offset(self):
        return self._buffer_adapter.offset

    @property
    def size(self):
        return self._buffer_adapter.size

    def set(self, queue, device_num, host_array, async_=False, dont_sync_other_devices=False):
        self._buffer_adapter.set(
            queue._queue_adapter, device_num, host_array,
            async_=async_, dont_sync_other_devices=dont_sync_other_devices)

    def get(self, queue, device_num, host_array, async_=False, dont_sync_other_devices=False):
        self._buffer_adapter.get(
            queue._queue_adapter, device_num, host_array,
            async_=async_, dont_sync_other_devices=dont_sync_other_devices)

    def get_sub_region(self, origin, size):
        return Buffer(self.context, self._buffer_adapter.get_sub_region(origin, size))

    def migrate(self, queue, device_num):
        self._buffer_adapter.migrate(queue._queue_adapter, device_num)
