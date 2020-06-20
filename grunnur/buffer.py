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

    def set(self, queue, host_array, async_=False, device_idx=None):
        device_idx = queue.device_idxs[0] if device_idx is None else device_idx
        self._buffer_adapter.set(queue._queue_adapter, device_idx, host_array, async_=async_)

    def get(self, queue, host_array, async_=False, device_idx=None):
        device_idx = queue.device_idxs[0] if device_idx is None else device_idx
        self._buffer_adapter.get(queue._queue_adapter, device_idx, host_array, async_=async_)

    def get_sub_region(self, origin, size):
        return Buffer(self.context, self._buffer_adapter.get_sub_region(origin, size))

    def migrate(self, queue, device_idx):
        self._buffer_adapter.migrate(queue._queue_adapter, device_idx)
