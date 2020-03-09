from .utils import prod
from .vsize import VirtualSizes


class StaticKernel:
    """
    An object containing a GPU kernel with fixed call sizes.

    .. py:attribute:: source

        Contains the source code of the program.
    """

    def __init__(
            self, context, src, name, global_size,
            local_size=None, device_nums=None, render_globals={}, **kwds):

        self.context = context

        if device_nums is None:
            device_nums = range(len(context.devices))
        else:
            device_nums = sorted(device_nums)

        kernels = []
        vs_metadata = []
        for device_num in device_nums:

            device_params = context.devices[device_num].params

            # Since virtual size function require some registers, they affect the maximum local size.
            # Start from the device's max local size as the first approximation
            # and recompile kernels with smaller local sizes until convergence.

            max_total_local_size = device_params.max_total_local_size

            while True:

                # Try to find kernel launch parameters for the requested local size.
                # May raise OutOfResourcesError if it's not possible,
                # just let it pass to the caller.
                vs = VirtualSizes(
                    max_total_local_size=max_total_local_size,
                    max_local_sizes=device_params.max_local_sizes,
                    max_num_groups=device_params.max_num_groups,
                    local_size_multiple=device_params.warp_size,
                    virtual_global_size=global_size,
                    virtual_local_size=local_size)

                # TODO: check that there are no name clashes with virtual size functions
                new_render_globals = dict(render_globals)
                new_render_globals['static'] = vs.vsize_modules

                # Try to compile the kernel with the corresponding virtual size functions
                program = context._render_and_compile_single_device(
                    device_num, src, render_globals=new_render_globals, **kwds)
                kernel = getattr(program, name)

                if kernel.max_total_local_size >= prod(vs.real_local_size):
                    # Kernel will execute with this local size, use it
                    break

                # By the contract of VirtualSizes,
                # prod(vs.real_local_size) <= max_total_local_size
                # Also, since we're still in this loop,
                # kernel.max_total_local_size < prod(vs.real_local_size).
                # Therefore the new max_total_local_size value is guaranteed
                # to be smaller than the previous one.
                max_total_local_size = kernel.max_total_local_size

                # TODO: prevent this being an endless loop

            kernels.append(kernel)
            vs_metadata.append(vs)

        self.context = context
        self._vs_metadata = vs_metadata
        self._kernels = kernels
        self._device_nums = device_nums

    def __call__(self, queue, *args, device_nums=None):
        """
        Execute the kernel.
        :py:class:`Array` objects are allowed as arguments.
        In case of the OpenCL backend, returns a ``pyopencl.Event`` object.
        """
        # TODO: speed this up. Probably shouldn't create sets on every kernel call.
        if device_nums is None:
            device_nums = queue.device_nums
        else:
            device_nums = [queue.device_nums[device_num] for device_num in device_nums]

        if not set(device_nums).issubset(self._device_nums):
            missing_dev_nums = [
                device_num for device_num in device_nums if device_num not in self._device_nums]
            raise ValueError(
                f"This kernel's program was not compiled for devices {missing_dev_nums.join(', ')}")

        # TODO: support "single-device" kernels to streamline the logic of kernel.__call__()
        # and reduce overheads?
        events = []
        for i, device_num in enumerate(device_nums):
            kernel = self._kernels[i]
            vs = self._vs_metadata[i]
            kernel_args = [arg[i] if isinstance(arg, (list, tuple)) else arg for arg in args]
            event = kernel(queue, vs.real_global_size, vs.real_local_size, *kernel_args)
            events.append(event)
        return events

    def set_constant_array(self, name, arr, queue=None):
        """
        Load a constant array (``arr`` can be either ``numpy`` array or a :py:class:`Array` object)
        corresponding to the symbol ``name`` to device.
        """
        for kernel in self._kernels:
            kernel.set_constant(name, arr, queue=queue)
