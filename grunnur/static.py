from .api import cuda_api_id
from .utils import prod
from .vsize import VirtualSizes
from .program import Program, SingleDeviceProgram, process_arg, MultiDevice, _call_kernels, _set_constant_array


class StaticKernel:
    """
    An object containing a GPU kernel with fixed call sizes.

    .. py:attribute:: source

        Contains the source code of the program.
    """

    def __init__(
            self, context, src, name, global_size,
            local_size=None, device_idxs=None, render_globals={}, constant_arrays={}, **kwds):

        if context.api.id != cuda_api_id() and len(constant_arrays) > 0:
            raise ValueError("Compile-time constant arrays are only supported by CUDA API")

        self.context = context

        if device_idxs is None:
            device_idxs = range(len(context.devices))
        else:
            device_idxs = sorted(device_idxs)

        kernel_adapters = {}
        vs_metadata = []
        for device_idx in device_idxs:

            device_params = context.devices[device_idx].params

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
                program = SingleDeviceProgram(
                    context, device_idx, src, render_globals=new_render_globals,
                    constant_arrays=constant_arrays, **kwds)
                kernel_adapter = program.get_kernel_adapter(name)

                if kernel_adapter.max_total_local_size >= prod(vs.real_local_size):
                    # Kernel will execute with this local size, use it
                    break

                # By the contract of VirtualSizes,
                # prod(vs.real_local_size) <= max_total_local_size
                # Also, since we're still in this loop,
                # kernel_adapter.max_total_local_size < prod(vs.real_local_size).
                # Therefore the new max_total_local_size value is guaranteed
                # to be smaller than the previous one.
                max_total_local_size = kernel_adapter.max_total_local_size

                # TODO: prevent this being an endless loop

            kernel_adapters[device_idx] = kernel_adapter
            vs_metadata.append(vs)

        self.context = context
        self._vs_metadata = vs_metadata
        self._sd_kernel_adapters = kernel_adapters
        self._device_idxs = device_idxs

        self._global_sizes = MultiDevice(*[vs.real_global_size for vs in self._vs_metadata])
        self._local_sizes = MultiDevice(*[vs.real_local_size for vs in self._vs_metadata])

    def __call__(self, queue, *args):
        """
        Execute the kernel.
        :py:class:`Array` objects are allowed as arguments.
        In case of the OpenCL backend, returns a ``pyopencl.Event`` object.
        """
        return _call_kernels(
            queue, self._sd_kernel_adapters, self._global_sizes, self._local_sizes, *args,
            device_idxs=self._device_idxs)

    def set_constant_array(self, queue, name, arr):
        """
        Load a constant array (``arr`` can be either ``numpy`` array or a :py:class:`Array` object)
        corresponding to the symbol ``name`` to device.
        """
        if self.context.api.id != cuda_api_id():
            raise ValueError("Constant arrays are only supported for CUDA API")
        for kernel_adapter in self._sd_kernel_adapters.values():
            _set_constant_array(queue, kernel_adapter.program_adapter, name, arr)
