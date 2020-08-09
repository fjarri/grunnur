from typing import Callable, Optional, Iterable, Union, Dict, Mapping, Tuple, Sequence

import numpy

from .api import cuda_api_id
from .template import DefTemplate
from .modules import Snippet
from .context import Context
from .queue import Queue
from .array import Array
from .utils import prod, wrap_in_tuple
from .vsize import VirtualSizes
from .program import Program, SingleDeviceProgram, process_arg, MultiDevice, _call_kernels, _set_constant_array


class StaticKernel:
    """
    An object containing a GPU kernel with fixed call sizes.
    """

    context: Context
    """The context this program was compiled for."""

    def __init__(
            self,
            context: Context,
            template_src: Union[str, Callable[..., str], DefTemplate, Snippet],
            name: str,
            global_size: Union[int, Sequence[int]],
            local_size: Union[int, Sequence[int], None]=None,
            device_idxs: Optional[Sequence[int]]=None,
            render_globals: Dict={},
            constant_arrays: Mapping[str, Tuple[int, numpy.dtype]]={},
            **kwds):
        """
        :param context: context to compile the kernel on.
        :param template_src: a string with the source code, or a Mako template source to render.
        :param name: the kernel's name.
        :param global_size: the total number of threads (CUDA)/work items (OpenCL) in each dimension
            (column-major). Note that there may be a maximum size in each dimension as well
            as the maximum number of dimensions. See :py:class:`DeviceParameters` for details.
        :param local_size: the number of threads in a block (CUDA)/work items in a
            work group (OpenCL) in each dimension (column-major).
            If ``None``, it will be chosen automatically.
        :param device_idxs: a list of device numbers to compile on.
            If ``None``, compile on all context's devices.
        :param render_globals: a dictionary of globals to pass to the template.
        :param constant_arrays: (**CUDA only**) a dictionary ``name: (size, dtype)``
            of global constant arrays to be declared in the program.
        """

        if context.api.id != cuda_api_id() and len(constant_arrays) > 0:
            raise ValueError("Compile-time constant arrays are only supported by CUDA API")

        self.context = context

        if device_idxs is None:
            device_idxs = range(len(context.devices))
        else:
            device_idxs = sorted(device_idxs)

        kernel_ls = wrap_in_tuple(local_size) if local_size is not None else local_size
        kernel_gs = wrap_in_tuple(global_size)

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
                    virtual_global_size=kernel_gs,
                    virtual_local_size=kernel_ls)

                # TODO: check that there are no name clashes with virtual size functions
                new_render_globals = dict(render_globals)
                new_render_globals['static'] = vs.vsize_modules

                # Try to compile the kernel with the corresponding virtual size functions
                program = SingleDeviceProgram(
                    context, device_idx, template_src, render_globals=new_render_globals,
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

    def __call__(self, queue: Queue, *args):
        """
        Execute the kernel.
        In case of the OpenCL backend, returns a ``pyopencl.Event`` object.

        :param queue: the multi-device queue to use.
        :param args: kernel arguments. Can be: :py:class:`~grunnur.Array` objects,
            :py:class:`~grunnur.Buffer` objects, ``numpy`` scalars.
        """
        return _call_kernels(
            queue, self._sd_kernel_adapters, self._global_sizes, self._local_sizes, *args,
            device_idxs=self._device_idxs)

    def set_constant_array(self, queue: Queue, name: str, arr: Union[Array, numpy.ndarray]):
        """
        Uploads a constant array to the context's devices (**CUDA only**).

        :param queue: the queue to use for the transfer.
        :param name: the name of the constant array symbol in the code.
        :param arr: either a device or a host array.
        """
        if self.context.api.id != cuda_api_id():
            raise ValueError("Constant arrays are only supported for CUDA API")
        for kernel_adapter in self._sd_kernel_adapters.values():
            _set_constant_array(queue, kernel_adapter.program_adapter, name, arr)
