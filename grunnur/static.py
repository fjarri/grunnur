from typing import Callable, Optional, Union, Dict, Mapping, Tuple, Sequence

import numpy

from .api import cuda_api_id
from .template import DefTemplate
from .modules import Snippet
from .context import Context
from .queue import Queue
from .array import Array
from .utils import prod, wrap_in_tuple, update_dict
from .vsize import VirtualSizes, VirtualSizeError
from .program import SingleDeviceProgram, MultiDevice, PreparedKernel, _set_constant_array


# the name of the global in the template containing static kernel modules
_STATIC_MODULES_GLOBAL = 'static'


class StaticKernel:
    """
    An object containing a GPU kernel with fixed call sizes.

    The globals for the source template will contain an object with the name ``static``
    of the type :py:class:`~grunnur.vsize.VsizeModules` containing the id/size functions
    to be used instead of regular ones.
    """

    queue: Queue
    """The queue this static kernel was compiled and prepared for."""

    sources: Dict[int, str]
    """Source files used for each device."""

    def __init__(
            self,
            queue: Queue,
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
        context = queue.context

        if context.api.id != cuda_api_id() and len(constant_arrays) > 0:
            raise ValueError("Compile-time constant arrays are only supported by CUDA API")

        if device_idxs is None:
            device_idxs = range(len(context.devices))
        else:
            device_idxs = sorted(device_idxs)

        kernel_ls = wrap_in_tuple(local_size) if local_size is not None else local_size
        kernel_gs = wrap_in_tuple(global_size)

        kernel_adapters = {}
        sources = {}
        vs_metadata = []
        for device_idx in device_idxs:

            device_params = context.devices[device_idx].params

            # Since virtual size function require some registers,
            # they affect the maximum local size.
            # Start from the device's max local size as the first approximation
            # and recompile kernels with smaller local sizes until convergence.

            max_total_local_size = device_params.max_total_local_size

            while True:

                # Try to find kernel launch parameters for the requested local size.
                # May raise VirtualSizeError if it's not possible,
                # just let it pass to the caller.
                vs = VirtualSizes(
                    max_total_local_size=max_total_local_size,
                    max_local_sizes=device_params.max_local_sizes,
                    max_num_groups=device_params.max_num_groups,
                    local_size_multiple=device_params.warp_size,
                    virtual_global_size=kernel_gs,
                    virtual_local_size=kernel_ls)

                new_render_globals = update_dict(
                    render_globals, {_STATIC_MODULES_GLOBAL: vs.vsize_modules},
                    error_msg=f"The global name '{_STATIC_MODULES_GLOBAL}' is reserved in static kernels")

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

                # In most cases the iteration should stop at `max_total_local_size == 1`,
                # where the virtual size is trivial and always possible.
                # But occasionally we may get a kernel that cannot be executed at all
                # (e.g. it requests too much local memory),
                # and some platforms may return 0, which `VirtualSizes` will not like.
                # So we'll have a sanity check here.
                if max_total_local_size == 0:
                    raise VirtualSizeError(
                        "The kernel requires too much resourses to be executed with any local size")

            kernel_adapters[device_idx] = kernel_adapter
            sources[device_idx] = program.source
            vs_metadata.append(vs)

        self.queue = queue
        self.sources = sources
        self._vs_metadata = vs_metadata
        self._sd_kernel_adapters = kernel_adapters
        self._device_idxs = device_idxs

        global_sizes = MultiDevice(*[vs.real_global_size for vs in self._vs_metadata])
        local_sizes = MultiDevice(*[vs.real_local_size for vs in self._vs_metadata])

        self._prepared_kernel = PreparedKernel(
            kernel_adapters, queue, device_idxs, global_sizes, local_sizes)

    def __call__(self, *args):
        """
        Execute the kernel.
        In case of the OpenCL backend, returns a ``pyopencl.Event`` object.

        :param queue: the multi-device queue to use.
        :param args: kernel arguments. Can be: :py:class:`~grunnur.Array` objects,
            :py:class:`~grunnur.Buffer` objects, ``numpy`` scalars.
        """
        return self._prepared_kernel(*args)

    def set_constant_array(self, queue: Queue, name: str, arr: Union[Array, numpy.ndarray]):
        """
        Uploads a constant array to the context's devices (**CUDA only**).

        :param queue: the queue to use for the transfer.
        :param name: the name of the constant array symbol in the code.
        :param arr: either a device or a host array.
        """
        if self.queue.context.api.id != cuda_api_id():
            raise ValueError("Constant arrays are only supported for CUDA API")
        for kernel_adapter in self._sd_kernel_adapters.values():
            _set_constant_array(queue, kernel_adapter._program_adapter, name, arr)
