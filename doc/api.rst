Public API
==========


GPGPU API discovery
-------------------

.. autofunction:: grunnur.available_apis

.. autofunction:: grunnur.find_apis



Base classes
------------

.. autoclass:: grunnur.base_classes.APIID
    :members:

.. autoclass:: grunnur.base_classes.PlatformID
    :members:

.. autoclass:: grunnur.base_classes.DeviceID
    :members:

.. autoclass:: grunnur.base_classes.API
    :members:

.. autoclass:: grunnur.base_classes.Platform
    :members:

.. autoclass:: grunnur.base_classes.Device
    :members:

.. autoclass:: grunnur.base_classes.DeviceParameters
    :members:

.. autoclass:: grunnur.base_classes.DeviceType
    :members:

.. autoclass:: grunnur.base_classes.Context
    :members:

.. autoclass:: grunnur.base_classes.Queue
    :members:

.. autoclass:: grunnur.base_classes.Program
    :members:
    :special-members: __getitem__

.. autoclass:: grunnur.base_classes.SingleDeviceProgram
    :members:
    :special-members: __getitem__

.. autoclass:: grunnur.base_classes.Kernel
    :members:
    :special-members: __call__

.. autoclass:: grunnur.base_classes.SingleDeviceKernel
    :members:
    :special-members: __call__

.. autoclass:: grunnur.base_classes.Buffer
    :members:

.. autoclass:: grunnur.base_classes.Array
    :members:

.. autoclass:: grunnur.base_classes.SingleDeviceFactory
    :members:
    :special-members: __getitem__


Platform-specific API
---------------------

CUDA
~~~~

.. autoclass:: grunnur.cuda.CuDevice
    :show-inheritance:
    :members: from_pycuda_device

.. autoclass:: grunnur.cuda.CuContext
    :show-inheritance:
    :members: from_any_base, from_pycuda_devices, from_pycuda_contexts, from_devices, activate_device, deactivate

.. autoclass:: grunnur.cuda.CuProgram
    :show-inheritance:
    :members: set_constant_array

.. autoclass:: grunnur.cuda.CuSingleDeviceProgram
    :show-inheritance:
    :members: set_constant_array

OpenCL
~~~~~~

.. autoclass:: grunnur.opencl.OclPlatform
    :show-inheritance:
    :members: from_pyopencl_platform

.. autoclass:: grunnur.opencl.OclDevice
    :show-inheritance:
    :members: from_pyopencl_device

.. autoclass:: grunnur.opencl.OclContext
    :show-inheritance:
    :members: from_any_base, from_pyopencl_devices, from_pyopencl_context, from_devices

.. autoclass:: grunnur.opencl.OclQueue
    :show-inheritance:
    :members: from_pyopencl_commandqueues


Utilities
---------

.. autoclass:: grunnur.template.Template
    :members:

.. autoclass:: grunnur.template.DefTemplate
    :members:

.. autoclass:: grunnur.modules.Snippet
    :members:

.. autoclass:: grunnur.modules.Module
    :members:
