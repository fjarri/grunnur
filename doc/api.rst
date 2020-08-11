Public API
==========

.. currentmodule:: grunnur


GPGPU API discovery
-------------------

In many applications it would be enough to use dynamic module attributes to get an :py:class:`API` object:

.. code::

    from grunnur import cuda_api
    from grunnur import opencl_api
    from grunnur import any_api

For a finer programmatic control one can use the methods of the :py:class:`API` class:

.. autoclass:: API()
    :members:

.. autoclass:: grunnur.adapter_base.APIID()
    :members:

.. autofunction:: cuda_api_id

.. autofunction:: opencl_api_id

.. autofunction:: all_api_ids


Platforms
---------

A platform is an OpenCL term, but we use it for CUDA API as well for the sake of uniformity.
Naturally, there will always be a single (dummy) platform in CUDA.

.. autoclass:: Platform()
    :members:


Devices
-------

.. autoclass:: Device()
    :members:

.. autoclass:: grunnur.adapter_base.DeviceParameters()
    :members:

.. autoclass:: grunnur.adapter_base.DeviceType()
    :members:


Device discovery
----------------

.. autofunction:: platforms_and_devices_by_mask

.. autofunction:: select_devices


Contexts
--------

.. autoclass:: Context()
    :members:


Queues
------

.. autoclass:: Queue()
    :members:


Buffers and arrays
------------------

.. autoclass:: Buffer()
    :members:

.. autoclass:: Array()
    :members:

.. autoclass:: grunnur.array.SingleDeviceFactory()
    :members:
    :special-members: __getitem__


Programs and kernels
--------------------

.. autoclass:: Program
    :members:
    :special-members: __getattr__

.. autoclass:: grunnur.program.Kernel()
    :members:
    :special-members: __call__

.. autoclass:: MultiDevice


Static kernels
--------------

.. autoclass:: StaticKernel
    :members:
    :special-members: __call__

.. autoclass:: grunnur.vsize.VsizeModules
    :members:


Utilities
---------

.. autoclass:: grunnur.template.Template
    :members:

.. autoclass:: grunnur.template.DefTemplate
    :members:

.. autoclass:: grunnur.template.RenderError
    :members:

.. autoclass:: grunnur.modules.Snippet
    :members:

.. autoclass:: grunnur.modules.Module
    :members:

.. autofunction:: grunnur.modules.render_with_modules



Data type utilities
-------------------

.. module:: grunnur.dtypes

C interop
~~~~~~~~~

.. autofunction:: ctype

.. autofunction:: ctype_struct

.. autofunction:: complex_ctr

.. autofunction:: c_constant

.. autofunction:: align


Struct helpers
~~~~~~~~~~~~~~

.. autofunction:: c_path

.. autofunction:: flatten_dtype

.. autofunction:: extract_field


Data type checks and conversions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: normalize_type

.. autofunction:: is_complex

.. autofunction:: is_double

.. autofunction:: is_integer

.. autofunction:: is_real

.. autofunction:: result_type

.. autofunction:: min_scalar_type

.. autofunction:: detect_type

.. autofunction:: complex_for

.. autofunction:: real_for

.. autofunction:: cast


Function modules
----------------

.. automodule :: grunnur.functions
    :members:


Virtual buffers
---------------

Often one needs temporary buffers that are only used in one place in the code, but used many times.
Allocating them each time they are used may involve too much overhead; allocating real buffers and storing them increases the program's memory requirements.
A possible middle ground is using virtual allocations, where several of them can use the samy physical allocation.
The virtual allocation manager will make sure that two virtual buffers that are used simultaneously (as declared by the user) will not share the same physical space.

.. py:module:: grunnur.virtual_alloc

.. autoclass:: VirtualManager
    :members:

.. autoclass:: TrivialManager

.. autoclass:: ZeroOffsetManager

.. autoclass:: VirtualAllocator

.. autoclass:: VirtualAllocationStatistics
    :members:


.. _cluda-kernel-toolbox:

Kernel toolbox
--------------

There is a set of macros attached to any kernel depending on the API it is being compiled for:

.. c:macro:: GRUNNUR_CUDA_API

    If defined, specifies that the kernel is being compiled for CUDA API.

.. c:macro:: GRUNNUR_OPENCL_API

    If defined, specifies that the kernel is being compiled for CUDA API.

.. c:macro:: GRUNNUR_FAST_MATH

    If defined, specifies that the compilation for this kernel was requested with ``fast_math == True``.

.. c:macro:: LOCAL_BARRIER

    Synchronizes threads inside a block.

.. c:macro:: WITHIN_KERNEL

    Modifier for a device-only function declaration.

.. c:macro:: KERNEL

    Modifier for a kernel function declaration.

.. c:macro:: GLOBAL_MEM

    Modifier for a global memory pointer argument.

.. c:macro:: LOCAL_MEM

    Modifier for a statically allocated local memory variable.

.. c:macro:: LOCAL_MEM_DYNAMIC

    Modifier for a dynamically allocated local memory variable.

.. c:macro:: LOCAL_MEM_ARG

    Modifier for a local memory argument in device-only functions.

.. c:macro:: CONSTANT_MEM

    Modifier for a statically allocated constant memory variable.

.. c:macro:: CONSTANT_MEM_ARG

    Modifier for a constant memory argument in device-only functions.

.. c:macro:: INLINE

    Modifier for inline functions.

.. c:macro:: SIZE_T

    The type of local/global IDs and sizes.
    Equal to ``unsigned int`` for CUDA, and ``size_t`` for OpenCL
    (which can be 32- or 64-bit unsigned integer, depending on the device).

.. FIXME: techincally, it should be unsigned int here, but Sphinx gives warnings for 'unsigned'
.. c:function:: SIZE_T get_local_id(int dim)
.. c:function:: SIZE_T get_group_id(int dim)
.. c:function:: SIZE_T get_global_id(int dim)
.. c:function:: SIZE_T get_local_size(int dim)
.. c:function:: SIZE_T get_num_groups(int dim)
.. c:function:: SIZE_T get_global_size(int dim)

    Local, group and global identifiers and sizes.
    In case of CUDA mimic the behavior of corresponding OpenCL functions.

.. c:macro:: VSIZE_T

    The type of local/global IDs in the virtual grid.
    It is separate from :c:macro:`SIZE_T` because the former is intended to be equivalent to
    what the backend is using, while ``VSIZE_T`` is a separate type and can be made larger
    than ``SIZE_T`` in the future if necessary.

.. c:macro:: ALIGN(int)

    Used to specify an explicit alignment (in bytes) for fields in structures, as

    ::

        typedef struct {
            char ALIGN(4) a;
            int b;
        } MY_STRUCT;
