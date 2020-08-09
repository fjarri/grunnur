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

.. autoclass:: StaticKernel
    :members:
    :special-members: __call__

.. autoclass:: MultiDevice
