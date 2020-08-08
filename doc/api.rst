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
