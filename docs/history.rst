Version history
===============


0.6.0 (2025-09-13)
------------------

Changed
^^^^^^^

* Removed ``ArrayMetadataLike``, use the strict ``ArrayMetadata`` instead. (PR_20_)
* Removed ``None`` variant for ``compiler_options`` and ``constant_arrays``, use an empty list instead. (PR_20_)
* Module discovery is now fully automatic and does not need ``__process_modules__()`` to be defined in user classes. (PR_21_)
* ``Snippet.from_string()`` now takes an iterable of argument names. (PR_23_)
* Dynamic API imports (``cuda_api``, ``opencl_api``, ``any_api``) removed in favor of ``API`` static methods. (PR_25_)


Added
^^^^^

* Export ``DeviceParameters`` and ``BoundDevice``. (PR_20_)
* ``AsArrayMetadata`` ABC. (PR_20_)
* ``ArrayMetadata.with_()`` method. (PR_20_)
* Cache kernels when using PyOpenCL backend. (PR_20_)


.. _PR_20: https://github.com/fjarri/grunnur/pull/20
.. _PR_21: https://github.com/fjarri/grunnur/pull/21
.. _PR_23: https://github.com/fjarri/grunnur/pull/23
.. _PR_25: https://github.com/fjarri/grunnur/pull/25


0.5.0 (31 Jul 2024)
-------------------

Changed
^^^^^^^

* ``local_mem`` keyword parameter of kernel calls renamed to ``cu_dynamic_local_mem``. (PR_17_)
* Renamed ``no_async`` keyword parameter to ``sync``. (PR_18_)


Added
^^^^^

* Made ``ArrayMetadata`` public. (PR_18_)
* ``metadata`` attribute to ``Array``. (PR_18_)
* ``ArrayMetadata.buffer_size``, ``span``, ``min_offset``, ``first_element_offset``, and ``get_sub_region()``; ``Array.minimum_subregion()``. (PR_18_)


.. _PR_17: https://github.com/fjarri/grunnur/pull/17
.. _PR_18: https://github.com/fjarri/grunnur/pull/18



0.4.0 (25 Jul 2024)
-------------------

Changed
^^^^^^^

* Minimal Python version bumped to 3.10, and ``numpy`` to 2.0. (PR_12_)
* Refactored alignment logic to avoid inconsistent behavior and redundant calls. (PR_13_)
* ``Array`` does not support empty shape anymore. (PR_13_)
* ``dtypes.flatten_dtype`` returns ``FieldInfo`` objects. (PR_13_)


Added
^^^^^

* ``Array.empty_like()``. (PR_13_)
* ``DeviceParams.align_words()``. (PR_13_)


Fixed
^^^^^

* ``Array.from_host()`` is now synchronous if the device is given as the first parameter. (PR_13_)
* Mako tracebacks are shown render error. (PR_13_)


.. _PR_12: https://github.com/fjarri/grunnur/pull/12
.. _PR_13: https://github.com/fjarri/grunnur/pull/13


0.3.0 (29 Jan 2023)
-------------------

Changed
^^^^^^^

* ``device_idx`` parameters are gone; now high level functions take ``BoundDevice`` or ``BoundMultiDevice`` arguments to indicate which devices to use; these objects include the corresponding contexts as well, so they don't have to be passed separately.
* Now API adapters only use device indices in a sense of "device index in the platform"; context adapters keep internal objects in dictionaries indexed by these indices, instead of in lists.
* ``py.test`` plugin extracted into a separate package (``pytest-grunnur``).


0.2.0 (10 Mar 2021)
-------------------

Changed
^^^^^^^

* Arrays don't hold queues any more; they are passed explicitly to ``get()`` or ``set()``.
* Prepared kernels don't hold queues any more; they are passed on call.
* ``Queue`` now stands for a single-device queue only; multi-device queues are extracted into ``MultiQueue``.

Added
^^^^^

* ``MultiArray`` to simplify simultaneous kernel execution on multiple devices.


0.1.1 (9 Oct 2020)
------------------

Package build fixed.


0.1.0 (9 Oct 2020)
------------------

Initial version
