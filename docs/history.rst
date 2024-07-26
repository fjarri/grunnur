Version history
===============


0.4.0 (Unreleased)
------------------

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
