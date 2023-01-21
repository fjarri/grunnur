Version history
===============


Current development version
---------------------------

* (CHANGED) ``device_idx`` parameters are gone; now high level functions take ``BoundDevice`` or ``BoundMultiDevice`` arguments to indicate which devices to use; these objects include the corresponding contexts as well, so they don't have to be passed separately.
* Now API adapters only use device indices in a sense of "device index in the platform"; context adapters keep internal objects in dictionaries indexed by these indices, instead of in lists.


0.2.0 (10 Mar 2021)
-------------------

* (CHANGED) Arrays don't hold queues any more; they are passed explicitly to ``get()`` or ``set()``.
* (CHANGED) Prepared kernels don't hold queues any more; they are passed on call.
* (CHANGED) ``Queue`` now stands for a single-device queue only; multi-device queues are extracted into ``MultiQueue``.
* (ADDED) ``MultiArray`` to simplify simultaneous kernel execution on multiple devices.


0.1.1 (9 Oct 2020)
------------------

Package build fixed.



0.1.0 (9 Oct 2020)
------------------

Initial version
