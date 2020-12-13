Version history
===============


Current development version
---------------------------

* (CHANGED) Arrays don't hold queues any more; they are passed explicitly to ``get()`` or ``set()``.
* (CHANGED) Prepared kernels don't hold queues any more; they are passed on call.
* (CHANGED) ``Queue`` now stands for a single-device queue only; multi-device queues are extracted into ``MultiQueue``.
* (ADDED) ``MultiArray`` to simplify simultaneous kernel execution on multiple devices.
