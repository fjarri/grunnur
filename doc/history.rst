Version history
===============


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
