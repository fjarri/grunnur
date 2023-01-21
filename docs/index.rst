.. grunnur documentation master file, created by
   sphinx-quickstart on Tue Jan 28 16:50:56 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Grunnur, a generalized API for CUDA and OpenCL
==============================================

Grunnur is a thin layer on top of `PyCUDA <http://documen.tician.de/pycuda>`_ and `PyOpenCL <http://documen.tician.de/pyopencl>`_ that makes it easier to write platform-agnostic programs.
It is a reworked ``cluda`` submodule of `Reikna <http://reikna.publicfields.net>`_, extracted into a separate module.


Main features
-------------

* For the majority of cases, allows one to write platform-independent code.
* Simple usage of multiple GPUs (in particular, no need to worry about context switching for CUDA).
* A way to split kernel code into modules with dependencies between them (see :py:class:`~grunnur.Module` and :py:class:`~grunnur.Snippet`).
* Various mathematical functions (with complex numbers support) organized as modules.
* Static kernels, where you can use global/local shapes with any kinds of dimensions without worrying about assembling array indices from ``blockIdx`` and ``gridIdx``.
* A temporary buffer manager that can pack several virtual buffers into the same physical one depending on the declared dependencies between them.


Where to get help
-----------------

Please file issues in the `the issue tracker <https://github.com/fjarri/grunnur/issues>`_.

Discussions and questions are handled by Github's `discussion board <https://github.com/fjarri/grunnur/discussions>`_.


Table of contents
-----------------

.. toctree::
   :maxdepth: 2

   introduction
   modules
   api
   history


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
