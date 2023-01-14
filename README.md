# Grunnur, a base layer for GPGPU

[![pypi package][pypi-image]][pypi-link] ![License][pypi-license-image] [![Docs][rtd-image]][rtd-link] [![Coverage][cov-image]][cov-link] [![Code style: black][black-image]][black-link]

[pypi-image]: https://img.shields.io/pypi/v/grunnur
[pypi-link]: https://pypi.org/project/grunnur/
[pypi-license-image]: https://img.shields.io/pypi/l/grunnur
[rtd-image]: https://readthedocs.org/projects/grunnur/badge/?version=latest
[rtd-link]: https://grunnur.readthedocs.io/en/latest/
[cov-image]: https://codecov.io/gh/fjarri/grunnur/branch/master/graph/badge.svg
[cov-link]: https://codecov.io/gh/fjarri/grunnur
[black-image]: https://img.shields.io/badge/code%20style-black-000000.svg
[black-link]: https://github.com/psf/black


# What's with the name?

"Grunnur" means "foundation" in Icelandic.


# What does it do?

Grunnur is a thin layer on top of [PyCUDA](http://documen.tician.de/pycuda) and [PyOpenCL](http://documen.tician.de/pyopencl) that makes it easier to write platform-agnostic programs.
It is a reworked `cluda` submodule of [Reikna](http://reikna.publicfields.net), extracted into a separate module.

**Warning:** The current version is not very stable and the public API is subject to change as I'm transferring the functionality from Reikna and extending it to support multi-GPU configurations. Bug reports are welcome, and especially welcome are any suggestions about the public API.


# Main features

* For the majority of cases, allows one to write platform-independent code.
* Simple usage of multiple GPUs (in particular, no need to worry about context switching for CUDA).
* A way to split kernel code into modules with dependencies between them (see Modules and Snippets).
* Various mathematical functions (with complex numbers support) organized as modules.
* Static kernels, where you can use global/local shapes with any kinds of dimensions without worrying about assembling array indices from `blockIdx` and `gridIdx`.
* A temporary buffer manager that can pack several virtual buffers into the same physical one depending on the declared dependencies between them.
