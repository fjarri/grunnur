# Grunnur, a base layer for GPGPU

[![Documentation status][docs-image]][docs-link] [![CI status][ci-image]][ci-link] [![Coverage status][cov-image]][cov-link]

[docs-image]: https://readthedocs.org/projects/grunnur/badge/?version=latest
[docs-link]: http://grunnur.readthedocs.org/en/latest/?badge=latest
[ci-image]: https://travis-ci.org/fjarri/grunnur.svg?branch=master
[ci-link]: https://travis-ci.org/fjarri/grunnur
[cov-image]: https://coveralls.io/repos/github/fjarri/grunnur/badge.svg?branch=master
[cov-link]: https://coveralls.io/github/fjarri/grunnur?branch=master


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
