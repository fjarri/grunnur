Introduction
============

Grunnur is an abstraction layer on top of PyCUDA/PyOpenCL.
Its main purpose is to provide a uniform API for high-level GPGPU algorithms
automating some common tasks.

Consider the following example, which is very similar to the one from the index page on PyCUDA documentation:

.. testcode:: grunnur_simple_example

    import numpy
    from grunnur import any_api, Context, Queue, Program, Array

    N = 256

    context = Context.from_devices([any_api.platforms[0].devices[0]])
    queue = Queue(context.device)

    program = Program(
        [context.device],
        """
        KERNEL void multiply_them(
            GLOBAL_MEM float *dest,
            GLOBAL_MEM float *a,
            GLOBAL_MEM float *b)
        {
            const SIZE_T i = get_global_id(0);
            dest[i] = a[i] * b[i];
        }
        """)

    multiply_them = program.kernel.multiply_them

    a = numpy.random.randn(N).astype(numpy.float32)
    b = numpy.random.randn(N).astype(numpy.float32)
    a_dev = Array.from_host(queue, a)
    b_dev = Array.from_host(queue, b)
    dest_dev = Array.empty(context.device, a.shape, a.dtype)

    multiply_them(queue, [N], None, dest_dev, a_dev, b_dev)
    print((dest_dev.get(queue) - a * b == 0).all())

.. testoutput:: grunnur_simple_example
    :hide:

    True

If you are familiar with ``PyCUDA`` or ``PyOpenCL``, you will easily understand most of the the steps we have made here.
The ``any_api`` object returns some API of the ones available (so, depending of whether ``PyOpenCL`` or ``PyCUDA`` are installed).
More precise control over API is available via :ref:`API discovery functions <api-discovery>`.

The abstraction from specific C interface of OpenCL or CUDA is achieved by using generic API module on the Python side, and special macros (:c:macro:`KERNEL`, :c:macro:`GLOBAL_MEM`, :ref:`and others <kernel-toolbox>`) on the kernel side.

The argument of :py:class:`~grunnur.Program` constructor can also be a template, which is quite useful for metaprogramming, and also used to compensate for the lack of complex number operations in CUDA and OpenCL.
Let us illustrate both scenarios by making the initial example multiply complex arrays.
The template engine of choice in ``grunnur`` is `Mako <http://www.makotemplates.org>`_, and you are encouraged to read about it as it is quite useful. For the purpose of this example all we need to know is that ``${python_expression()}`` is a synthax construction which renders the expression result.

.. testcode:: grunnur_template_example

    import numpy
    from numpy.linalg import norm
    import grunnur.dtypes as dtypes
    import grunnur.functions as functions
    from grunnur import any_api, Context, Queue, Program, Array

    context = Context.from_devices([any_api.platforms[0].devices[0]])
    queue = Queue(context.device)

    N = 256
    dtype = numpy.complex64

    program = Program(
        [context.device],
        """
        KERNEL void multiply_them(
            GLOBAL_MEM ${ctype} *dest,
            GLOBAL_MEM ${ctype} *a,
            GLOBAL_MEM ${ctype} *b)
        {
          const SIZE_T i = get_global_id(0);
          dest[i] = ${mul}(a[i], b[i]);
        }
        """,
        render_globals=dict(
            ctype=dtypes.ctype(dtype),
            mul=functions.mul(dtype, dtype)))

    multiply_them = program.kernel.multiply_them

    r1 = numpy.random.randn(N).astype(numpy.float32)
    r2 = numpy.random.randn(N).astype(numpy.float32)
    a = r1 + 1j * r2
    b = r1 - 1j * r2
    a_dev = Array.from_host(queue, a)
    b_dev = Array.from_host(queue, b)
    dest_dev = Array.empty(context.device, a.shape, a.dtype)

    multiply_them(queue, [N], None, dest_dev, a_dev, b_dev)
    print(norm(dest_dev.get(queue) - a * b) / norm(a * b) <= 1e-6)

.. testoutput:: grunnur_template_example
    :hide:

    True

Here we have passed two values to the template: ``ctype`` (a string with C type name), and ``mul`` which is a :py:class:`~grunnur.Module` object containing a single multiplication function.
The object is created by a function :py:func:`~grunnur.functions.mul` which takes data types being multiplied and returns a module that was parametrized accordingly.
Inside the template the variable ``mul`` is essentially the prefix for all the global C objects (functions, structures, macros etc) from the module.
If there is only one public object in the module (which is recommended), it is a common practice to give it the name consisting just of the prefix, so that it could be called easily from the parent code.

For more information on modules, see :ref:`tutorial-modules`; the complete list of things available in Grunnur can be found in :ref:`API reference <api-reference>`.
