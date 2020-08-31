.. _tutorial-modules:

Tutorial: modules and snippets
==============================

Modules and snippets are important primitives in Grunnur.
Even if you do not write modules yourself, you will most likely use operations from the :py:mod:`~grunnur.functions` module, which are essentially module factories (callables returning :py:class:`~grunnur.Module` objects).
Therefore it helps if you know how they work under the hood.


Snippets
--------

Snippets are ``Mako`` template defs (essentially functions returning rendered text) with the associated dictionary of render globals.
When a snippet is used in a template, the result is quite straightworward: its template function is called, rendering and returning its contents, just as a normal ``Mako`` def.

Let us demonstrate it with a simple example.
Consider the following snippet:

::

    add = Snippet.from_callable(
        lambda varname: """
        ${varname} + ${num}
        """,
        render_globals=dict(num=1))

Now we can compile a template which uses this snippet:

::

    program = Program(
        context,
        """
        KERNEL void test(GLOBAL_MEM int *arr)
        {
            const SIZE_T idx = get_global_id(0);
            int x = arr[idx];
            arr[idx] = ${add('x')};
        }
        """,
        render_globals=dict(add=add))

As a result, the code that gets compiled is

::

    KERNEL void test(GLOBAL_MEM int *arr)
    {
        const SIZE_T idx = get_global_id(0);
        int x = arr[idx];
        arr[idx] = x + 1;
    }

If the snippet is used without parentheses (e.g. ``${add}``), it is equivalent to calling it without arguments (``${add()}``).

The root code that gets passed to :py:meth:`~grunnur.Program` can be viewed as a snippet with an empty signature.


Modules
-------

Modules are quite similar to snippets in a sense that they are also ``Mako`` defs with an associated dictionary of render keywords.
The difference lies in the way they are processed.
Consider a module containing a single function:

::

    add = Module.from_callable(
        lambda prefix, arg: """
        FUNCTION int ${prefix}(int x)
        {
            return x + ${num} + ${arg};
        }
        """,
        name="foobar",
        render_globals=dict(num=1))

Modules contain complete C entities (function, macros, structures) and get rendered in the root level of the source file.
In order to avoid name clashes, their def gets a string as a first argument, which it has to use to prefix these entities' names.
If the module contains only one entity that is supposed to be used by the parent code, it is a good idea to set its name to ``prefix`` only, to simplify its usage.

Let us now create a kernel that uses this module:

::

    program = Program(
        context,
        """
        KERNEL void test(GLOBAL_MEM int *arr)
        {
            const SIZE_T idx = get_global_id(0);
            int x = arr[idx];
            arr[idx] = ${add(2)}(x);
        }
        """,
        render_globals=dict(add=add))

Before the compilation render keywords are inspected, and if a module object is encountered, the following things happen:

1. This object's ``render_globals`` are inspected recursively and any modules there are rendered in the same way as described here, producing a source file.
2. The module itself gets assigned a new prefix and its template function is rendered with this prefix as the first argument, with the positional arguments given following it.
   The result is attached to the source file.
3. The corresponding value in the current ``render_globals`` is replaced by the newly assigned prefix.

With the code above, the rendered module will produce the code

::

    FUNCTION int _mod_foobar_0_(int x)
    {
        return x + 1 + 2;
    }

and the ``add`` keyword in the ``render_globals`` gets its value changed to ``_mod_foobar_0_``.
Then the main code is rendered and appended to the previously renderd parts, giving

::

    FUNCTION int _mod_foobar_0_(int x)
    {
        return x + 1 + 2;
    }

    KERNEL void test(GLOBAL_MEM int *arr)
    {
        const SIZE_T idx = get_global_id(0);
        int x = arr[idx];
        arr[idx] = _mod_foobar_0_(x);
    }

which is then passed to the compiler.
If your module's template def does not take any arguments except for ``prefix``, you can call it in the parent template just as ``${add}`` (without empty parentheses).

.. warning::

    Note that ``add`` in this case is not a string, it is an object that has ``__str__()`` defined.
    If you want to concatenate a module prefix with some other string, you have to either call ``str()`` explicitly (``str(add) + "abc"``), or concatenate it inside a template (``${add}abc``).

Modules can reference snippets in their ``render_globals``, which, in turn, can reference other modules.
This produces a tree-like structure with the snippet made from the code passed by user at the root.
When it is rendered, it is traversed depth-first, modules are extracted from it and arranged in a flat list in the order of appearance.
Their positions in ``render_globals`` are replaced by assigned prefixes.
This flat list is then rendered, producing a single source file being fed to the compiler.

Note that if the same module object was used without arguments in several other modules or in the kernel itself, it will only be rendered once.
Therefore one can create a "root" module with the data structure declaration and then use that structure in other modules without producing type errors on compilation.


Other constructors
------------------

If the arguments are not know at compile time, you can use :py:meth:`DefTemplate.from_string <grunnur.DefTemplate.from_string>` with a regular constructor:

::

    argnames = ['varname']
    add = Snippet(
        DefTemplate.from_string("_func", argnames, "${varname} + ${num}"),
        render_globals=dict(num=1))

Modules can be constructed in a similar way.
The only difference is that the template must have at least one positional parameter which will receive the prefix value.

Alternatively, one can create a snippet with no parameters or a module with a single prefix parameter with a ``from_string()`` constructor:

::

    add = Module.from_string("""
        FUNCTION int ${prefix}(int x)
        {
            return x + ${num};
        }
        """,
        render_globals=dict(num=1))


Module and snippet discovery
----------------------------

Sometimes you may want to pass a module or a snippet inside a template as an attribute of a custom object.
In order for CLUDA to be able to discover and process it without modifying your original object, you need to make your object comply to a discovery protocol.
The protocol method takes a processing function and is expected to return a **new object** of the same class with the processing function applied to all the attributes that may contain a module or a snippet.
By default, objects of type ``tuple``, ``list``, and ``dict`` are discoverable.

For example:

::

    class MyClass:

        def __init__(self, coeff, mul_module, div_module):
            self.coeff = coeff
            self.mul = mul_module
            self.div = div_module

        def __process_modules__(self, process):
            return MyClass(self.coeff, process(self.mul), process(self.div))


Nontrivial example
------------------

Modules were introduced to help split big kernels into small reusable pieces which in ``CUDA`` or ``OpenCL`` program would be put into different source or header files.
For example, a random number generator may be assembled from a function generating random integers, a function transforming these integers into random numbers with a certain distribution, and a parallel computation calling these functions and saving results to global memory.
These functions can be extracted into separate modules, so that a user could call them from some custom kernel if he does not need to store the intermediate results.

Going further with this example, one notices that functions that produce randoms with sophisticated distributions are often based on simpler distributions.
For instance, the commonly used Marsaglia algorithm for generating Gamma-distributed random numbers requires several uniformly and normally distributed randoms.
Normally distributed randoms, in turn, require several uniformly distributed randoms --- with the range which differs from the one for uniformly distributed randoms used by the initial Gamma distribution.
Instead of copy-pasting the function or setting its parameters dynamically (which in more complicated cases may affect the performance), one just specifies the dependencies between modules and lets the underlying system handle things.

The final render tree may look like:

::

    Snippet(
        PureParallel,
        render_globals={
            base_rng -> Snippet(...)
            gamma -> Snippet(
        }                Gamma,
    )                    render_globals = {
                             uniform -> Snippet(...)
                             normal -> Snippet(
                         }                 Normal,
                     )                     render_globals = {
                                               uniform -> Snippet(...)
                                           }
                                       )
