"""The example illustrating how to manually use a temporary array manager (if you must)."""

# ruff: noqa: T201

import numpy

from grunnur import API, Array, Context, Queue, VirtualManager, ZeroOffsetManager, dtypes


def demo_array_dependencies(queue: Queue) -> None:
    # ZeroOffsetManager attempts to pack temporary allocations
    # in a collection of real allocations with minimal total size.
    # All the virtual allocations start at the beginning of the real allocations.

    # Create a manager that will try to minimize the total size of real allocations
    # every time a temporary allocation occurs, or a temporary array is freed.
    # Note that this may involve re-pointing a temporary array to a different part of memory,
    # so all of the data in it is lost.
    temp_manager = ZeroOffsetManager(queue.device)

    # Alternatively one can pass `False` to these keywords and call `.pack()` manually.
    # This can be useful if a lot of allocations are happening in a specific place at once.

    # Create two arrays that do not depend on each other.
    # This means the manager will allocate a single (200, int32) real array,
    # and point both `a1` and `a2` to its beginning.
    _a1 = Array.empty(queue.device, (100,), numpy.int32, allocator=temp_manager.allocator())
    _a2 = Array.empty(queue.device, (200,), numpy.int32, allocator=temp_manager.allocator())
    temp_manager.pack(queue)

    # You can see that the total size of virtual arrays is 1200,
    # but the total size of real arrays is only 800 (the size of the larger array).
    print("Allocated a1 = (100, int32) and a2 = (200, int32)")
    print(temp_manager.statistics())

    # Now we allocate a dependent array.
    # This means that the real memory `a3` points to cannot intersect with that of `a1`.
    # If we could point temporary arrays at any address within real allocations,
    # we could fit it into the second half of the existing real allocation.
    # But `ZeroOffsetManager` cannot do that, so it has to create another allocation.
    _a3 = Array.empty(
        queue.device, (100,), numpy.int32, allocator=temp_manager.allocator(dependencies=_a1)
    )
    temp_manager.pack(queue)

    print("Allocated a3 = (100, int32) depending on a1")
    print(temp_manager.statistics())

    # Now that we deallocated `a1`, `a3` can now fit in the same real allocation as `a2`,
    # so one of the real allocations will be removed.
    del _a1
    temp_manager.pack(queue)

    print("Freed a1")
    print(temp_manager.statistics())


class MyComputation:
    def __init__(self, temp_manager: VirtualManager):
        self.temp_array = Array.empty(
            queue.device, (100,), numpy.int32, allocator=temp_manager.allocator()
        )

        # The magic property containing temporary arrays used
        self.__virtual_allocations__ = [self.temp_array]


def demo_object_dependencies(queue: Queue) -> None:
    temp_manager = ZeroOffsetManager(queue.device)

    # A `MyComputation` instance creates a temporary array for internal usage
    comp = MyComputation(temp_manager)

    print("MyComputation created")
    print(temp_manager.statistics())

    # Create another temporary array whose usage does not intersect with `MyComputation` usage.
    # This means that if `comp` is called, the contents of `a1` may be rewritten.
    _a1 = Array.empty(queue.device, (100,), numpy.int32, allocator=temp_manager.allocator())

    # It is put in the same real allocation as the temporary array of `comp`.
    print("Allocated a1 = (100, int32)")
    print(temp_manager.statistics())

    # Now let's say we want to put the result of `comp` call somewhere.
    # This means we want to make sure it does not occupy the same memory
    # as any of the temporary arrays in `comp`, so we are passing `comp` as a dependency.
    # It will pick up whatever `comp` declared in its `__virtual_allocations__` attribute.
    _result = Array.empty(
        queue.device, (100,), numpy.int32, allocator=temp_manager.allocator(dependencies=comp)
    )

    # You can see that a new real allocation was created to host the result.
    print("Allocated result = (100, int32)")
    print(temp_manager.statistics())


if __name__ == "__main__":
    context = Context.from_devices([API.any().platforms[0].devices[0]])
    queue = Queue(context.device)

    demo_array_dependencies(queue)
    demo_object_dependencies(queue)
