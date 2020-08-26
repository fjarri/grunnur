import pytest

from grunnur.template import DefTemplate, RenderError
from grunnur.modules import Snippet, Module, render_with_modules


def test_snippet_from_callable():
    snippet = Snippet.from_callable(lambda x, y: "${x} + ${y} + ${z}", render_globals=dict(z=3))
    res = render_with_modules("${s(1, 2)}", render_globals=dict(s=snippet)).strip()
    assert res == "1 + 2 + 3"


def test_snippet_from_string():
    snippet = Snippet.from_string("${z}", render_globals=dict(z=3))
    res = render_with_modules("${s}", render_globals=dict(s=snippet)).strip()
    assert render_with_modules(snippet).strip() == "3"


def test_module_from_callable():
    module = Module.from_callable(
        lambda prefix, y: "${prefix} + ${y} + ${z}", render_globals=dict(z=3), name="foo")
    res = render_with_modules("module call: ${m(4)}", render_globals=dict(m=module)).strip()
    # The module's source gets rendered and attached at the beginning of the main template,
    # instead of being rendered inplace like in Snippet's case
    assert res == "_mod_foo_0_ + 4 + 3\n\n\nmodule call: _mod_foo_0_"


def test_module_from_string():
    module = Module.from_string("${prefix} + ${z}", render_globals=dict(z=3), name="foo")
    res = render_with_modules("module call: ${m}", render_globals=dict(m=module)).strip()
    # The module's source gets rendered and attached at the beginning of the main template,
    # instead of being rendered inplace like in Snippet's case
    assert res == "_mod_foo_0_ + 3\n\n\nmodule call: _mod_foo_0_"


def test_render_snippet():
    snippet = Snippet.from_callable(lambda x, y: "${x} + ${y} + ${z}", render_globals=dict(z=3))
    assert render_with_modules(snippet, render_args=[1, 2]).strip() == "1 + 2 + 3"


def test_render_snippet_with_render_globals():
    # Check that provided render globals are added to those of the snippet
    snippet = Snippet.from_callable(lambda x, y: "${x} + ${y} + ${z} + ${q}", render_globals=dict(z=3))
    assert render_with_modules(snippet, render_args=[1, 2], render_globals=dict(q=4)).strip() == "1 + 2 + 3 + 4"
    with pytest.raises(ValueError, match="Cannot add a global 'z' - it already exists"):
        render_with_modules(snippet, render_args=[1, 2], render_globals=dict(z=5, q=4))


def test_render_string():
    assert render_with_modules("abcde").strip() == "abcde"


def test_render_string_with_args():
    with pytest.raises(ValueError):
        render_with_modules("abcde", render_args=[1, 2])


def test_render_callable():
    res = render_with_modules(
        lambda x, y: "${x} + ${y} + ${z}",
        render_args=[1, 2],
        render_globals=dict(z=3)).strip()
    assert res == "1 + 2 + 3"


def test_render_def_template():
    tmpl = DefTemplate.from_callable("test", lambda x, y: "${x} + ${y} + ${z}")
    res = render_with_modules(tmpl,
        render_args=[1, 2],
        render_globals=dict(z=3)).strip()
    assert res == "1 + 2 + 3"


def test_render_unknown_type():
    with pytest.raises(TypeError):
        render_with_modules(1)


def test_render_error():
    module = Module.from_callable(
        lambda prefix, x: "${prefix} + ${x} + ${bar}", render_globals=dict(baz=3), name="foo")
    with pytest.raises(RenderError) as exc_info:
        render_with_modules("module call: ${m(1)}", render_globals=dict(m=module))

    assert exc_info.type == RenderError

    # Check that we get a correct info from a render error nested in the module hierarchy
    e = exc_info.value

    assert e.args == ('_mod_foo_0_', 1)
    assert e.globals == dict(baz=3)
    assert type(e.exception) == NameError
    assert e.source == module.template.source


class CustomObj:
    """
    A class supporting custom module processing.
    """
    def __init__(self, module):
        self.module = module

    def __process_modules__(self, process):
        return RenderableCustomObj(process(self.module))


class RenderableCustomObj:

    def __init__(self, processed_module):
        self.module = processed_module


def test_process_objects():
    # Checks that all supported types of objects are correctly traversed
    # in search for Modules.


    m1 = Module.from_string("m1: ${prefix}", name="m1")
    m2 = Module.from_string("m2: ${prefix}", name="m2")
    m3 = Module.from_string("m3: ${prefix}", name="m3")
    m4 = Module.from_string("m4: ${prefix}", name="m4")
    m5 = Module.from_string("m5: ${prefix}", name="m5")

    res = render_with_modules(
        """
        ${module_obj}
        ${type(custom_obj) == ref_type} ${custom_obj.module}
        ${dict_obj['module']}
        ${list_obj[0]}
        ${tuple_obj[0]}
        ${non_module_obj}
        """,
        render_globals=dict(
            module_obj=m1,
            custom_obj=CustomObj(m2),
            ref_type=RenderableCustomObj,
            dict_obj=dict(module=m3),
            list_obj=[m4],
            tuple_obj=(m5,),
            non_module_obj=1
            )).strip()

    assert res == (
        "m1: _mod_m1_0_\n\n\n"
        "m2: _mod_m2_1_\n\n\n"
        "m3: _mod_m3_2_\n\n\n"
        "m4: _mod_m4_3_\n\n\n"
        "m5: _mod_m5_4_\n\n\n\n"
        "        _mod_m1_0_\n"
        "        True _mod_m2_1_\n"
        "        _mod_m3_2_\n"
        "        _mod_m4_3_\n"
        "        _mod_m5_4_\n"
        "        1")


def test_module_cache():

    # a module with no parameters
    m1 = Module.from_callable(lambda prefix: "m1: ${prefix}", name="m1")

    # a module with several parameters
    m2 = Module.from_callable(lambda prefix, x, y: "m2: ${prefix} ${x} ${y}", name="m2")

    res = render_with_modules(
        """
        ${m1}
        ${m1}
        ${m2(1, 2)}
        ${m2(2, 3)}
        ${m2(1, 2)}
        """,
        render_globals=dict(m1=m1, m2=m2)).strip()

    assert res == (
        "m1: _mod_m1_0_\n\n\n"
        "m2: _mod_m2_1_ 1 2\n\n\n"
        "m2: _mod_m2_2_ 2 3\n\n\n\n"
        "        _mod_m1_0_\n"
        "        _mod_m1_0_\n"
        "        _mod_m2_1_\n"
        "        _mod_m2_2_\n"
        "        _mod_m2_1_")
