import pytest
from mako.template import Template as MakoTemplate
import numpy

from grunnur.template import Template, DefTemplate, RenderError, _extract_def_source


def test_extract_def_source():
    src = """
        <%def name="add(varname)">
        ${varname} + ${num}
        </%def>

        <%def name='sub(varname, kwd=">")'>
        ${varname} - ${num}
        </%def>
    """

    def_src = _extract_def_source(src, 'add')
    template = MakoTemplate(def_src)
    assert 'add' in template.list_defs()
    assert 'sub' not in template.list_defs()

    def_src = _extract_def_source(src, 'sub')
    template = MakoTemplate(def_src)
    assert 'sub' in template.list_defs()
    assert 'add' not in template.list_defs()


def test_extract_def_source_missing_def():
    src = """
        <%def name="add(varname)">
        ${varname} + ${num}
        </%def>
    """

    # Technically, this warning and the subsequent return of the full source should occur
    # only if _extract_def_source() encounters a def it cannot recognize, but Mako still can -
    # otherwise get_def() will fail anyway.
    # But if I could think of something that would confuse _extract_def_source(), but
    # not Mako, I would modify _extract_def_source() accordingly.
    # So to model the situtation I'm just trying to find a non-existent def.

    with pytest.warns(SyntaxWarning):
        def_src = _extract_def_source(src, 'sub')
    assert def_src == src


def test_template_from_associated_file():
    template = Template.from_associated_file(__file__)
    assert template.get_def("test").render().strip() == "template body"


def test_template_from_string():
    src = """
        <%def name="test()">
        template body
        </%def>
        """
    template = Template.from_string(src)
    assert template.get_def("test").render().strip() == "template body"


def test_missing_def():
    src = """
        <%def name="test()">
        template body
        </%def>
        """
    template = Template.from_string(src)
    with pytest.raises(AttributeError, match="Template has no def 'foo'"):
        template.get_def("foo")


def test_template_caching():
    src = """
        <%def name="test()">
        template body
        </%def>
        """
    template = Template.from_string(src)
    def1 = template.get_def('test')
    def2 = template.get_def('test')
    assert def1 is def2


def test_def_template_from_callable():
    template = DefTemplate.from_callable("test", lambda x, y: "${x} + ${y}")
    assert template.render(1, 2).strip() == "1 + 2"


def test_def_template_from_string():
    template = DefTemplate.from_string("test", ['x', 'y'], "${x} + ${y}")
    assert template.render(1, 2).strip() == "1 + 2"


def test_render_error():
    template = DefTemplate.from_callable("test", lambda x, y: "${x} + ${y} + ${z}")
    with pytest.raises(RenderError) as e:
        template.render(1, 2, kwd=3)
    assert e.value.args == (1, 2)
    assert e.value.globals == dict(kwd=3)
    assert e.value.source == template.source
    assert type(e.value.exception) == NameError
    assert str(e.value.exception) in str(e.value)


def test_render_error_pass_through():
    # check that if one template tries to render another, the inner RenderError gets propagated
    template1 = DefTemplate.from_callable("test1", lambda x, y: "${x} + ${y} + ${z}")
    template2 = DefTemplate.from_callable("test2", lambda a, t1: "${a} + ${t1.render(a, 1)}")
    with pytest.raises(RenderError) as e:
        template2.render(10, template1, kwd=3)
    assert e.value.args == (10, 1)
    assert e.value.globals == dict()
    assert e.value.source == template1.source
    assert type(e.value.exception) == NameError
    assert str(e.value.exception) in str(e.value)


def test_template_builtins():
    # Check for the builtins we add to every template
    template = DefTemplate.from_callable("test", lambda numpy_ref: "${numpy == numpy_ref}")
    assert template.render(numpy).strip() == "True"

    template = DefTemplate.from_string("test", ['numpy_ref'], "${numpy == numpy_ref}")
    assert template.render(numpy).strip() == "True"

    template = Template.from_string('<%def name="test(numpy_ref)">${numpy == numpy_ref}</%def>')
    template_def = template.get_def("test")
    assert template_def.render(numpy).strip() == "True"

    template = Template.from_associated_file(__file__)
    assert template.get_def("test_builtins").render(numpy).strip() == "True"
