import os.path

from mako.template import Template


class TemplateWrapper:

    def __init__(self, template):
        self.name = "<root>"
        self.template = template
        self.source = template.source

    def render(self, *args, **kwds):
        return self.template.render(*args, **kwds)

    def get_def(self, name):
        return DefTemplateWrapper(name, self.template.get_def(name))


class DefTemplateWrapper:

    def __init__(self, name, template_def):
        self.name = name
        self.template_def = template_def
        self.source = template_def.source

    def render(self, *args, **kwds):
        return self.template_def.render(*args, **kwds)


def make_template(template, filename=False):
    kwds = dict(
        strict_undefined=True,
        imports=['import numpy'])

    # Creating a template from a filename results in more comprehensible stack traces,
    # so we are taking advantage of this if possible.
    if filename:
        kwds['filename'] = template
        return TemplateWrapper(Template(**kwds))
    else:
        return TemplateWrapper(Template(template, **kwds))


def template_for(filename):
    """
    Returns the Mako template object created from the file
    which has the same name as ``filename`` and the extension ``.mako``.
    Typically used in computation modules as ``template_for(__filename__)``.
    """
    name, _ext = os.path.splitext(os.path.abspath(filename))
    return make_template(name + '.mako', filename=True)


def render_template(template, *args, **kwds):
    try:
        return template.render(*args, **kwds)
    except: # TODO catch an actual error
        print(
            "Failed to render template with\n"
            f"args: {args}\nkwds: {kwds}\nsource:\n{template.source}\n")
        raise
