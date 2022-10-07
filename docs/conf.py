# Configuration file for the Sphinx documentation builder

import os
import textwrap
import colorsys
import webcolors
import numpy as np

# -- Project information -----------------------------------------------------

project = 'sandman'
copyright = '2022, Ellen M. Price'
author = '@emprice'

# The full version, including alpha/beta/rc tags
import sandman
release = sandman.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings
extensions = ['furiosa', 'sphinx.ext.todo', 'sphinx.ext.mathjax',
              'sphinx.ext.autodoc', 'sphinx.ext.intersphinx',
              'hoverxref.extension']
todo_include_todos = True

smartquotes = True
primary_domain = 'py'
highlight_language = 'default'
intersphinx_mapping = { \
    'python' : ('https://docs.python.org/3', None),
    'numpy' : ('https://numpy.org/doc/stable', None),
    'matplotlib' : ('https://matplotlib.org/stable', None),
    'cmasher' : ('https://cmasher.readthedocs.io', None),
    'colorspacious' : ('https://colorspacious.readthedocs.io/en/latest', None),
    'PIL' : ('https://pillow.readthedocs.io/en/stable', None) }

hoverxref_default_type = 'tooltip'
hoverxref_auto_ref = True
hoverxref_domains = ['std', 'py']
hoverxref_roles = ['mod', 'class']
hoverxref_intersphinx = ['python', 'numpy']
hoverxref_ignore_refs = ['genindex', 'modindex', 'search', 'cvd']

html_logo = '_static/sandman.svg'
html_title = 'sandman'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

html_theme = 'furiosa'
pygments_style = 'nordlight'
pygments_dark_style = 'norddark'

# Add any paths that contain templates here, relative to this directory
templates_path = ['_templates']

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory
html_static_path = ['_static']
html_favicon = 'favicon.ico'

html_css_files = ['custom.css']


# -- Fanciness ---------------------------------------------------------------

from docutils.nodes import raw

def hexcolor_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    node = raw(rawtext, f'<span style="color: #{text}">{text}</span>', format='html')
    return [node], []

def rainbow_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    # hue = 0 is red, hue = 5/6 is magenta
    # the number of hues depends on the number of *printable* characters,
    # so that the hue moves along the string linearly
    prnt = [c for c in text if c.isprintable()]
    hues = iter(np.linspace(0, 5. / 6., len(prnt)))

    prettytext = '<span class="rainbow" style="font-style: italic">'

    for c in text:
        if c.isprintable():
            # change to the next hue
            rgb = colorsys.hls_to_rgb(next(hues), 0.5, 1)
            hc = webcolors.rgb_to_hex((np.array(rgb) * 255).astype(int))
            pre = f'<span style="color: {hc}">'
            post = '</span>'
        else:
            pre = post = ''
        prettytext += pre + c + post

    prettytext += '</span>'

    node = raw(rawtext, prettytext, format='html')
    return [node], []

def colormap_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    parts = text.split('.')
    most = '.'.join(parts[:-1])
    last = parts[-1]

    # fetch the colormap
    import importlib
    mod = importlib.import_module(most)
    cmap = getattr(mod, last)

    # the number of colors depends on the number of *printable* characters
    prnt = [c for c in text if c.isprintable()]
    values = iter(np.linspace(0., 1., len(prnt)))

    prettytext = f'<span class="tooltip colormap" title="{text}" style="font-style: italic; font-weight: bold">'

    for c in text:
        if c.isprintable():
            # change to the next value
            rgb = cmap(next(values))[:-1]
            hc = webcolors.rgb_to_hex((np.array(rgb) * 255).astype(int))
            pre = f'<span style="color: {hc}">'
            post = '</span>'
        else:
            pre = post = ''
        prettytext += pre + c + post

    prettytext += '</span>'

    node = raw(rawtext, prettytext, format='html')
    return [node], []

def setup(app):
    app.add_role('hexcolor', hexcolor_role, override=True)
    app.add_role('rainbow', rainbow_role, override=True)
    app.add_role('colormap', colormap_role, override=True)
    return dict(version='1.0', parallel_read_safe=True, parallel_write_safe=True)

# vim: set ft=python:
