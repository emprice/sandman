![sandman logo](/docs/_static/sandman.svg)

[![build-docs action](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/emprice/sandman/gh-pages/endpoint.json)](https://github.com/emprice/sandman/actions/workflows/main.yml)
[![License: MIT](https://img.shields.io/github/license/emprice/sandman?style=for-the-badge)](https://opensource.org/licenses/MIT)
![CodeFactor Grade](https://img.shields.io/codefactor/grade/github/emprice/sandman/main?logo=codefactor&style=for-the-badge)
![GitHub Repo stars](https://img.shields.io/github/stars/emprice/sandman?style=for-the-badge)

**Full documentation is available [here](https://emprice.github.io/sandman).**

 + [Project motivation](#rainbow-project-motivation)
 + [Building and installing](#nut_and_bolt-building-and-installing)
 + [Credits and dependencies](#gem-credits-and-dependencies)

# :rainbow: Project motivation

Making scientific data and visualizations accessible to as many people as
possible is clearly an excellent goal. However, everyone has different
aesthetic preferences, and sacrificing visual appeal for accessibility is
not generally necessary. If you have found that you are unsatisfied or bored
with existing perceptually uniform colormaps, this project may be what you need.

# :nut_and_bolt: Building and installing

To build and install this package with `pip`, the same procedure can be
followed as for most pure Python packages that use `setuptools`. In the
install step, some optional flags are given below in brackets that can be
useful for a non-root build (`--user`) or for active development
(`--ignore-installed --force-reinstall`), but these are not strictly
necessary to install the package itself.

```bash
pip3 install [--force-reinstall --ignore-installed] [--user] git+https://github.com/emprice/sandman.git@main
```

# :gem: Credits and dependencies

This package depends on several packages not in the Python standard
library, including the following.

 + [NumPy](https://numpy.org) and [SciPy](https://scipy.org) are instrumental
   for optimizing colormaps and working with the numeric data efficiently.
 + [colorspacious](https://colorspacious.readthedocs.io/en/latest) is used
   for modeling human vision perception.
 + [webcolors](https://webcolors.readthedocs.io) is used for parsing
   hexadecimal color strings, which is currently the only recognized
   user-facing input format.

The command-line tool further relies on the following extra dependencies.

 + [Matplotlib](https://matplotlib.org) and
   [Pillow](https://pillow.readthedocs.io) are needed for previewing and
   simulating the colormaps.
 + [pronounceable](https://pypi.org/project/pronounceable) is used for
   generating random, but pronounceable, colormap names when maps are
   generated in bulk.

Finally, the following dependencies deserve special recognition for their
use in the HTML documentation.

 + [Furo](https://github.com/pradyunsg/furo) is the base theme used in
   conjunction with [Sphinx](https://www.sphinx-doc.org/en/master) to build
   the documentation pages.
 + The [Nord](https://www.nordtheme.com) colorscheme is a beautiful set
   of color palettes with a frosty blue feel. The documentation colors and
   example colormaps are based on these colors.

<!-- vim: set ft=markdown: -->
