''' Make a string look like rainbow vomit, unapologetically.
'''

import colorsys
import numpy as np

def make_rgb_escape(rgb):
    ''' Produces the terminal escape sequence to change the foreground
    color to the given RGB color.

    :param rgb: Red, green, and blue channel intensities
    :type rgb: tuple of float
    :return: ANSI escape sequence
    :rtype: str
    '''
    r, g, b = map(lambda x: int(255 * x), rgb)
    return f'\x1b[38;2;{r};{g};{b}m'

def rainbowify(s):
    ''' Applies the rainbow effect to the given string.

    :param s: Original, boring string
    :type s: str
    :return: New, beautiful string
    :rtype: str
    '''
    # start with the reset sequence
    pretty = '\x1b[0m'

    # hue = 0 is red, hue = 5/6 is magenta
    # the number of hues depends on the number of *printable* characters,
    # so that the hue moves along the string linearly
    prnt = [c for c in s if c.isprintable()]
    hues = iter(np.linspace(0, 5. / 6., len(prnt)))

    for c in s:
        if c.isprintable():
            # change to the next hue
            rgb = colorsys.hls_to_rgb(next(hues), 0.5, 1)
            esc = make_rgb_escape(rgb)
        else:
            esc = ''
        pretty += esc + c

    # end with the reset sequence
    pretty += '\x1b[0m'
    return pretty

# vim: set ft=python:
