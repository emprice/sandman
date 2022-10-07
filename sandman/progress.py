''' Wrappers for tqdm progress bars used by the CLI. Putting this in
its own file so that an ImportError (for tqdm, since it is an optional
dependency) is easily recoverable.
'''

import tqdm
from .obnoxious import rainbowify

class FancyProgressBar(tqdm.tqdm):
    ''' Same as standard tqdm progress bar, but enforces ASCII mode and
    applies the rainbow effect to the bar.
    '''

    def __init__(self, *args, **kwargs):
        kwargs['ascii'] = True
        super().__init__(*args, **kwargs)

    def __str__(self):
        base = super().__str__()
        return rainbowify(base)

class BoringProgressBar(tqdm.tqdm):
    ''' Same as standard tqdm progress bar, but enforces ASCII mode.
    '''

    def __init__(self, *args, **kwargs):
        kwargs['ascii'] = True
        super().__init__(*args, **kwargs)

# vim: set ft=python:
