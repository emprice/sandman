import re
import json
import random
import colorsys
import argparse
import functools
import numpy as np
from .obnoxious import rainbowify

class FancyHelpAction(argparse._HelpAction):
    ''' Overrides the default "help" action in :module:`argparse` by passing
    along the user-specified flag to the parser when printing the help
    message.
    '''

    def __call__(self, parser, namespace, values, option_string=None):
        ''' Formats the help message for the parser.

        :param parser: Parser requesting the formatting
        :type parser: FancyArgparse
        :param namespace: Namespace with user options
        :type namespace: argparse.Namespace
        '''
        boring_flag = False
        if hasattr(namespace, 'boring'):
            boring_flag = namespace.boring
        parser.print_help(boring_flag)
        parser.exit()

class FancyArgparse(argparse.ArgumentParser):
    ''' Variant of the default :class:`argparse.ArgumentParser` with some
    minor changes useful for this tool.
    '''

    def __init__(self, *args, **kwargs):
        ''' Initializes the parser with different handling for help messages
        and removing terminal coloring.
        '''

        # get any values that the user passed
        add_help = kwargs.get('add_help', True)
        add_boring = kwargs.get('add_boring', True)

        # sanitize the inputs to pass along to the superclass
        kwargs['add_help'] = False
        kwargs.pop('add_boring', None)
        super().__init__(*args, **kwargs)

        if add_help:
            # add a help option, but using the fancy action
            self.register('action', 'fancy_help', FancyHelpAction)
            self.add_argument('-h', '--help', action='fancy_help',
                default=argparse.SUPPRESS, help='Show this help message and quit')

        if add_boring:
            # add a boring option; nothing special here, but it's needed by
            # multiple parsers, so putting it here is convenient
            self.add_argument('--boring', action='store_true',
                help='Make text output less shiny by using default terminal settings')

    def _parse_known_args(self, arg_strings, namespace):
        ''' Overrides the default parsing behavior so we can move options
        around if needed.
        '''
        if '--help' in arg_strings:
            # help option needs to be parsed last for boring option to
            # have any effect; there may be a more elegant solution to this
            arg_strings.remove('--help')
            arg_strings.append('--help')

        # redirect to the default handler
        return super()._parse_known_args(arg_strings, namespace)

    def add_argument_group(self, name, *rest, **kwargs):
        ''' Capitalizes the first letter of the group name and then
        forwards the new parameters to the superclass.
        '''
        namefix = name[0].upper() + name[1:]
        return super().add_argument_group(namefix, *rest, **kwargs)

    def print_help(self, boring_flag):
        ''' If the user requests no coloring, this function removes all
        escape sequences from the help string; otherwise, this is a no-op.
        '''
        old_desc = self.description
        if boring_flag and old_desc:
            # this regex removes all escape sequences; using *? instead
            # of * is key, because, without the lazy mode, this can
            # also remove "m" characters from the text
            p = re.compile('\x1b' + r'\[(\d+;)*?\d+m')
            self.description = re.sub(p, '', old_desc)
        super().print_help()
        self.description = old_desc

class SubcommandHelpFormatter(argparse.RawDescriptionHelpFormatter):
    ''' Source: https://stackoverflow.com/a/13429281/1552418

    A simple wrapper around :class:`argparse.RawDescriptionHelpFormatter`
    with better formatting for subcommands.
    '''

    def _format_action(self, action):
        ''' Removes the first part of the subparser help formatted by the
        base class and returns the newline-joined pieces.
        '''
        parts = super()._format_action(action)
        if action.nargs == argparse.PARSER:
            parts = '\n'.join(parts.split('\n')[1:])
        return parts

    def _format_usage(self, usage, actions, groups, prefix):
        ''' Wraps the default usage message but overrides the default prefix.
        '''
        return super()._format_usage(usage, actions, groups, 'Usage: ')

class NoPrefixHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    ''' A simple wrapper around :class:`argparse.ArgumentDefaultsHelpFormatter`
    that does not print a usage prefix (to suppress a duplicate prefix when
    subcommands are used.
    '''

    def _format_usage(self, usage, actions, groups, prefix):
        ''' Wraps the default usage message but removes any prefix.
        '''
        return super()._format_usage(usage, actions, groups, '')

def make_colormap_name(*stuff):
    ''' Based on the :code:`generate_word` function from the package
    https://github.com/patarapolw/pronounceable, but enforces a
    random seed based on the input parameters so the output is reproducible.
    '''
    from pronounceable.components import INITIAL_CONSONANTS, \
        FINAL_CONSONANTS, double_vowels

    seed = ','.join(list(map(str, stuff)))
    random.seed(seed)

    return random.choice(INITIAL_CONSONANTS) \
        + random.choice(random.choice(['aeiouy', list(double_vowels())])) \
        + random.choice(['', random.choice(FINAL_CONSONANTS)])

def sequential_task(colors, **kwargs):
    ''' Core task for generating sequential colormaps.

    Keywords are passed directly to :func:`find_optimal_params_sequential` and
    used to generate a colormap name. Supported keywords are currently
    :code:`lightness_base`, :code:`saturation_base`, :code:`min_intensity`,
    and :code:`max_intensity`.

    :param colors: Sequence of colors to be optimized
    :type colors: list
    :return: Optimized colormap data
    :rtype: dict
    '''
    from .sandman import find_optimal_params_sequential

    hexcolors = list(map(lambda s: '#' + s, colors))
    params, score = find_optimal_params_sequential(hexcolors, **kwargs)
    name = make_colormap_name('seq', *colors, *list(kwargs.values()))
    return dict(name=name, params=list(params), colors=colors, score=score)

def diverging_task(colors, **kwargs):
    ''' Core task for generating diverging colormaps.

    Keywords are passed directly to :func:`find_optimal_params_diverging` and
    used to generate a colormap name. Supported keywords are currently
    :code:`lightness_base`, :code:`saturation_base`, :code:`min_intensity`,
    and :code:`max_intensity`.

    :param colors: Sequence of colors to be optimized
    :type colors: list
    :return: Optimized colormap data
    :rtype: dict
    '''
    from .sandman import find_optimal_params_diverging

    hexcolors = list(map(lambda s: '#' + s, colors))
    params, score = find_optimal_params_diverging(hexcolors, **kwargs)
    name = make_colormap_name('div', *colors, *list(kwargs.values()))
    return dict(name=name, params=list(params), colors=colors, score=score)

def cyclic_task(colors, **kwargs):
    ''' Core task for generating cyclic colormaps.

    Keywords are passed directly to :func:`find_optimal_params_diverging` and
    used to generate a colormap name. Supported keywords are currently
    :code:`lightness_base`, :code:`saturation_base`, :code:`min_intensity`,
    and :code:`max_intensity`.

    :param colors: Sequence of colors to be optimized
    :type colors: list
    :return: Optimized colormap data
    :rtype: dict
    '''
    from .sandman import find_optimal_params_diverging

    colors = list(colors)
    colors.append(colors[0])

    hexcolors = list(map(lambda s: '#' + s, colors))
    params, score = find_optimal_params_diverging(hexcolors, **kwargs)
    name = make_colormap_name('cyc', *colors, *list(kwargs.values()))
    return dict(name=name, params=list(params), colors=colors, score=score)

def random_colors(n=1):

    import colorsys
    import webcolors

    hsv = [0., 0.9, 0.9]

    for i in range(n):
        hsv[0] = np.random.uniform()
        rgb = np.array(colorsys.hsv_to_rgb(*hsv))
        rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        yield webcolors.rgb_to_hex(rgb)[1:]

def fake_process_map(fn, iterable, max_workers=1, tqdm_class=None):

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        result = executor.map(fn, iterable)
    return result

def optimize_handler(args):

    import itertools

    try:
        from .progress import FancyProgressBar, BoringProgressBar
        from tqdm.contrib.concurrent import process_map
        ProgressBar = BoringProgressBar if args.boring else FancyProgressBar
    except ImportError:
        ProgressBar = None
        process_map = fake_process_map

    if len(args.colors) > 0:
        colors = args.colors
    else:
        colors = list(random_colors(n=args.pool))

    colors = list(map(lambda s: s.lower(), colors))

    min_colors, max_colors = 3, args.count

    if args.kind == 'diverging' and max_colors % 2 == 0:
        raise ValueError('for divering maps, set --count to an odd number')
    elif args.kind == 'cyclic' and max_colors % 2 == 1:
        raise ValueError('for cyclic maps, set --count to an even number')

    if len(colors) < min_colors:
        raise ValueError(f'need at least {min_colors} colors')

    step = 2 if args.kind in ['diverging', 'cyclic'] else 1
    rvals = [max_colors] if args.exact else range(min_colors, max_colors + 1, step)
    allparams = list()

    if args.kind == 'sequential':
        taskfn = sequential_task
    elif args.kind == 'diverging':
        taskfn = diverging_task
    elif args.kind == 'cyclic':
        taskfn = cyclic_task

    kwargs = dict(lightness_base=args.lightdark / 100.,
        saturation_base=args.saturate / 100.,
        min_intensity=args.lowval, max_intensity=args.highval)
    task = functools.partial(taskfn, **kwargs)

    for r in rvals:
        allcolors = list(itertools.permutations(colors, r=r))
        allparams.extend(process_map(task, allcolors,
            max_workers=args.nproc, tqdm_class=ProgressBar))

    jdata = json.dumps(dict(kind=args.kind, pool=colors,
        optimized=allparams, options=kwargs))
    if args.save:
        with open(args.save, 'w') as f:
            f.write(jdata)
    else:
        print(jdata)

def make_preview_figure(args):

    import matplotlib.cm as cm
    import matplotlib.text as text
    import matplotlib.pyplot as plt
    import matplotlib.colors as clr
    import matplotlib.patches as patches
    import matplotlib.transforms as transforms
    from matplotlib.gridspec import GridSpec

    try:
        from nordplotlib.png import install; install()
    except ImportError:
        pass

    from .sandman import cmap_from_params

    w, h = 8, 0.5
    rows, cols = args.rows, args.cols
    nmaps = rows * cols

    with open(args.input, 'r') as f:
        data = json.loads(f.read())

    opt_data = sorted(data['optimized'], key=lambda p: p['score'])
    opt_data = opt_data[:nmaps]

    kwargs = data['options']
    kw = { k : kwargs[k] for k in ['lightness_base', 'saturation_base'] }

    norm = clr.Normalize(vmin=0, vmax=w)
    cmaps = list(map(lambda p: cmap_from_params(p['name'],
        list(map(lambda s: '#' + s, p['colors'])),
        p['params'], kind=data['kind'], **kw), opt_data))

    X = np.linspace(0, w, 250)
    Y = np.linspace(0, h, 2)
    X, Y = np.meshgrid(X, Y, indexing='ij')

    fig = plt.figure(figsize=(w*cols, h*rows), constrained_layout=True)
    gs = GridSpec(rows, cols, figure=fig)
    axs = np.ravel(gs.subplots())

    for i, (cmap, ax) in enumerate(zip(cmaps, axs)):
        P = ax.pcolormesh(X, Y, X, cmap=cmap, norm=norm)
        P.set_rasterized(True)
        P.set_edgecolor('face')

        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.set_ylabel(f'{cmap.name} ({i+1})', rotation=0,
            ha='right', va='center', fontsize=14)

    return fig

def preview_handler(args):

    import matplotlib.pyplot as plt

    fig = make_preview_figure(args)
    if args.save: fig.savefig(args.save)
    else: plt.show()

def simulate_handler(args):

    from .sandman import simulate_cvd, simulate_grayscale

    fig = make_preview_figure(args)

    if args.cvd == 'achromatomaly':
        img = simulate_grayscale(fig)
    else:
        img = simulate_cvd(fig, kind=args.cvd, severity=args.severity)

    if args.save: img.save(args.save)
    else: img.show()

desc = 'A tool for procedurally generating ' + \
    rainbowify('perceptually uniform colormaps')
parser = FancyArgparse(prog='python -m sandman',
    formatter_class=SubcommandHelpFormatter, description=desc)
subparsers = parser.add_subparsers(title='Available sub-commands',
    parser_class=FancyArgparse)

subparser1 = subparsers.add_parser('optimize',
    help='Generate new colormap data by optimizing random or ' \
         'user-selected colors', formatter_class=NoPrefixHelpFormatter)
subparser1.add_argument('colors', type=str, nargs='*',
    help='User-selected pool of colors from which colormap colors ' \
         'will be selected; use six hex characters with no leading hash')
subparser1.add_argument('--save', type=str, default=None,
    help='Output file for generated colormap data, in JSON format; ' \
         'if unspecified, the data will be printed to stdout')
subparser1.add_argument('--pool', type=int, default=5,
    help='Number of random colors in the pool of available colors; ' \
         'only relevant if colors are not given explicitly')
subparser1.add_argument('--count', type=int, default=3,
    help='Maximum number of colors to include in each map')
subparser1.add_argument('--exact', action='store_true',
    help='Use exactly --count colors, otherwise consider all ' \
         'combinations of lengths 3 to --count')
subparser1.add_argument('--nproc', type=int, default=1,
    help='Number of processors to use for working in parallel')
subparser1.add_argument('--kind', type=str, default='sequential',
    choices=['sequential', 'diverging', 'cyclic'],
    help='Type of colormaps to generate')
subparser1.add_argument('--lightdark', type=int, default=0,
    help='Base lightness value')
subparser1.add_argument('--saturate', type=int, default=0,
    help='Base saturation value')
subparser1.add_argument('--lowval', type=int, default=40,
    help='Minimum intensity value')
subparser1.add_argument('--highval', type=int, default=90,
    help='Maximum intensity value')
subparser1.set_defaults(func=optimize_handler)

subparser2 = subparsers.add_parser('preview',
    help='Preview the "best" colormaps using data computed by the ' \
         'optimization sub-step', formatter_class=NoPrefixHelpFormatter)
subparser2.add_argument('input', type=str,
    help='Existing JSON file containing optimized colormaps')
subparser2.add_argument('--rows', type=int, default=10,
    help='Number of rows of colormaps in the preview image')
subparser2.add_argument('--cols', type=int, default=2,
    help='Number of columns of colormaps in the preview image')
subparser2.add_argument('--save', type=str, default=None,
    help='Output file for the preview image; if unspecified, the ' \
         'image will be shown but not saved')
subparser2.set_defaults(func=preview_handler)

subparser3 = subparsers.add_parser('simulate',
    help='Roughly simulate how the colormaps would appear with ' \
         'different color vision deficiencies',
    formatter_class=NoPrefixHelpFormatter)
subparser3.add_argument('input', type=str,
    help='Existing JSON file containing optimized colormaps')
subparser3.add_argument('--rows', type=int, default=10,
    help='Number of rows of colormaps in the preview image')
subparser3.add_argument('--cols', type=int, default=2,
    help='Number of columns of colormaps in the preview image')
subparser3.add_argument('--save', type=str, default=None,
    help='Output file for the preview image; if unspecified, the ' \
         'image will be shown but not saved')
subparser3.add_argument('--cvd', type=str,
    choices=['protanomaly', 'deuteranomaly', 'tritanomaly', 'achromatomaly'],
    help='Type of color vision deficiency to simulate')
subparser3.add_argument('--severity', type=int, default=100,
    help='Severity of the color vision deficiency')
subparser3.set_defaults(func=simulate_handler)

args = parser.parse_args()
args.func(args)

# vim: set ft=python:
