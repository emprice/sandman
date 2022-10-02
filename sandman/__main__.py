import json
import random
import argparse
import numpy as np

def make_colormap_name(*stuff):

    ''' Based on the :code:`generate_word` function from the package
    https://github.com/patarapolw/pronounceable. The only change I've
    made is to use a reproducible random generator.
    '''

    from pronounceable.components import INITIAL_CONSONANTS, \
        FINAL_CONSONANTS, double_vowels

    seed = ','.join(stuff)
    random.seed(seed)

    return random.choice(INITIAL_CONSONANTS) \
        + random.choice(random.choice(['aeiouy', list(double_vowels())])) \
        + random.choice(['', random.choice(FINAL_CONSONANTS)])

def sequential_task(colors):

    from .sandman import find_optimal_params_sequential

    params, score = find_optimal_params_sequential(colors,
        min_intensity=40, max_intensity=90)
    name = make_colormap_name('sequential', *colors)
    return dict(name=name, params=list(params), colors=colors, score=score)

def diverging_task(colors):

    from .sandman import find_optimal_params_diverging

    params, score = find_optimal_params_diverging(colors,
        min_intensity=40, max_intensity=90)
    name = make_colormap_name('diverging', *colors)
    return dict(name=name, params=list(params), colors=colors, score=score)

def cyclic_task(colors):

    from .sandman import find_optimal_params_diverging

    colors = list(colors)
    colors.append(colors[0])
    params, score = find_optimal_params_diverging(colors,
        min_intensity=40, max_intensity=90)
    name = make_colormap_name('cyclic', *colors)
    return dict(name=name, params=list(params), colors=colors, score=score)

def random_colors(n=1):

    import colorsys
    import webcolors

    hsv = [0., 0.9, 0.9]

    for i in range(n):
        hsv[0] = np.random.uniform()
        rgb = np.array(colorsys.hsv_to_rgb(*hsv))
        rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        yield webcolors.rgb_to_hex(rgb)

def fake_process_map(fn, iterable, max_workers=1):

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        result = executor.map(fn, iterable)
    return result

def optimize_handler(args):

    import itertools

    try:
        from tqdm.contrib.concurrent import process_map
    except ImportError:
        process_map = fake_process_map

    if len(args.colors) > 0:
        colors = args.colors
    else:
        colors = list(random_colors(n=args.pool))

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

    task = dict(sequential=sequential_task, diverging=diverging_task,
        cyclic=cyclic_task)[args.kind]

    for r in rvals:
        allcolors = list(itertools.permutations(colors, r=r))
        allparams.extend(process_map(task, allcolors, max_workers=args.nproc))

    jdata = json.dumps(dict(kind=args.kind, optimized=allparams))
    if args.save:
        with open(args.save, 'w') as f:
            f.write(jdata)
    else:
        print(jdata)

def make_preview_figure(args):

    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    from matplotlib.gridspec import GridSpec
    from .sandman import cmap_from_params

    w, h = 8, 0.5
    rows, cols = args.rows, args.cols
    nmaps = rows * cols

    with open(args.input, 'r') as f:
        data = json.loads(f.read())

    opt_data = sorted(data['optimized'], key=lambda p: p['score'])
    opt_data = opt_data[:nmaps]

    norm = colors.Normalize(vmin=0, vmax=w)
    cmaps = list(map(lambda p: cmap_from_params(p['name'],
        p['colors'], p['params'], kind=data['kind']), opt_data))

    X = np.linspace(0, w, 250)
    Y = np.linspace(0, h, 5)
    X, Y = np.meshgrid(X, Y, indexing='ij')

    fig = plt.figure(figsize=(w*cols, h*rows), constrained_layout=True)
    gs = GridSpec(rows, cols, figure=fig)
    axs = np.ravel(gs.subplots())

    for i, cmap in enumerate(cmaps):
        axs[i].pcolormesh(X, Y, X, cmap=cmap, norm=norm)
        axs[i].xaxis.set_ticks([])
        axs[i].yaxis.set_ticks([])
        axs[i].xaxis.set_ticklabels([])
        axs[i].yaxis.set_ticklabels([])
        axs[i].set_ylabel(cmap.name, rotation=0, ha='right')

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

if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='python -m sandman')
    subparsers = parser.add_subparsers(title='available sub-commands')

    subparser1 = subparsers.add_parser('optimize')
    subparser1.add_argument('colors', type=str, nargs='*')
    subparser1.add_argument('--save', type=str, default=None)
    subparser1.add_argument('--pool', type=int, default=5)
    subparser1.add_argument('--count', type=int, default=3,
        help='maximum number of colors to include in each map')
    subparser1.add_argument('--exact', action='store_true',
        help='use exactly --count colors, otherwise consider all ' \
             'combinations of lengths 3 to --count')
    subparser1.add_argument('--nproc', type=int, default=1,
        help='number of processors to use for working in parallel')
    subparser1.add_argument('--kind', type=str, default='sequential',
        choices=['sequential', 'diverging', 'cyclic'],
        help='type of colormaps to generate')
    subparser1.set_defaults(func=optimize_handler)

    subparser2 = subparsers.add_parser('preview')
    subparser2.add_argument('input', type=str)
    subparser2.add_argument('--rows', type=int, default=10)
    subparser2.add_argument('--cols', type=int, default=2)
    subparser2.add_argument('--save', type=str, default=None)
    subparser2.set_defaults(func=preview_handler)

    subparser3 = subparsers.add_parser('simulate')
    subparser3.add_argument('input', type=str)
    subparser3.add_argument('--rows', type=int, default=10)
    subparser3.add_argument('--cols', type=int, default=2)
    subparser3.add_argument('--save', type=str, default=None)
    subparser3.add_argument('--cvd', type=str,
        choices=['protanomaly', 'deuteranomaly', 'tritanomaly', 'achromatomaly'])
    subparser3.add_argument('--severity', type=int, default=100)
    subparser3.set_defaults(func=simulate_handler)

    args = parser.parse_args()

    args.func(args)
    #except AttributeError:
    #    parser.print_help()

# vim: set ft=python:
