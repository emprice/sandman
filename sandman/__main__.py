import json
import argparse
import numpy as np

def task(colors):

    from .sandman import find_optimal_params

    params, score = find_optimal_params(colors,
        min_intensity=30, max_intensity=90)
    return dict(params=list(params), colors=colors, score=score)

def random_colors(n=1):

    import colorsys
    import webcolors

    hsv = np.ones((3,))

    for _ in range(n):
        hsv[0] = np.random.uniform()
        hsv[1] = 0.9
        rgb = np.array(colorsys.hsv_to_rgb(*hsv))
        rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        yield webcolors.rgb_to_hex(rgb)

def fake_process_map(fn, iterable, max_workers=1):

    with ProcessPoolExecutor() as executor:
        result = executor.map(fn, iterable)
    return result

def optimize_handler(args):

    import itertools

    try:
        from tqdm.contrib.concurrent import process_map
    except ImportError:
        process_map = fake_process_map

    if len(args.colors) == 1 and args.colors[0] == 'random':
        colors = list(random_colors(n=args.pool))
    else:
        colors = args.colors

    if len(colors) < 3:
        raise ValueError('need at least 3 colors')

    rvals = [args.count] if args.exact else range(3, args.count + 1)
    allparams = list()

    for r in rvals:
        allcolors = list(itertools.permutations(colors, r=args.count))
        allparams.extend(process_map(task, allcolors, max_workers=args.nproc))

    with open(args.output, 'w') as f:
        f.write(json.dumps(allparams))

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
        allparams = json.loads(f.read())

    allparams = sorted(allparams, key=lambda p: p['score'])
    allparams = allparams[:nmaps]

    cmaps = list(map(lambda p: cmap_from_params('custom', p['colors'], p['params']), allparams))

    X = np.linspace(0, w, 250)
    Y = np.linspace(0, h, 5)
    X, Y = np.meshgrid(X, Y, indexing='ij')

    fig = plt.figure(figsize=(w*cols, h*rows), constrained_layout=True)
    gs = GridSpec(rows, cols, figure=fig)
    axs = np.ravel(gs.subplots())

    for i, cmap in enumerate(cmaps):
        axs[i].pcolormesh(X, Y, X, cmap=cmap)
        axs[i].xaxis.set_ticks([])
        axs[i].yaxis.set_ticks([])
        axs[i].xaxis.set_ticklabels([])
        axs[i].yaxis.set_ticklabels([])

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

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    subparser1 = subparsers.add_parser('optimize')
    subparser1.add_argument('colors', type=str, nargs='+')
    subparser1.add_argument('output', type=str)
    subparser1.add_argument('--pool', type=int, default=5)
    subparser1.add_argument('--count', type=int, default=3)
    subparser1.add_argument('--exact', action='store_true')
    subparser1.add_argument('--nproc', type=int, default=1)
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
    subparser3.add_argument('--severity', type=int, default=0)
    subparser3.set_defaults(func=simulate_handler)

    args = parser.parse_args()
    args.func(args)

# vim: set ft=python:
